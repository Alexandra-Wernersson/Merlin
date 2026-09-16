from dataclasses import dataclass

import numpy as np

from .params import PARAMS, PARAM_GROUPS, resolve_derived_names

_KIND_ALIASES = {"fixed": "fixed", "uniform": "uniform", "normal": "normal", "norm": "normal"}


@dataclass(frozen=True)
class PriorSpec:
    name: str
    kind: str                  # "fixed" | "uniform" | "normal"
    lower: float = None        # uniform only
    upper: float = None        # uniform only
    mean: float = None         # normal only
    sigma: float = None        # normal only


def _parse_entry(name, entry):
    raw_kind = entry.get("type")
    if raw_kind not in _KIND_ALIASES:
        raise ValueError(
            f"PRIORS.params[{name!r}]: unknown type {raw_kind!r} "
            f"(expected one of {sorted(set(_KIND_ALIASES))})"
        )
    kind = _KIND_ALIASES[raw_kind]

    if kind == "fixed":
        return PriorSpec(name=name, kind="fixed")

    if kind == "uniform":
        lower, upper = entry.get("lower"), entry.get("upper")
        if lower is None or upper is None:
            raise ValueError(f"PRIORS.params[{name!r}]: uniform prior needs 'lower' and 'upper'")
        if not (lower < upper):
            raise ValueError(f"PRIORS.params[{name!r}]: lower ({lower}) must be < upper ({upper})")
        return PriorSpec(name=name, kind="uniform", lower=float(lower), upper=float(upper))

    mean, sigma = entry.get("mean"), entry.get("sigma")
    if mean is None or sigma is None:
        raise ValueError(f"PRIORS.params[{name!r}]: normal prior needs 'mean' and 'sigma'")
    if not (sigma > 0):
        raise ValueError(f"PRIORS.params[{name!r}]: sigma ({sigma}) must be > 0")
    return PriorSpec(name=name, kind="normal", mean=float(mean), sigma=float(sigma))


def resolve_priors(config):
    """
    Cross-validate config["FIDUCIAL"] and config["PRIORS"]["params"] against
    params.PARAMS (every name must appear in both) and resolve into
    PARAMS-ordered structures.

    Returns
    -------
    fiducial       : list[float], PARAMS order.
    specs          : list[PriorSpec], PARAMS order.
    varied_names   : list[str] — non-fixed subset, PARAMS order.
    varied_indices : list[int] — their positions in PARAMS.
    """
    fiducial_cfg = config["FIDUCIAL"]
    priors_params = config["PRIORS"]["params"]

    params_set = set(PARAMS)
    missing_fid = params_set - set(fiducial_cfg)
    extra_fid   = set(fiducial_cfg) - params_set
    if missing_fid or extra_fid:
        raise ValueError(
            "FIDUCIAL does not match params.PARAMS exactly: "
            f"missing={sorted(missing_fid)}, unexpected={sorted(extra_fid)}"
        )
    missing_pr = params_set - set(priors_params)
    extra_pr   = set(priors_params) - params_set
    if missing_pr or extra_pr:
        raise ValueError(
            "PRIORS.params does not match params.PARAMS exactly: "
            f"missing={sorted(missing_pr)}, unexpected={sorted(extra_pr)}"
        )

    fiducial = [float(fiducial_cfg[name]) for name in PARAMS]
    specs    = [_parse_entry(name, priors_params[name]) for name in PARAMS]

    varied_names   = [s.name for s in specs if s.kind != "fixed"]
    varied_indices = [i for i, s in enumerate(specs) if s.kind != "fixed"]

    return fiducial, specs, varied_names, varied_indices


# Per-parameter sigma_scale overrides for D_i (photo-z shift) parameters,
# calibrated at PRIORS.sigma_scale=5 and rescaled by scale/5 in
# apply_fisher_bounds so they still scale with PRIORS.sigma_scale. Fisher's
# marginalized sigma over-predicts D_i's true posterior support (near-total
# linear-order degeneracy among the D_i, broken in the true non-Gaussian
# posterior); these tighter, hand-tuned scales concentrate simulations where
# the posterior actually has mass.
_DI_SIGMA_SCALE_OVERRIDE = {
    **{f"D_{i}": 2.0 for i in (1,)},
    **{f"D_{i}": 1.4 for i in (2, 3)},
    **{f"D_{i}": 0.65 for i in (4, 5, 6)},
    **{f"D_{i}": 0.9 for i in (7, 8)},
    **{f"D_{i}": 1.9 for i in (9,)},
    **{f"D_{i}": 2.5 for i in (10,)},
    **{f"D_{i}": 3.8 for i in (11,)},
    **{f"D_{i}": 5.0 for i in (12, 13)},
}


def apply_fisher_bounds(specs, fiducial, varied_indices, sigmas, scale):
    """
    Return a new specs list (input unchanged) with (lower, upper) at each
    varied_indices entry set to fiducial[i] +/- p_scale*sigma, where p_scale
    is `scale` (or the D_i override, rescaled by scale/5). "uniform" entries
    get these as their new bounds; "normal" entries keep mean/sigma but get
    truncated to this window (see PriorSampler/_TruncatedNormalRejection).
    "fixed" entries pass through unchanged. sigmas[k] must correspond to
    varied_indices[k] (as returned by fisher.run_fisher).
    """
    new_specs = list(specs)
    for k, i in enumerate(varied_indices):
        spec = specs[i]
        sigma = sigmas[k]
        di_override = _DI_SIGMA_SCALE_OVERRIDE.get(spec.name)
        # Widening this (e.g. x2 or more) could help reduce the ns/logAs posterior shift relative to Nautilus 
        p_scale = di_override * (scale / 5) if di_override is not None else scale
        lower, upper = fiducial[i] - p_scale * sigma, fiducial[i] + p_scale * sigma
        if spec.kind == "uniform":
            new_specs[i] = PriorSpec(name=spec.name, kind="uniform", lower=lower, upper=upper)
        elif spec.kind == "normal":
            new_specs[i] = PriorSpec(
                name=spec.name, kind="normal", mean=spec.mean, sigma=spec.sigma,
                lower=lower, upper=upper,
            )
    return new_specs


def load_fisher_sigmas(finv_file, varied_indices, varied_names):
    """
    Load an Finv saved by fisher.run_fisher and return sigma =
    sqrt(diag(Finv)), validating that its saved varied_names match
    `varied_names` (raises ValueError if stale — e.g. PRIORS's fixed/varied
    flags changed since Fisher was last run; re-run via
    observation.generate_observation or fisher.run_fisher).
    """
    data = np.load(finv_file, allow_pickle=False)
    saved_names = list(data["varied_names"])
    if saved_names != list(varied_names):
        raise ValueError(
            f"Fisher matrix at {finv_file!r} is stale: it was computed for "
            f"varied parameters {saved_names}, but the current PRIORS config "
            f"has varied parameters {list(varied_names)}. Re-run Fisher "
            "(observation.generate_observation or fisher.run_fisher)."
        )
    Finv = data["Finv"]
    return np.sqrt(np.diag(Finv))


# Default for CLOELIB_SETTINGS.sigma_scale_derived (independent of
# PRIORS.sigma_scale, which only governs the COSMO-param box) -- see
# CLOELIB_SETTINGS.restrict_prior_for_derived. Used when the config key is
# unset, and as the fallback for any direct/programmatic caller.
DERIVED_REJECTION_SIGMA_SCALE = 7.0


def load_derived_fisher_sigmas(finv_file, derived_names):
    """
    Load delta-method Fisher sigmas + fiducial values for `derived_names`,
    saved by fisher.run_fisher when restrict_prior_for_derived is on.
    Raises ValueError if finv_file has no saved derived-quantity info (Fisher
    ran before the flag was enabled), or is missing any of `derived_names`
    (add_derived changed since) -- re-run Fisher either way.
    """
    data = np.load(finv_file, allow_pickle=False)
    if "derived_names" not in data:
        raise ValueError(
            f"Fisher matrix at {finv_file!r} has no saved derived-quantity "
            "sigmas -- restrict_prior_for_derived is on but Fisher was "
            "computed before it was enabled. Re-run Fisher."
        )
    saved_names = list(data["derived_names"])
    missing = [n for n in derived_names if n not in saved_names]
    if missing:
        raise ValueError(
            f"Fisher matrix at {finv_file!r} has derived sigmas for "
            f"{saved_names}, missing {missing} -- add_derived changed since "
            "Fisher last ran. Re-run Fisher."
        )
    sigma = dict(zip(saved_names, data["derived_fisher_sigma"]))
    fid   = dict(zip(saved_names, data["derived_fiducial"]))
    return ({n: float(sigma[n]) for n in derived_names},
            {n: float(fid[n]) for n in derived_names})


def load_derived_fisher_box(finv_file, box_names, scale=DERIVED_REJECTION_SIGMA_SCALE):
    """{name: (lower, upper)} = fiducial +/- scale*sigma_Fisher for box_names."""
    sigma, fid = load_derived_fisher_sigmas(finv_file, box_names)
    return {n: (fid[n] - scale * sigma[n], fid[n] + scale * sigma[n]) for n in box_names}


def resolve_inference_params(config):
    """
    Resolve config["TRAINING"]["params_to_infer"] (default "COSMO") to
    (names, indices), validated against the varied parameter set from
    resolve_priors(config).

    Names resolve against PARAMS first, then against this config's own
    CLOELIB_SETTINGS.add_derived subset (sigma8/Omega_m/S8 — computed, not
    sampled) — not the full params.DERIVED_PARAMS registry, since a store's
    "derived" array only has as many columns as its own add_derived asked
    for, at positions matching that subset's order, not DERIVED_PARAMS's
    canonical positions. Indices are returned in a concatenated index space
    so network.py can gather from torch.cat([z, derived], dim=-1) directly:
    PARAMS-space indices are unchanged; derived-space indices are offset by
    len(PARAMS), ordered per this config's add_derived. This is the single
    source of truth for that offset convention.

    Raises
    ------
    ValueError — a resolved PARAMS-space name is marked fixed in PRIORS
    (nothing to infer over a fixed value; doesn't apply to derived names,
    which have no PRIORS entry to be fixed relative to).
    ValueError — a resolved name is in neither PARAMS nor this config's
    CLOELIB_SETTINGS.add_derived.
    """
    _, _, varied_names, _ = resolve_priors(config)
    varied_set = set(varied_names)
    store_derived_names = resolve_derived_names(config.get("CLOELIB_SETTINGS", {}).get("add_derived"))

    param_spec = config.get("TRAINING", {}).get("params_to_infer", "COSMO")
    if isinstance(param_spec, str):
        if param_spec not in PARAM_GROUPS:
            raise ValueError(
                f"Unknown param group {param_spec!r}. Choose from: {list(PARAM_GROUPS)}"
            )
        names = PARAM_GROUPS[param_spec]
    else:
        names = list(param_spec)

    unknown = [
        name for name in names
        if name not in PARAMS and name not in store_derived_names
    ]
    if unknown:
        raise ValueError(
            f"TRAINING.params_to_infer names not found in params.PARAMS or "
            f"this config's CLOELIB_SETTINGS.add_derived ({list(store_derived_names)}): "
            f"{unknown}"
        )

    fixed_selected = [
        name for name in names
        if name in PARAMS and name not in store_derived_names and name not in varied_set
    ]
    if fixed_selected:
        raise ValueError(
            f"TRAINING.params_to_infer selects parameter(s) marked 'fixed' in PRIORS: "
            f"{fixed_selected} — a fixed parameter has no prior/posterior to infer."
        )

    indices = [
        PARAMS.index(name) if name in PARAMS else len(PARAMS) + store_derived_names.index(name)
        for name in names
    ]
    return names, indices
