from dataclasses import dataclass

import numpy as np

from .params import PARAMS, PARAM_GROUPS

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
    Cross-validate config["FIDUCIAL VALUES"] and config["PRIORS"]["params"]
    against params.PARAMS (every one of the 50 canonical names must appear in
    both, no extras — raises ValueError naming the mismatch otherwise), and
    resolve them into PARAMS-ordered, name-independent structures.

    Returns
    -------
    fiducial       : list[float], len(PARAMS), in PARAMS order.
    specs          : list[PriorSpec], len(PARAMS), in PARAMS order.
    varied_names   : list[str]  — PARAMS-order subset with kind != "fixed".
    varied_indices : list[int]  — their positions in PARAMS.
    """
    fiducial_cfg = config["FIDUCIAL VALUES"]
    priors_params = config["PRIORS"]["params"]

    params_set = set(PARAMS)
    missing_fid = params_set - set(fiducial_cfg)
    extra_fid   = set(fiducial_cfg) - params_set
    if missing_fid or extra_fid:
        raise ValueError(
            "FIDUCIAL VALUES does not match params.PARAMS exactly: "
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


# Hard-coded per-parameter sigma_scale override for D_i (photo-z shift)
# parameters, calibrated at PRIORS.sigma_scale=5 and rescaled by scale/5 in
# apply_fisher_bounds below (so raising/lowering PRIORS.sigma_scale still
# scales D_i proportionally, instead of pinning it to a fixed absolute
# value) -- Fisher's marginalized sigma badly over-predicts where D_i's true
# posterior actually has support (near-total linear-order degeneracy among
# the D_i, largely broken in the true non-Gaussian posterior; see
# prior_setup_Di.png), so simulations drawn at the configured scale are
# mostly wasted far outside the relevant region. Tighter, hand-tuned scales
# here concentrate the budget where it matters instead.
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
    Return a NEW specs list (does not mutate input) where every entry at
    varied_indices has (lower, upper) overwritten to
    (fiducial[i] - p_scale*sigma, fiducial[i] + p_scale*sigma), where p_scale
    is `scale` for every parameter except D_i, which uses its own hard-coded
    override (_DI_SIGMA_SCALE_OVERRIDE, rescaled by scale/5) instead — for "uniform" entries this
    replaces the prior's bounds entirely; for "normal" entries it restricts
    the SAME Normal(mean, sigma) density to that window (a truncated normal —
    see simulator.PriorSampler/_TruncatedNormalRejection), leaving mean/sigma
    untouched, so the density shape is unchanged, only its support is
    narrowed to where the posterior actually has mass. "fixed" entries pass
    through unchanged. sigmas[k] must correspond to varied_indices[k] (as
    returned together by fisher.run_fisher).
    """
    new_specs = list(specs)
    for k, i in enumerate(varied_indices):
        spec = specs[i]
        sigma = sigmas[k]
        di_override = _DI_SIGMA_SCALE_OVERRIDE.get(spec.name)
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
    Load an Finv saved by fisher.run_fisher (an .npz with keys Finv,
    varied_indices, varied_names) and return sigma = sqrt(diag(Finv)),
    validating that the saved varied_names exactly match `varied_names`
    (raises ValueError on mismatch — e.g. if PRIORS.params's fixed/varied
    flags changed since Fisher was last run, this Finv is stale and must be
    recomputed via run_fisher/generate_observation).
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


def resolve_inference_params(config):
    """
    Resolve config["TRAINING"]["params_to_infer"] (default "COSMO") to (names,
    indices) into the full PARAMS vector, validated against the varied
    parameter set derived from resolve_priors(config).

    Raises
    ------
    ValueError — naming any resolved parameter that is FIXED per
    config["PRIORS"] (there is no prior/posterior to infer over a fixed
    value).
    """
    _, _, varied_names, _ = resolve_priors(config)
    varied_set = set(varied_names)

    param_spec = config.get("TRAINING", {}).get("params_to_infer", "COSMO")
    if isinstance(param_spec, str):
        if param_spec not in PARAM_GROUPS:
            raise ValueError(
                f"Unknown param group {param_spec!r}. Choose from: {list(PARAM_GROUPS)}"
            )
        names = PARAM_GROUPS[param_spec]
    else:
        names = list(param_spec)

    fixed_selected = [name for name in names if name not in varied_set]
    if fixed_selected:
        raise ValueError(
            f"TRAINING.params_to_infer selects parameter(s) marked 'fixed' in PRIORS: "
            f"{fixed_selected} — a fixed parameter has no prior/posterior to infer."
        )

    indices = [PARAMS.index(name) for name in names]
    return names, indices
