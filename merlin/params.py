import numpy as np

# Redshift grid used by all tracers
ZS = np.linspace(1e-4, 3, 100)

# Parameter name lists — order must match the config [FIDUCIAL] section
COSMO_PARAMS       = ["H0", "Omega_b0", "Omega_cdm0", "ns", "ln10^{10}A_s"]
# Background/perturbation params cloelib also takes but aren't in the
# network-inferred COSMO group (see PARAM_GROUPS).
EXTRA_COSMO_PARAMS = ["w0", "wa", "Omega_k0", "mnu", "gamma_MG", "log10TAGN"]
IA_PARAMS          = ["A_IA", "eta_IA"]
GAL_BIAS_PARAMS    = [f"b_g{i}"   for i in range(1, 5)]
MAG_BIAS_PARAMS    = [f"b_mag{i}" for i in range(1, 14)]
SHEAR_CALIB_PARAMS = [f"m_{i}"    for i in range(1, 14)]
PHOTOZ_PARAMS      = [f"D_{i}"    for i in range(1, 14)]
WIDTH_POS_PARAMS   = [f"width_pos_{i}"   for i in range(1, 14)]
WIDTH_SHEAR_PARAMS = [f"width_shear_{i}" for i in range(1, 14)]

PARAMS        = (COSMO_PARAMS + EXTRA_COSMO_PARAMS + IA_PARAMS + GAL_BIAS_PARAMS + MAG_BIAS_PARAMS
                  + SHEAR_CALIB_PARAMS + PHOTOZ_PARAMS + WIDTH_POS_PARAMS + WIDTH_SHEAR_PARAMS)
N_COSMO       = len(COSMO_PARAMS)

# Derived (computed, not sampled) quantities -- deterministic functions of
# the sampled cosmology (see simulator.Simulator, CLOELIB_SETTINGS.add_derived).
# Kept separate from PARAMS since PriorSampler requires every PARAMS entry to
# have a FIDUCIAL/PRIORS.params entry, which derived quantities lack.
DERIVED_PARAMS = ["sigma8", "Omega_m", "S8"]   # canonical order
# Everything after COSMO, positionally matching z[N_COSMO:] in
# Simulator.get_sample_Cls.
NUISANCE_KEYS = PARAMS[N_COSMO:]

# Named groups usable in config under TRAINING.params_to_infer
PARAM_GROUPS = {
    "COSMO":       COSMO_PARAMS,
    "EXTRA_COSMO": EXTRA_COSMO_PARAMS,
    "IA":          IA_PARAMS,
    "GAL_BIAS":    GAL_BIAS_PARAMS,
    "MAG_BIAS":    MAG_BIAS_PARAMS,
    "SHEAR_CALIB": SHEAR_CALIB_PARAMS,
    "PHOTOZ":      PHOTOZ_PARAMS,
    "WIDTH_POS":   WIDTH_POS_PARAMS,
    "WIDTH_SHEAR": WIDTH_SHEAR_PARAMS,
    "NUISANCE":    NUISANCE_KEYS,
    "ALL":         PARAMS,
    "DERIVED":     DERIVED_PARAMS,
}

# LaTeX labels for corner plots
PARAM_LABELS = {
    "H0":              r"$H_0$",
    "Omega_b0":        r"$\Omega_b$",
    "Omega_cdm0":      r"$\Omega_{\rm cdm}$",
    "ns":              r"$n_s$",
    "ln10^{10}A_s":    r"$\ln(10^{10}A_s)$",
    "w0":              r"$w_0$",
    "wa":              r"$w_a$",
    "Omega_k0":        r"$\Omega_k$",
    "mnu":             r"$m_\nu$",
    "gamma_MG":        r"$\gamma_{\rm MG}$",
    "log10TAGN":       r"$\log_{10}T_{\rm AGN}$",
    "A_IA":            r"$A_{\rm IA}$",
    "eta_IA":          r"$\eta_{\rm IA}$",
    **{f"b_g{i}":      rf"$b_{{g{i}}}$"   for i in range(1, 5)},
    **{f"b_mag{i}":    rf"$b_{{\rm mag,{i}}}$" for i in range(1, 14)},
    **{f"m_{i}":       rf"$m_{{{i}}}$"        for i in range(1, 14)},
    **{f"D_{i}":       rf"$\Delta z_{{{i}}}$" for i in range(1, 14)},
    **{f"width_pos_{i}":   rf"$w^{{\rm pos}}_{{{i}}}$" for i in range(1, 14)},
    **{f"width_shear_{i}": rf"$w^{{\rm she}}_{{{i}}}$" for i in range(1, 14)},
    "sigma8":          r"$\sigma_8$",
    "Omega_m":         r"$\Omega_m$",
    "S8":              r"$S_8$",
}


# Mapping from merlin param names → MCMC chain key names
MCMC_KEY_MAP = {
    "H0":           "H0",
    "Omega_b0":     "Omega_b0",
    "Omega_cdm0":   "Omega_cdm0",
    "ns":           "ns",
    "ln10^{10}A_s": "logAs",
    "A_IA":         "AIA",
    "eta_IA":       "EtaIA",
    **{f"b_g{i}":   f"b1_photo_poly{i-1}" for i in range(1, 5)},
    **{f"b_mag{i}": f"magnification_bias_{i}" for i in range(1, 14)},
    **{f"m_{i}":    f"multiplicative_bias_{i}" for i in range(1, 14)},
    **{f"D_{i}":    f"dz_pos_{i}" for i in range(1, 14)},
    # sigma8 sits under its own top-level "derived" key in Nautilus chain
    # .npz files, not inside "chain" (see plotting.load_mcmc_overlay).
    # Omega_m/S8 have no chain column either -- plotting.load_mcmc_overlay
    # computes and injects them under these same key names.
    "sigma8":       "sigma8_0",
    "Omega_m":      "Omega_m",
    "S8":           "S8",
}


def resolve_derived_names(add_derived):
    """
    Parse CLOELIB_SETTINGS.add_derived (null / name / list of names) into an
    ordered subset of DERIVED_PARAMS, always in DERIVED_PARAMS' canonical
    order so the "derived" graph node's columns match what
    priors.resolve_inference_params expects.

    Lives here rather than simulator.py/priors.py to avoid an import cycle
    between them.
    """
    if not add_derived:
        return ()
    requested = {add_derived} if isinstance(add_derived, str) else set(add_derived)
    unknown = requested - set(DERIVED_PARAMS)
    if unknown:
        raise ValueError(
            f"CLOELIB_SETTINGS.add_derived names not in params.DERIVED_PARAMS: "
            f"{sorted(unknown)} — choose from {DERIVED_PARAMS}"
        )
    return tuple(name for name in DERIVED_PARAMS if name in requested)


def resolve_derived_box_names(derived_names):
    """
    Business rule for CLOELIB_SETTINGS.restrict_prior_for_derived: which of
    add_derived's requested quantities gate rejection-sampling acceptance
    in simulator.Simulator's derived_box. If all three of sigma8/Omega_m/S8
    are requested, the box is on Omega_m and S8 only (sigma8 is still
    computed/stored in the "derived" array, just doesn't gate acceptance);
    for any other 1- or 2-name subset, the box is on exactly what's requested.
    """
    derived_names = tuple(derived_names)
    if set(derived_names) == {"sigma8", "Omega_m", "S8"}:
        return ("Omega_m", "S8")
    return derived_names


def resolve_params(param_spec):
    """
    Resolve a group name or list of parameter names to (names, indices).

    Parameters
    ----------
    param_spec : str or list
        A group key from PARAM_GROUPS (e.g. "COSMO") or an explicit list
        of parameter names (e.g. ["H0", "A_IA"]).

    Returns
    -------
    names   : list of str
    indices : list of int  — positions in the full PARAMS vector
    """
    if isinstance(param_spec, str):
        if param_spec not in PARAM_GROUPS:
            raise ValueError(
                f"Unknown param group '{param_spec}'. "
                f"Choose from: {list(PARAM_GROUPS)}"
            )
        names = PARAM_GROUPS[param_spec]
    else:
        names = list(param_spec)

    indices = [PARAMS.index(name) for name in names]
    return names, indices
