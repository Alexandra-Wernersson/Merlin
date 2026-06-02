import numpy as np

# Redshift grid used by all tracers
ZS = np.linspace(1e-4, 3, 100)

# Parameter name lists — order must match the config [FIDUCIAL VALUES] section
COSMO_PARAMS       = ["H0", "Omega_b0", "Omega_cdm0", "ns", "ln10^{10}A_s"]
IA_PARAMS          = ["A_IA", "eta_IA"]
GAL_BIAS_PARAMS    = [f"b_g{i}"   for i in range(1, 5)]
MAG_BIAS_PARAMS    = [f"b_mag{i}" for i in range(1, 14)]
SHEAR_CALIB_PARAMS = [f"m_{i}"    for i in range(1, 14)]
PHOTOZ_PARAMS      = [f"D_{i}"    for i in range(1, 14)]

PARAMS        = COSMO_PARAMS + IA_PARAMS + GAL_BIAS_PARAMS + MAG_BIAS_PARAMS + SHEAR_CALIB_PARAMS + PHOTOZ_PARAMS
N_COSMO       = len(COSMO_PARAMS)
NUISANCE_KEYS = PARAMS[N_COSMO:]

# Named groups usable in config under INFERENCE.params
PARAM_GROUPS = {
    "COSMO":       COSMO_PARAMS,
    "IA":          IA_PARAMS,
    "GAL_BIAS":    GAL_BIAS_PARAMS,
    "MAG_BIAS":    MAG_BIAS_PARAMS,
    "SHEAR_CALIB": SHEAR_CALIB_PARAMS,
    "PHOTOZ":      PHOTOZ_PARAMS,
    "NUISANCE":    NUISANCE_KEYS,
    "ALL":         PARAMS,
}

# LaTeX labels for corner plots
PARAM_LABELS = {
    "H0":              r"$H_0$",
    "Omega_b0":        r"$\Omega_b$",
    "Omega_cdm0":      r"$\Omega_{\rm cdm}$",
    "ns":              r"$n_s$",
    "ln10^{10}A_s":    r"$\ln(10^{10}A_s)$",
    "A_IA":            r"$A_{\rm IA}$",
    "eta_IA":          r"$\eta_{\rm IA}$",
    **{f"b_g{i}":      rf"$b_{{g{i}}}$"   for i in range(1, 5)},
    **{f"b_mag{i}":    rf"$b_{{\rm mag,{i}}}$" for i in range(1, 14)},
    **{f"m_{i}":       rf"$m_{i}$"        for i in range(1, 14)},
    **{f"D_{i}":       rf"$\Delta z_{i}$" for i in range(1, 14)},
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
}


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
