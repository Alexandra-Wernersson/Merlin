"""
Parameter definitions and global constants shared across merlin.
"""

import numpy as np

# Redshift grid used for tracer integration
ZS = np.linspace(1e-4, 3, 100)

# ── Cosmological parameters ─────────────────────────────────────────────────
COSMO_PARAMS = [
    "H0",
    "Omega_b0",
    "Omega_cdm0",
    "ns",
    "ln10^{10}A_s",
]

# ── Nuisance parameters ──────────────────────────────────────────────────────
IA_PARAMS = ["A_IA", "eta_IA"]
GAL_BIAS_PARAMS = [f"b_g{i}" for i in range(1, 5)]
MAG_BIAS_PARAMS = [f"b_mag{i}" for i in range(1, 14)]
SHEAR_CALIB_PARAMS = [f"m_{i}" for i in range(1, 14)]
PHOTOZ_PARAMS = [f"D_{i}" for i in range(1, 14)]

# ── Full parameter vector ────────────────────────────────────────────────────
PARAMS = (
    COSMO_PARAMS
    + IA_PARAMS
    + GAL_BIAS_PARAMS
    + MAG_BIAS_PARAMS
    + SHEAR_CALIB_PARAMS
    + PHOTOZ_PARAMS
)

N_COSMO = len(COSMO_PARAMS)
NUISANCE_KEYS = PARAMS[N_COSMO:]
