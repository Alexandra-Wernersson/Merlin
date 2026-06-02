"""
Tracer construction helpers and redshift-distribution utilities.
"""

import numpy as np
from scipy import integrate

import euclidlib as el
from cloelib.observables.photo import ShearTracer, PositionsTracer

from .params import ZS


def normalize_and_resample(nz_dict, z_grid, z_target):
    """Normalise each n(z) slice and resample onto *z_target*."""
    nz_array = np.vstack(
        [nz / integrate.trapezoid(nz, z_grid) for nz in nz_dict.values()]
    )
    return np.array([np.interp(z_target, z_grid, nz) for nz in nz_array])


def load_dndz(nz_path):
    """Load n(z) distributions from an euclidlib-format FITS file."""
    z_nz, nz_dict = el.photo.redshift_distributions(nz_path)
    return normalize_and_resample(nz_dict, z_nz, ZS)


def get_positiontracer(nuis_params, n_pos_bins, perturbations, dndz):
    """Build a :class:`PositionsTracer` from nuisance parameters."""
    pos_nuisance_params = {}

    for i in range(4):
        pos_nuisance_params[f"b1_photo_poly{i}"] = nuis_params[f"b_g{i + 1}"]

    for i in range(n_pos_bins):
        pos_nuisance_params[f"magnification_bias_{i + 1}"] = nuis_params[f"b_mag{i + 1}"]

    for i in range(n_pos_bins):
        pos_nuisance_params[f"dz_pos_{i + 1}"] = nuis_params[f"D_{i + 1}"]

    return PositionsTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=ZS,
        galaxy_bias_model="poly",
        nuisance_params=pos_nuisance_params,
    )


def get_sheartracer(nuis_params, n_she_bins, perturbations, dndz):
    """Build a :class:`ShearTracer` from nuisance parameters."""
    she_nuisance_params = {}

    she_nuisance_params["AIA"] = nuis_params["A_IA"]
    she_nuisance_params["EtaIA"] = nuis_params["eta_IA"]
    she_nuisance_params["CIA"] = nuis_params["CIA"]

    for i in range(n_she_bins):
        she_nuisance_params[f"multiplicative_bias_{i + 1}"] = nuis_params[f"m_{i + 1}"]

    for i in range(n_she_bins):
        she_nuisance_params[f"dz_shear_{i + 1}"] = nuis_params[f"D_{i + 1}"]

    return ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=ZS,
        nuisance_params=she_nuisance_params,
    )
