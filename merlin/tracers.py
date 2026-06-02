import numpy as np
from scipy import integrate

from . import _compat  # noqa: F401 — sets up cloelib path
from .params import ZS


def load_dndz(nz_path):
    """Load and normalise n(z) distributions from a FITS file."""
    import euclidlib as el
    z_nz, nz_dict = el.photo.redshift_distributions(nz_path)
    nz_array = np.vstack([nz / integrate.trapezoid(nz, z_nz) for nz in nz_dict.values()])
    return np.array([np.interp(ZS, z_nz, nz) for nz in nz_array])


def get_position_tracer(nuis_params, n_bins, perturbations, dndz):
    from cloelib.observables.photo import PositionsTracer
    pos_nuisance = {}
    for i in range(4):
        pos_nuisance[f"b1_photo_poly{i}"] = nuis_params[f"b_g{i+1}"]
    for i in range(n_bins):
        pos_nuisance[f"magnification_bias_{i+1}"] = nuis_params[f"b_mag{i+1}"]
    for i in range(n_bins):
        pos_nuisance[f"dz_pos_{i+1}"] = nuis_params[f"D_{i+1}"]
    return PositionsTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=ZS,
        galaxy_bias_model="poly",
        nuisance_params=pos_nuisance,
    )


def get_shear_tracer(nuis_params, n_bins, perturbations, dndz):
    from cloelib.observables.photo import ShearTracer
    she_nuisance = {
        "AIA":   nuis_params["A_IA"],
        "EtaIA": nuis_params["eta_IA"],
        "CIA":   nuis_params["CIA"],
    }
    for i in range(n_bins):
        she_nuisance[f"multiplicative_bias_{i+1}"] = nuis_params[f"m_{i+1}"]
    for i in range(n_bins):
        she_nuisance[f"dz_shear_{i+1}"] = nuis_params[f"D_{i+1}"]
    return ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=ZS,
        nuisance_params=she_nuisance,
    )
