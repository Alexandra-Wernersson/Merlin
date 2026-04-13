import numpy as np
# Euclid imports
import euclidlib as el
from scipy import integrate
import sys
import importlib

# 1. Remove pip-installed cloelib if already loaded
if "cloelib" in sys.modules:
    del sys.modules["cloelib"]

# 2. Force Python to search your local project first
sys.path.insert(0, "/home/awernersson/projects")

# 3. Now import local cloelib
import cloelib
print("Loaded from:", cloelib.__file__)

import cloelib
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations, CAMBNonLinearPerturbations
from cloelib.cosmology.HMcode2020Emu_cosmology import HMemuLinearPerturbations, HMemuNonLinearPerturbations
from cloelib.observables.photo import ShearTracer, PositionsTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
from cloelib.summary_statistics.angular_correlation_function_wigner import AngularCorrelationFunctionWigner

zs = np.linspace(1e-4, 3, 100)
nz_example = '/home/awernersson/projects/playground/tutorials/observables/nzTabSPV3_euclidlib_format.fits'
z_nz, nz_example = el.photo.redshift_distributions(nz_example)

def normalize_and_resample(nz_dict, z_grid, z_target):
    nz_array = np.vstack([nz / integrate.trapezoid(nz, z_grid) for nz in nz_dict.values()])
    return np.array([np.interp(z_target, z_grid, nz) for nz in nz_array])

my_dndz_pos_norm = normalize_and_resample(nz_example, z_nz, zs)
my_dndz_she_norm = normalize_and_resample(nz_example, z_nz, zs)

def get_sigmas_bounds(fiducial:list, finv_file: str, N_pars: int):
    Finv = np.load(finv_file)
    sigmas = np.sqrt(np.diag(Finv))

    lower_bounds = [fiducial[i] - 6 * sigmas[i] for i in range(N_pars)]
    upper_bounds = [fiducial[i] + 6 * sigmas[i] for i in range(N_pars)]
    return sigmas, lower_bounds, upper_bounds

def get_positiontracer(nuis_params, n_pos_bins, perturbations):
    pos_nuisance_params = {}

    # 1. Poly bias (b_g1, b_g2, b_g3, b_g4)
    for i in range(4):
        pos_nuisance_params[f"b1_photo_poly{i}"] = nuis_params[f"b_g{i+1}"]
    
    # 2. Magnification bias (b_mag1, ..., b_mag13)
    for i in range(n_pos_bins):
        pos_nuisance_params[f"magnification_bias_{i+1}"] = nuis_params[f"b_mag{i+1}"]

    # 3. Photo-z shift (D_1, ..., D_13) - FIXED KEY NAME
    for i in range(n_pos_bins):
        pos_nuisance_params[f"dz_pos_{i+1}"] = nuis_params[f"D_{i+1}"]

    tracer_pos = PositionsTracer(
        perturbations=perturbations,
        dndz=my_dndz_pos_norm,
        z=zs,
        galaxy_bias_model="poly",
        nuisance_params=pos_nuisance_params,
    )

    return tracer_pos


def get_sheartracer(nuis_params, n_she_bins, perturbations):
    she_nuisance_params = {}
    
    # 1. Intrinsic alignment
    she_nuisance_params["AIA"] = nuis_params["A_IA"]
    she_nuisance_params["EtaIA"] = nuis_params["eta_IA"]
    she_nuisance_params["CIA"] = nuis_params["CIA"]  # Note: beta_IA maps to CIA
    
    # 2. Multiplicative bias (m_1, ..., m_13)
    for i in range(n_she_bins):
        she_nuisance_params[f"multiplicative_bias_{i+1}"] = nuis_params[f"m_{i+1}"]
    
    # 3. Photo-z shift (D_1, ..., D_13) - SAME AS POSITIONS
    for i in range(n_she_bins):
        she_nuisance_params[f"dz_shear_{i+1}"] = nuis_params[f"D_{i+1}"]
    
    tracer_she = ShearTracer(
        perturbations=perturbations,
        dndz=my_dndz_she_norm,
        z=zs,
        nuisance_params=she_nuisance_params,
    )
    
    return tracer_she
