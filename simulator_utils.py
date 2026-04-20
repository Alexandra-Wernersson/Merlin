import swyft
import configparser
import torch
from torch.distributions import Uniform, Normal
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import pickle
import psutil
from joblib import Parallel, delayed
import time
# Euclid imports
import euclidlib as el

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


from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations, CAMBNonLinearPerturbations
from cloelib.cosmology.HMcode2020Emu_cosmology import HMemuLinearPerturbations, HMemuNonLinearPerturbations
from cloelib.observables.photo import ShearTracer, PositionsTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
from cloelib.summary_statistics.angular_correlation_function_wigner import AngularCorrelationFunctionWigner

zs = np.linspace(1e-4, 3, 100)


def normalize_and_resample(nz_dict, z_grid, z_target):
    nz_array = np.vstack([nz / integrate.trapezoid(nz, z_grid) for nz in nz_dict.values()])
    return np.array([np.interp(z_target, z_grid, nz) for nz in nz_array])


def load_dndz(nz_path):
    z_nz, nz_dict = el.photo.redshift_distributions(nz_path)
    return normalize_and_resample(nz_dict, z_nz, zs)



# ============================================================
# Parameter definitions
# ============================================================

COSMO_PARAMS = [
    "H0",
    "Omega_b0",
    "Omega_cdm0",
    "ns",
    "ln10^{10}A_s",
]

IA_PARAMS = [
    "A_IA",
    "eta_IA",
]

GAL_BIAS_PARAMS = [f"b_g{i}" for i in range(1, 5)]

MAG_BIAS_PARAMS = [f"b_mag{i}" for i in range(1, 14)]

SHEAR_CALIB_PARAMS = [f"m_{i}" for i in range(1, 14)]

PHOTOZ_PARAMS = [f"D_{i}" for i in range(1, 14)]


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

class PriorSampler:

    def __init__(self, lower_bounds, upper_bounds, zmean):

        self.priors = []

        # uniform priors for cosmology + bias parameters
        for lo, hi in zip(lower_bounds, upper_bounds):
            self.priors.append(
                Uniform(
                    torch.tensor(lo),
                    torch.tensor(hi)
                )
            )

        # Gaussian priors for shear calibration and photo-z
        for i in range(len(zmean)*2):

            loc = torch.tensor(0.0)

            if i < len(zmean):
                scale = torch.tensor(0.01)

            else:
                scale = torch.tensor(
                    0.002 * (1.0 + zmean[i-len(zmean)])
                )

            self.priors.append(
                Normal(loc, scale)
            )

    def __call__(self, shape=()):

        samples = [
            p.sample(shape)
            for p in self.priors
        ]

        return torch.stack(samples, dim=-1).numpy()


class Simulator(swyft.Simulator):

    def __init__(
        self,
        fiducial,
        covmat,
        n_bins,
        lower_bounds,
        upper_bounds,
        zmean,
        ell_theory,
        dndz,
    ):

        super().__init__()

        # dtype consistency with float64 training
        self.transform_samples = swyft.to_numpy

        self.fiducial = fiducial
        self.n_bins = n_bins
        self.ells = ell_theory
        self.dndz = dndz

        self.sample_z = PriorSampler(
            lower_bounds,
            upper_bounds,
            zmean
        )

        # Cholesky of covariance
        self.Lfid = np.linalg.cholesky(covmat)

        # precompute key ordering
        self._build_keys()

    # --------------------------------------------------------

    def _build_keys(self):

        self.WL_keys = [
            ("SHE", "SHE", i, j)
            for i in range(1, self.n_bins+1)
            for j in range(i, self.n_bins+1)
        ]

        self.GG_keys = [
            ("POS", "POS", i, j)
            for i in range(1, self.n_bins+1)
            for j in range(i, self.n_bins+1)
        ]

        self.GGL_keys = [
            ("POS", "SHE", i, j)
            for i in range(1, self.n_bins+1)
            for j in range(1, self.n_bins+1)
        ]

        self.n_data = (
            len(self.WL_keys)
            + len(self.GGL_keys)
            + len(self.GG_keys)
        )

    # --------------------------------------------------------

    def generate_observation(self):

        return self.sample(
            conditions={"z": np.array(self.fiducial)}
        )

    # --------------------------------------------------------

    def get_sample_Cls(self, z):

        # --- cosmology ---
        background = CAMBBackground(

            H0=z[0],
            Omega_b0=z[1],
            Omega_cdm0=z[2],

            w0=-1,
            wa=0,
            Omega_k0=0.0,

            ns=z[3],
            As=np.exp(z[4]) / 1e10,

            mnu=0.077,
            gamma_MG=0.545,
            N_mnu=1
        )

        linear_perturbations = HMemuLinearPerturbations(
            background,
            zs
        )

        nonlinear_perturbations = HMemuNonLinearPerturbations(

            background,
            linear_perturbations,
            zs,
            log10TAGN=7.75
        )

        perturbations = nonlinear_perturbations

        # --- nuisance parameters ---
        nuisance = dict(
            zip(
                NUISANCE_KEYS,
                z[N_COSMO:]
            )
        )

        nuisance["CIA"] = 0.0134

        tracer_pos = get_positiontracer(
            nuisance,
            self.n_bins,
            perturbations,
            self.dndz,
        )

        tracer_she = get_sheartracer(
            nuisance,
            self.n_bins,
            perturbations,
            self.dndz,
        )

        # --- compute Cls ---
        cls_sheshe = AngularTwoPoint(
            tracer_she,
            tracer_she
        ).get_Cl(
            self.ells,
            0,
            perturbations.k
        )

        cls_posshe = AngularTwoPoint(
            tracer_pos,
            tracer_she
        ).get_Cl(
            self.ells,
            0,
            perturbations.k
        )

        cls_pospos = AngularTwoPoint(
            tracer_pos,
            tracer_pos
        ).get_Cl(
            self.ells,
            0,
            perturbations.k
        )

        cells = {}

        cells.update(cls_posshe)
        cells.update(cls_sheshe)
        cells.update(cls_pospos)

        vec_WL = np.array(
            [cells[k][0, 0] for k in self.WL_keys]
        ).flatten()

        vec_GCph = np.array(
            [cells[k] for k in self.GG_keys]
        ).flatten()

        vec_GGL = np.array(
            [cells[k][0] for k in self.GGL_keys]
        ).flatten()

        return np.concatenate([
            vec_WL,
            vec_GGL,
            vec_GCph
        ])

    # --------------------------------------------------------

    def get_sample_noise(self):

        n = np.random.normal(
            size=self.Lfid.shape[0]
        )

        return self.Lfid @ n

    # --------------------------------------------------------

    def build(self, graph):

        z = graph.node(
            "z",
            self.sample_z
        )

        C_ells = graph.node(
            "C_ells",
            self.get_sample_Cls,
            z
        )

        noise = graph.node(

            "noise",

            self.get_sample_noise
        )


def get_sigmas_bounds(fiducial: list, finv_file: str, N_pars: int):
    Finv = np.load(finv_file)
    sigmas = np.sqrt(np.diag(Finv))
    lower_bounds = [fiducial[i] - 5 * sigmas[i] for i in range(N_pars)]
    upper_bounds = [fiducial[i] + 5 * sigmas[i] for i in range(N_pars)]
    return sigmas, lower_bounds, upper_bounds


def get_positiontracer(nuis_params, n_pos_bins, perturbations, dndz):
    pos_nuisance_params = {}

    for i in range(4):
        pos_nuisance_params[f"b1_photo_poly{i}"] = nuis_params[f"b_g{i+1}"]

    for i in range(n_pos_bins):
        pos_nuisance_params[f"magnification_bias_{i+1}"] = nuis_params[f"b_mag{i+1}"]

    for i in range(n_pos_bins):
        pos_nuisance_params[f"dz_pos_{i+1}"] = nuis_params[f"D_{i+1}"]

    return PositionsTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=zs,
        galaxy_bias_model="poly",
        nuisance_params=pos_nuisance_params,
    )


def get_sheartracer(nuis_params, n_she_bins, perturbations, dndz):
    she_nuisance_params = {}

    she_nuisance_params["AIA"] = nuis_params["A_IA"]
    she_nuisance_params["EtaIA"] = nuis_params["eta_IA"]
    she_nuisance_params["CIA"] = nuis_params["CIA"]

    for i in range(n_she_bins):
        she_nuisance_params[f"multiplicative_bias_{i+1}"] = nuis_params[f"m_{i+1}"]

    for i in range(n_she_bins):
        she_nuisance_params[f"dz_shear_{i+1}"] = nuis_params[f"D_{i+1}"]

    return ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=zs,
        nuisance_params=she_nuisance_params,
    )

