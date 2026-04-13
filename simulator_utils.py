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


import cloelib
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations, CAMBNonLinearPerturbations
from cloelib.cosmology.HMcode2020Emu_cosmology import HMemuLinearPerturbations, HMemuNonLinearPerturbations
from cloelib.observables.photo import ShearTracer, PositionsTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
from cloelib.summary_statistics.angular_correlation_function_wigner import AngularCorrelationFunctionWigner

from helper_utils import get_positiontracer, get_sheartracer, get_sigmas_bounds

zs = np.linspace(1e-4, 3, 100)
nz_example = '/home/awernersson/projects/playground/tutorials/observables/nzTabSPV3_euclidlib_format.fits'
z_nz, nz_example = el.photo.redshift_distributions(nz_example)

def normalize_and_resample(nz_dict, z_grid, z_target):
    nz_array = np.vstack([nz / integrate.trapezoid(nz, z_grid) for nz in nz_dict.values()])
    return np.array([np.interp(z_target, z_grid, nz) for nz in nz_array])

my_dndz_pos_norm = normalize_and_resample(nz_example, z_nz, zs)
my_dndz_she_norm = normalize_and_resample(nz_example, z_nz, zs)

ell_theory = np.array([
    10.97557970, 13.11709027, 15.67644367, 18.73516773, 22.39069761,
    26.75947964, 31.98068069, 38.22062129, 45.67807378, 54.59059412,
    65.24208925, 77.97186088, 93.18541388, 111.36737359, 133.09692347,
    159.06625492, 190.10261691, 227.19466787, 271.52396925, 324.50262398,
    387.81825878, 463.48778323, 553.92163814, 662.00057974, 791.16744571,
    945.53682628, 1130.02613377, 1350.51224608, 1614.01871364, 1928.93949355,
    2305.30633773, 2755.10835283
])

zmean_dr3 = np.array([0.2894, 0.3763, 0.4374, 0.5363, 0.6186, 0.7093,
                      0.802, 0.8591, 0.976, 1.093, 1.246, 1.489, 1.922])


import numpy as np
import torch
from torch.distributions import Uniform, Normal
import swyft


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


# ============================================================
# Prior sampler
# ============================================================

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


# ============================================================
# Simulator
# ============================================================

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
    ):

        super().__init__()

        # dtype consistency with float64 training
        self.transform_samples = swyft.to_numpy

        self.fiducial = fiducial
        self.n_bins = n_bins
        self.ells = ell_theory

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
            perturbations
        )

        tracer_she = get_sheartracer(

            nuisance,
            self.n_bins,
            perturbations
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


def compute_cls(theta, ells, nuisance_config, zs):

    background = CAMBBackground(
        H0 = theta[0],
        Omega_b0 = theta[1],
        Omega_cdm0 = theta[2],
        ns = theta[3],
        As = np.exp(theta[4]) / 1e10,
        w0 = 0,
        wa = -1,
        Omega_k0 = 0.0,
        mnu = 0.077,
        gamma_MG = 0.545,
        N_mnu = 1
    )

    lin = HMemuLinearPerturbations(background, zs)

    nonlin = HMemuNonLinearPerturbations(
        background,
        lin,
        zs,
        log10TAGN = 7.75
    )

    perturbations = nonlin
    nuisance = build_nuisance(theta)
    tracer_pos = get_positiontracer(nuisance, nuisance_config["n_bins"], perturbations)
    tracer_she = get_sheartracer(nuisance, nuisance_config["n_bins"], perturbations)
    cls_sheshe = AngularTwoPoint(tracer_she, tracer_she).get_Cl(ells, 0, perturbations.k)
    cls_posshe = AngularTwoPoint(tracer_pos, tracer_she).get_Cl(ells, 0, perturbations.k)
    cls_pospos = AngularTwoPoint(tracer_pos, tracer_pos).get_Cl(ells, 0, perturbations.k)
    cells = {**cls_posshe, **cls_sheshe, **cls_pospos}

    return flatten_cls(cells, nuisance_config)

class FisherSimulator:

    def __init__(self, config):

        
        self.cov = np.load(
            config["FINV"]["covmat"]
        )["Gauss"]

        self.ells = ell_theory
        self.n_bins = int(config["FINV"]["Nbin_z"])
        self.fiducial = np.array([
            float(v)
            for v in config["FIDUCIAL VALUES"].values()
        ])

        lower_bounds = np.load("lower_bounds.npy")
        upper_bounds = np.load("upper_bounds.npy")

        self.sim = Simulator(

            fiducial=self.fiducial,

            covmat=self.cov,

            n_bins=self.n_bins,

            lower_bounds=lower_bounds,

            upper_bounds=upper_bounds,

            zmean=zmean_dr3,

            ell_theory=self.ells
        )

    def finite_difference(self, i, eps=1e-2):

        theta_p = self.fiducial.copy()
        theta_m = self.fiducial.copy()

        # same step logic as notebook
        if self.fiducial[i] != 0:
            step = eps * self.fiducial[i]
        else:
            step = 5e-4

        theta_p[i] += step
        theta_m[i] -= step

        try:

            C1 = self.sim.sample(
                conditions={'z': theta_p},
                targets=['C_ells']
            )['C_ells']

            C2 = self.sim.sample(
                conditions={'z': theta_m},
                targets=['C_ells']
            )['C_ells']

        except Exception as err:

            print(f"Derivative failed for parameter {i}")
            print("step =", step)
            print(err)

            raise err

        return (C1 - C2) / (2 * step)

def fisher_analysis(config):

    sim = FisherSimulator(config)

    N_pars = len(sim.fiducial)

    inv_cov = np.linalg.inv(sim.cov)

    derivatives = []

    for i in range(N_pars):

        dC = sim.finite_difference(i, eps=1e-2)

        derivatives.append(dC)

    F = np.zeros((N_pars, N_pars))

    for i in range(N_pars):

        for j in range(i, N_pars):

            F[i, j] = derivatives[i] @ inv_cov @ derivatives[j]

            F[j, i] = F[i, j]

    Finv = np.linalg.inv(F)

    sigmas = np.sqrt(np.diag(Finv))

    return Finv, sigmas

class old_PriorSampler:
   def __init__(self, lower_bounds, upper_bounds):
       self.uniform_priors = [Uniform(torch.tensor(lower_bounds[i]), torch.tensor(upper_bounds[i]))
                              for i in range(24)]
       self.gaussian_priors = []
       for i in range(26):
           loc = torch.tensor(0.0)
           if (i < 13):
               scale = torch.tensor(0.01)
           else:
               scale = torch.tensor(0.002*(1.0+zmean_dr3[i-13]))
           base_dist = Normal(loc, scale)
           self.gaussian_priors.append(base_dist)

       self.priors = self.uniform_priors + self.gaussian_priors

   def __call__(self, shape=()):
       samples = []
       for i, prior in enumerate(self.priors):
           sample = prior.sample(shape)
           samples.append(sample)

       return torch.stack(samples, dim=-1).numpy()

class old_Simulator(swyft.Simulator):
    def __init__(self, config_file):

        super().__init__()
        self.transform_samples = swyft.to_numpy32
        self.config = config_file
        case_mapping = {
            'h0': 'H0',       # Map 'h_0' to 'H_0'
            'ln10^{10}a_s': 'ln10^{10}A_s',  # Map 'ln10^{10}a_s' to 'ln10^{10}A_s'
            'a_ia': 'A_IA',     # Map 'a_ia' to 'A_IA'
            'eta_ia': 'eta_IA' # Map 'eta_ia' to 'eta_IA'

        }
        # Load Fiducial Values (convert them to float)
        self.fiducial_values = {
            case_mapping.get(k, k): float(v) for k, v in self.config['FIDUCIAL VALUES'].items()
        }
        fiducial = list(self.fiducial_values.values())
        self.n_bins = int(self.config['FINV']['Nbin_z'])
        finv_file = self.config['FINV']['finv_file']
        N_pars = int(self.config['FINV']['N_pars'])

        _, lower_bounds, upper_bounds = get_sigmas_bounds(fiducial, finv_file, N_pars)
        self.sample_z = PriorSampler(lower_bounds, upper_bounds)

        self.ells = ell_theory
        self.covmat_fid = np.load(self.config['FINV']['covmat'])['Gauss']
        self.Lfid = np.linalg.cholesky(self.covmat_fid)

        self.WL_keys = [("SHE", "SHE", i, j)
            for i in range(1, self.n_bins + 1)
            for j in range(i, self.n_bins + 1)]

        self.GG_keys = [("POS", "POS", i, j)
            for i in range(1, self.n_bins + 1)
            for j in range(i, self.n_bins + 1)]

        self.GGL_keys = [("POS", "SHE", i, j)
            for i in range(1, self.n_bins + 1)
            for j in range(1, self.n_bins + 1)]


    def generate_observation(self):
        fiducial = list(self.fiducial_values.values())
        return self.sample(
            conditions={"z": fiducial},
        )

    def get_sample_Cls(self,z):

        # Define the cosmology
        background = CAMBBackground(H0=z[0],
                            Omega_b0=z[1],
                            Omega_cdm0= z[2],
                            w0=-1,
                            wa=0,
                            Omega_k0 = 0.0,
                            ns = z[3],
                            As = np.exp(z[4])/1e10,
                            mnu = 0.077,
                            gamma_MG = 0.545,
                            N_mnu = 1)
        linear_perturbations_emu = HMemuLinearPerturbations(background, zs)
        nonlinear_perturbations_emu = HMemuNonLinearPerturbations(background, linear_perturbations_emu, zs, log10TAGN=7.75)
        perturbations = nonlinear_perturbations_emu

       # Define the nuisance
        self.nuisance  = {'A_IA': z[5],'eta_IA': z[6],'CIA': 0.0134,
                            'b_g1': z[7],'b_g2': z[8],'b_g3': z[9],'b_g4': z[10],'b_mag1': z[11],
                            'b_mag2': z[12],'b_mag3': z[13],'b_mag4': z[14],'b_mag5': z[15],'b_mag6':z[16],
                            'b_mag7': z[17],'b_mag8': z[18],'b_mag9': z[19],'b_mag10': z[20],'b_mag11':z[21],
                            'b_mag12': z[22],'b_mag13': z[23],
                            'm_1': z[24], 'm_2': z[25], 'm_3': z[26], 'm_4': z[27], 'm_5': z[28],
                            'm_6': z[29], 'm_7': z[30], 'm_8': z[31], 'm_9': z[32], 'm_10': z[33],
                            'm_11': z[34], 'm_12': z[35], 'm_13': z[36],
                            'D_1': z[37], 'D_2': z[38], 'D_3': z[39], 'D_4': z[40], 'D_5': z[41],
                            'D_6': z[42], 'D_7': z[43], 'D_8': z[44], 'D_9': z[45], 'D_10': z[46],
                            'D_11': z[47], 'D_12': z[48], 'D_13': z[49]}

        #Compute the tracers
        tracer_pos = get_positiontracer(self.nuisance, self.n_bins, perturbations)
        tracer_she = get_sheartracer(self.nuisance, self.n_bins, perturbations)

        cls_sheshe = AngularTwoPoint(tracer_she, tracer_she).get_Cl(self.ells, 0, perturbations.k)
        cls_posshe = AngularTwoPoint(tracer_pos, tracer_she).get_Cl(self.ells, 0, perturbations.k)
        cls_pospos = AngularTwoPoint(tracer_pos, tracer_pos).get_Cl(self.ells, 0, perturbations.k)

        cells ={**cls_posshe, **cls_sheshe, **cls_pospos}

        vec_WL   = np.array([cells[key][0, 0] for key in self.WL_keys]).flatten()
        vec_GCph = np.array([cells[key] for key in self.GG_keys]).flatten()
        vec_GGL  = np.array([cells[key][0] for key in self.GGL_keys]).flatten()
        return np.concatenate([vec_WL,vec_GGL,vec_GCph])

    def get_sample_noise(self):
        n = np.random.normal(size=self.Lfid.shape[0])
        noise = self.Lfid @ n
        return noise

    def build(self, graph):
        # define the computational graph of the simulator
        z = graph.node('z', self.sample_z) #draw parameters from the prior
        C_ells = graph.node('C_ells', self.get_sample_Cls, z) #compute un-noised Cls
        noise  = graph.node('noise', self.get_sample_noise) # get the noise, we will add it to the un-noised Cls later

