import numpy as np
import torch
from torch.distributions import Uniform, Normal

import swyft

from . import _compat  # noqa: F401 — sets up cloelib path
from .params import ZS, N_COSMO, NUISANCE_KEYS
from .tracers import load_dndz, get_position_tracer, get_shear_tracer


# ============================================================
# Prior bounds
# ============================================================

def get_sigmas_bounds(fiducial, finv_file, N_pars, scale=5):
    Finv = np.load(finv_file)
    sigmas = np.sqrt(np.diag(Finv))
    lower_bounds = [fiducial[i] - scale * sigmas[i] for i in range(N_pars)]
    upper_bounds = [fiducial[i] + scale * sigmas[i] for i in range(N_pars)]
    return sigmas, lower_bounds, upper_bounds


# ============================================================
# Prior sampler
# ============================================================

class PriorSampler:

    def __init__(self, lower_bounds, upper_bounds, zmean):
        self.priors = []
        for lo, hi in zip(lower_bounds, upper_bounds):
            self.priors.append(Uniform(torch.tensor(lo), torch.tensor(hi)))
        for i in range(len(zmean) * 2):
            loc   = torch.tensor(0.0)
            scale = torch.tensor(0.01) if i < len(zmean) \
                    else torch.tensor(0.002 * (1.0 + zmean[i - len(zmean)]))
            self.priors.append(Normal(loc, scale))

    def __call__(self, shape=()):
        return torch.stack([p.sample(shape) for p in self.priors], dim=-1).numpy()


# ============================================================
# Simulator
# ============================================================

class Simulator(swyft.Simulator):

    def __init__(self, fiducial, covmat, n_bins, lower_bounds, upper_bounds,
                 zmean, ell_theory, dndz):
        super().__init__()
        self.transform_samples = swyft.to_numpy
        self.fiducial = fiducial
        self.n_bins   = n_bins
        self.ells     = ell_theory
        self.dndz     = dndz
        self.sample_z = PriorSampler(lower_bounds, upper_bounds, zmean)
        self.Lfid     = np.linalg.cholesky(covmat)
        self._build_keys()

    def _build_keys(self):
        self.WL_keys = [
            ("SHE", "SHE", i, j)
            for i in range(1, self.n_bins + 1)
            for j in range(i, self.n_bins + 1)
        ]
        self.GG_keys = [
            ("POS", "POS", i, j)
            for i in range(1, self.n_bins + 1)
            for j in range(i, self.n_bins + 1)
        ]
        self.GGL_keys = [
            ("POS", "SHE", i, j)
            for i in range(1, self.n_bins + 1)
            for j in range(1, self.n_bins + 1)
        ]

    @property
    def n_data(self):
        """Total length of the flattened data vector."""
        return (len(self.WL_keys) + len(self.GG_keys) + len(self.GGL_keys)) * len(self.ells)

    def generate_observation(self):
        return self.sample(conditions={"z": np.array(self.fiducial)})

    def get_sample_Cls(self, z):
        from cloelib.cosmology.HMcode2020Emu_cosmology import HMemuLinearPerturbations, HMemuNonLinearPerturbations
        from cloelib.cosmology.camb_cosmology import CAMBBackground
        from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

        background = CAMBBackground(
            H0=z[0], Omega_b0=z[1], Omega_cdm0=z[2],
            w0=-1, wa=0, Omega_k0=0.0,
            ns=z[3], As=np.exp(z[4]) / 1e10,
            mnu=0.077, gamma_MG=0.545, N_mnu=1,
        )
        linear        = HMemuLinearPerturbations(background, ZS)
        perturbations = HMemuNonLinearPerturbations(background, linear, ZS, log10TAGN=7.75)

        nuisance = dict(zip(NUISANCE_KEYS, z[N_COSMO:]))
        nuisance["CIA"] = 0.0134

        tracer_pos = get_position_tracer(nuisance, self.n_bins, perturbations, self.dndz)
        tracer_she = get_shear_tracer(nuisance, self.n_bins, perturbations, self.dndz)

        cls_sheshe = AngularTwoPoint(tracer_she, tracer_she).get_Cl(self.ells, 0, perturbations.k)
        cls_posshe = AngularTwoPoint(tracer_pos, tracer_she).get_Cl(self.ells, 0, perturbations.k)
        cls_pospos = AngularTwoPoint(tracer_pos, tracer_pos).get_Cl(self.ells, 0, perturbations.k)

        cells    = {**cls_posshe, **cls_sheshe, **cls_pospos}
        vec_WL   = np.array([cells[k][0, 0] for k in self.WL_keys]).flatten()
        vec_GCph = np.array([cells[k]        for k in self.GG_keys]).flatten()
        vec_GGL  = np.array([cells[k][0]     for k in self.GGL_keys]).flatten()

        return np.concatenate([vec_WL, vec_GGL, vec_GCph])

    def get_sample_noise(self):
        return self.Lfid @ np.random.normal(size=self.Lfid.shape[0])

    def build(self, graph):
        z      = graph.node("z",      self.sample_z)
        C_ells = graph.node("C_ells", self.get_sample_Cls, z)
        noise  = graph.node("noise",  self.get_sample_noise)


# ============================================================
# Factory
# ============================================================

def build_simulator(config):
    """Construct a Simulator from a loaded YAML config dict."""
    fiducial  = list(config["FIDUCIAL VALUES"].values())
    covmat    = np.load(config["FINV"]["covmat"])["Gauss"]
    n_bins    = config["FINV"]["Nbin_z"]
    finv_file = config["FINV"]["finv_file"]
    N_pars    = config["FINV"]["N_pars"]
    ell_theory = np.load(config["AUX FILES"]["ell_file"])
    zmean      = np.load(config["AUX FILES"]["zmean_file"])
    dndz       = load_dndz(config["FINV"]["nz_example"])

    _, lower_bounds, upper_bounds = get_sigmas_bounds(fiducial, finv_file, N_pars)

    return Simulator(
        fiducial=fiducial, covmat=covmat, n_bins=n_bins,
        lower_bounds=lower_bounds, upper_bounds=upper_bounds,
        zmean=zmean, ell_theory=ell_theory, dndz=dndz,
    )
