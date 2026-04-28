"""
Simulator and prior sampler for the Merlin SBI pipeline.

Corresponds to the original ``simulator_utils.py``.
"""

import numpy as np
import swyft
import torch
from torch.distributions import Uniform, Normal

from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.cosmology.HMcode2020Emu_cosmology import (
    HMemuLinearPerturbations,
    HMemuNonLinearPerturbations,
)
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

from .params import ZS, NUISANCE_KEYS, N_COSMO
from .tracers import get_positiontracer, get_sheartracer


# ── Prior sampler ─────────────────────────────────────────────────────────────

class PriorSampler:
    """
    Sample cosmological and nuisance parameters from their prior distributions.

    Cosmological and galaxy-bias parameters are drawn from uniform priors
    bounded by the Fisher-matrix-derived ±5σ interval.  Shear-calibration and
    photo-z shift parameters are drawn from narrow Gaussians.
    """

    def __init__(self, lower_bounds, upper_bounds, zmean):
        self.priors = []

        # Uniform priors for cosmology + bias parameters
        for lo, hi in zip(lower_bounds, upper_bounds):
            self.priors.append(
                Uniform(torch.tensor(lo), torch.tensor(hi - lo))
            )

        # Gaussian priors for shear calibration and photo-z shifts/widths
        for i in range(len(zmean) * 2):
            loc = torch.tensor(0.0)
            if i < len(zmean):
                scale = torch.tensor(0.01)
            else:
                scale = torch.tensor(0.002 * (1.0 + zmean[i - len(zmean)]))
            self.priors.append(Normal(loc, scale))

    def __call__(self, shape=()):
        samples = [p.sample(shape) for p in self.priors]
        return torch.stack(samples, dim=-1).numpy()


# ── Simulator ─────────────────────────────────────────────────────────────────

class Simulator(swyft.Simulator):
    """
    Swyft simulator for the Euclid 3×2pt observable.

    For each parameter sample *z* the simulator computes:

    * ``C_ells`` — concatenated WL / GGL / GCph angular power spectra.
    * ``noise`` — a Gaussian noise realisation drawn from the fiducial
      covariance (Cholesky factored).
    """

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
        self.transform_samples = swyft.to_numpy

        self.fiducial = fiducial
        self.n_bins = n_bins
        self.ells = ell_theory
        self.dndz = dndz

        self.sample_z = PriorSampler(lower_bounds, upper_bounds, zmean)

        # Cholesky factor of the fiducial covariance
        self.Lfid = np.linalg.cholesky(covmat)

        self._build_keys()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_keys(self):
        n = self.n_bins
        self.WL_keys = [
            ("SHE", "SHE", i, j)
            for i in range(1, n + 1)
            for j in range(i, n + 1)
        ]
        self.GG_keys = [
            ("POS", "POS", i, j)
            for i in range(1, n + 1)
            for j in range(i, n + 1)
        ]
        self.GGL_keys = [
            ("POS", "SHE", i, j)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
        ]
        self.n_data = len(self.WL_keys) + len(self.GGL_keys) + len(self.GG_keys)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_observation(self):
        """Return a single noiseless realisation at the fiducial cosmology."""
        return self.sample(conditions={"z": np.array(self.fiducial)})

    # ── Node functions for swyft graph ────────────────────────────────────────

    def get_sample_Cls(self, z):
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
            N_mnu=1,
        )
        lin = HMemuLinearPerturbations(background, ZS)
        nonlin = HMemuNonLinearPerturbations(background, lin, ZS, log10TAGN=7.75)

        nuisance = dict(zip(NUISANCE_KEYS, z[N_COSMO:]))
        nuisance["CIA"] = 0.0134

        tracer_pos = get_positiontracer(nuisance, self.n_bins, nonlin, self.dndz)
        tracer_she = get_sheartracer(nuisance, self.n_bins, nonlin, self.dndz)

        cls_sheshe = AngularTwoPoint(tracer_she, tracer_she).get_Cl(
            self.ells, 0, nonlin.k
        )
        cls_posshe = AngularTwoPoint(tracer_pos, tracer_she).get_Cl(
            self.ells, 0, nonlin.k
        )
        cls_pospos = AngularTwoPoint(tracer_pos, tracer_pos).get_Cl(
            self.ells, 0, nonlin.k
        )

        cells = {}
        cells.update(cls_posshe)
        cells.update(cls_sheshe)
        cells.update(cls_pospos)

        vec_WL = np.array([cells[k][0, 0] for k in self.WL_keys]).flatten()
        vec_GCph = np.array([cells[k] for k in self.GG_keys]).flatten()
        vec_GGL = np.array([cells[k][0] for k in self.GGL_keys]).flatten()

        return np.concatenate([vec_WL, vec_GGL, vec_GCph])

    def get_sample_noise(self):
        n = np.random.normal(size=self.Lfid.shape[0])
        return self.Lfid @ n

    def build(self, graph):
        z = graph.node("z", self.sample_z)
        graph.node("C_ells", self.get_sample_Cls, z)
        graph.node("noise", self.get_sample_noise)
