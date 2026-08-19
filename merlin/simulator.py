import contextlib
import os

import numpy as np
import torch
from torch.distributions import Uniform, Normal

import swyft

from .io import load_array
from .params import ZS, N_COSMO, NUISANCE_KEYS
from .priors import resolve_priors, apply_fisher_bounds, load_fisher_sigmas
from .tracers import load_dndz, get_position_tracer, get_shear_tracer


@contextlib.contextmanager
def _suppress_stdout():
    """
    HMcode2020Emu (via cloelib) prints its install path on first import per
    process, plus "Loading .../.. loaded in memory." on every emulator
    construction — plain print() calls with no verbosity flag exposed through
    cloelib's own wrapper, so silence stdout around the calls that trigger them.
    """
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


# ============================================================
# Prior sampler
# ============================================================

class _Constant:
    """
    Degenerate 'distribution' for a fixed parameter: always returns its fixed
    value, broadcast to the requested sample shape. Duck-types the
    torch.distributions .sample(shape) interface so PriorSampler can treat
    fixed and varied entries uniformly, with no special-casing in __call__.
    """

    def __init__(self, value):
        self.value = torch.tensor(float(value))

    def sample(self, shape=torch.Size()):
        return self.value.expand(shape)


class _TruncatedNormalRejection:
    """
    Normal(mean, sigma) restricted to [lower, upper] (see
    priors.apply_fisher_bounds — used for the m_i/D_i-style nuisance
    parameters when PRIORS.use_Fisher_priors narrows their support to
    fiducial +/- sigma_scale*sigma_Fisher, same as it already does for
    uniform-type parameters). Duck-types the torch.distributions
    .sample(shape) interface like _Constant.

    Sampled via rejection sampling -- draw from the untruncated
    Normal(mean, sigma) and discard points outside [lower, upper] -- rather
    than scipy.stats.truncnorm's exact inverse-CDF, which was found to
    produce systematically overconfident/biased trained posteriors for
    Fisher-truncated Gaussian-type priors in this pipeline despite the two
    methods targeting the same distribution.
    """

    def __init__(self, mean, sigma, lower, upper):
        self.mean, self.sigma = mean, sigma
        self.lower, self.upper = lower, upper

    def sample(self, shape=torch.Size()):
        n = int(np.prod(shape))
        out = np.empty(n, dtype=np.float64)
        filled = 0
        while filled < n:
            need = n - filled
            batch = np.random.normal(self.mean, self.sigma, size=max(need * 2, 1000))
            batch = batch[(batch >= self.lower) & (batch <= self.upper)]
            take = min(len(batch), need)
            out[filled:filled + take] = batch[:take]
            filled += take
        return torch.tensor(out, dtype=torch.float32).reshape(shape)


class PriorSampler:
    """
    Prior over the full PARAMS vector, one entry per parameter in PARAMS
    order (see priors.resolve_priors): Uniform, Normal, or a fixed constant,
    per config["PRIORS"]. Fixed entries ARE included in sample_z's output
    (as constants) — the sampled "z" vector is always the full N_pars length,
    positionally matching PARAMS, since get_sample_Cls indexes into it
    positionally (z[0]=H0, ..., NUISANCE_KEYS = PARAMS[N_COSMO:]).
    Network.forward's z_full[..., self.param_indices] is unaffected since
    param_indices only ever indexes VARIED positions (enforced by
    priors.resolve_inference_params).
    """

    def __init__(self, specs, fiducial):
        self.priors = []
        for spec, fid in zip(specs, fiducial):
            if spec.kind == "fixed":
                self.priors.append(_Constant(fid))
            elif spec.kind == "uniform":
                self.priors.append(Uniform(torch.tensor(spec.lower), torch.tensor(spec.upper)))
            elif spec.kind == "normal":
                if spec.lower is not None and spec.upper is not None:
                    self.priors.append(_TruncatedNormalRejection(spec.mean, spec.sigma, spec.lower, spec.upper))
                else:
                    self.priors.append(Normal(torch.tensor(spec.mean), torch.tensor(spec.sigma)))
            else:
                raise ValueError(f"Unknown prior kind {spec.kind!r} for parameter {spec.name!r}")

    def __call__(self, shape=()):
        return torch.stack([p.sample(shape) for p in self.priors], dim=-1).numpy()


def sample_correlated_noise(Lfid, shape=()):
    """
    Draw correlated Gaussian noise realization(s) with covariance
    Lfid @ Lfid.T.

    shape=() (default): a single n_data-length vector — mathematically
    Lfid @ randn(n_data), computed here as the equivalent randn(n_data) @
    Lfid.T (identical for a 1D vector: (Lfid @ v)_i == (v @ Lfid.T)_i for
    all i). Used by Simulator.get_sample_noise, matching swyft's per-sample
    DAG sampling convention (one noise draw per graph.node call).
    shape=(n,): a batch of n independent draws in a single vectorized
    matmul, shape (n, n_data) — used by preprocessing.preprocess's
    ANALYSIS VARIANTS.regenerate_noise_samples, for the same reason
    inference.infer batches sim.sample_z instead of looping swyft's own
    per-sample sample(). Both call sites go through this one function so
    "regenerate_noise_samples" is provably the same draw swyft's own
    Simulator.build graph would produce, batched.
    """
    z = np.random.normal(size=(*shape, Lfid.shape[0]))
    return z @ Lfid.T


# ============================================================
# Simulator
# ============================================================

class Simulator(swyft.Simulator):

    def __init__(self, fiducial, covmat, n_bins, specs, ell_theory, dndz):
        super().__init__()
        self.transform_samples = swyft.to_numpy
        self.fiducial = fiducial
        self.n_bins   = n_bins
        self.ells     = ell_theory
        self.dndz     = dndz
        self.sample_z = PriorSampler(specs, fiducial)
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
        # IMPORTANT: (j, i) not (i, j), this reflects the convention of latest cloelib version
        self.GGL_keys = [
            ("POS", "SHE", j, i)
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
        with _suppress_stdout():
            from cloelib.cosmology.HMcode2020Emu_cosmology import HMemuLinearPerturbations, HMemuNonLinearPerturbations
        from cloelib.cosmology.camb_cosmology import CAMBBackground
        from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

        nuisance = dict(zip(NUISANCE_KEYS, z[N_COSMO:]))

        # N_mnu is internally fixed to 1 (not a config-exposed parameter).
        background = CAMBBackground(
            H0=z[0], Omega_b0=z[1], Omega_cdm0=z[2],
            w0=nuisance["w0"], wa=nuisance["wa"], Omega_k0=nuisance["Omega_k0"],
            ns=z[3], As=np.exp(z[4]) / 1e10,
            mnu=nuisance["mnu"], gamma_MG=nuisance["gamma_MG"], N_mnu=1,
        )
        with _suppress_stdout():
            linear        = HMemuLinearPerturbations(background, ZS)
            perturbations = HMemuNonLinearPerturbations(background, linear, ZS, log10TAGN=nuisance["log10TAGN"])

        # CIA is internally fixed to 0.0134 (not a config-exposed parameter).
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
        return sample_correlated_noise(self.Lfid)

    def build(self, graph):
        z      = graph.node("z",      self.sample_z)
        C_ells = graph.node("C_ells", self.get_sample_Cls, z)
        noise  = graph.node("noise",  self.get_sample_noise)


# ============================================================
# Factory
# ============================================================

def build_simulator(config):
    """
    Construct a Simulator from a loaded YAML config dict.

    If config["PRIORS"]["use_Fisher_priors"] is true, LOADS the Fisher matrix
    already computed by fisher.run_fisher (config["PRIORS"]["finv_file"] must
    already exist — see observation.generate_observation, which runs Fisher
    once before the first build_simulator call in the pipeline) and narrows
    every VARIED parameter's support in-memory to fiducial ± sigma_scale*sigma:
    "uniform" parameters get those bounds directly; "normal" parameters keep
    their configured mean/sigma but get their density truncated to that same
    window (see PriorSampler/_TruncatedNormalRejection) — concentrating simulation
    density where the posterior actually has support instead of wasting most
    of the budget on the (exponentially, in high dimensions) more likely
    events far from it. Raises a clear error (via priors.load_fisher_sigmas)
    if finv_file is missing or stale relative to the current PRIORS
    fixed/varied set.
    """
    fiducial, specs, varied_names, varied_indices = resolve_priors(config)

    if config["PRIORS"].get("use_Fisher_priors", False):
        sigmas = load_fisher_sigmas(config["PRIORS"]["finv_file"], varied_indices, varied_names)
        scale  = config["PRIORS"].get("sigma_scale", 5)
        specs  = apply_fisher_bounds(specs, fiducial, varied_indices, sigmas, scale)

    covmat     = np.load(config["AUX FILES"]["covmat"])["Gauss"]
    n_bins     = config["AUX FILES"]["Nbin_z"]
    ell_theory = load_array(config["AUX FILES"]["ell"])
    dndz       = load_dndz(config["AUX FILES"]["nz"])

    return Simulator(
        fiducial=fiducial, covmat=covmat, n_bins=n_bins,
        specs=specs, ell_theory=ell_theory, dndz=dndz,
    )
