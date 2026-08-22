import contextlib
import os

import numpy as np
import torch
from torch.distributions import Uniform, Normal

import swyft

from .io import load_array
from .params import ZS, N_COSMO, NUISANCE_KEYS, resolve_derived_names
from .priors import resolve_priors, apply_fisher_bounds, load_fisher_sigmas
from .tracers import load_dndz, get_position_tracer, get_shear_tracer


@contextlib.contextmanager
def _suppress_stdout():
    """Silence stdout: cloelib's HMcode2020Emu prints unconditionally on import/construction, with no verbosity flag to suppress it."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


# ============================================================
# Prior sampler
# ============================================================

class _Constant:
    """Degenerate 'distribution' for a fixed parameter: .sample(shape) always returns its value, broadcast. Duck-types torch.distributions so PriorSampler treats fixed/varied entries uniformly."""

    def __init__(self, value):
        self.value = torch.tensor(float(value))

    def sample(self, shape=torch.Size()):
        return self.value.expand(shape)


class _TruncatedNormalRejection:
    """
    Normal(mean, sigma) restricted to [lower, upper] (see
    priors.apply_fisher_bounds, used when PRIORS.use_Fisher_priors narrows
    m_i/D_i-style nuisance parameters to fiducial +/- sigma_scale*sigma_Fisher).
    Duck-types torch.distributions .sample(shape) like _Constant.

    Uses rejection sampling (draw untruncated Normal, discard outside
    bounds) rather than scipy.stats.truncnorm's inverse-CDF, which produces
    systematically overconfident/biased trained posteriors here despite
    targeting the same distribution.
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
    order: Uniform, Normal, or a fixed constant, per config["PRIORS"]. Fixed
    entries are included in sample_z's output as constants — "z" is always
    the full N_pars length, positionally matching PARAMS, since
    get_sample_Cls indexes into it positionally (z[0]=H0, ...,
    NUISANCE_KEYS = PARAMS[N_COSMO:]).
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

    shape=() (default): a single n_data-length vector, randn(n_data) @
    Lfid.T (equivalent to Lfid @ randn(n_data) for a 1D vector). Used by
    Simulator.get_sample_noise, one draw per graph.node call.
    shape=(n,): a batch of n draws via one vectorized matmul, shape
    (n, n_data) — used by preprocessing.preprocess's
    ANALYSIS_VARIANTS.regenerate_noise_samples. Both call sites share this
    function so batched regeneration matches the simulator's own draw exactly.
    """
    z = np.random.normal(size=(*shape, Lfid.shape[0]))
    return z @ Lfid.T


# ============================================================
# Simulator
# ============================================================

class Simulator(swyft.Simulator):

    def __init__(self, fiducial, covmat, n_bins, specs, ell_theory, dndz, derived_names=()):
        super().__init__()
        self.transform_samples = swyft.to_numpy
        self.fiducial = fiducial
        self.n_bins   = n_bins
        self.ells     = ell_theory
        self.dndz     = dndz
        self.sample_z = PriorSampler(specs, fiducial)
        self.Lfid     = np.linalg.cholesky(covmat)
        self.derived_names = tuple(derived_names)
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
        # (j, i) not (i, j): matches cloelib's own key convention.
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

        # N_mnu is internally fixed to 1 (not config-exposed). z's entries are
        # numpy.float32 (PriorSampler stacks float32 torch tensors) so cast
        # explicitly to plain Python float: cloelib's CAMBBackground
        # requires isinstance(mnu, float), which numpy.float32 fails.
        background = CAMBBackground(
            H0=float(z[0]), Omega_b0=float(z[1]), Omega_cdm0=float(z[2]),
            w0=float(nuisance["w0"]), wa=float(nuisance["wa"]), Omega_k0=float(nuisance["Omega_k0"]),
            ns=float(z[3]), As=float(np.exp(z[4]) / 1e10),
            mnu=float(nuisance["mnu"]), gamma_MG=float(nuisance["gamma_MG"]), N_mnu=1,
        )
        with _suppress_stdout():
            linear        = HMemuLinearPerturbations(background, ZS)
            perturbations = HMemuNonLinearPerturbations(background, linear, ZS, log10TAGN=nuisance["log10TAGN"])

        if self.derived_names:
            self._derived_values = self._compute_derived(background, perturbations)

        # CIA is internally fixed (not config-exposed).
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

    def _compute_derived(self, background, perturbations):
        """
        sigma8/Omega_m/S8 from the same background/perturbations objects
        get_sample_Cls already built for C_ells (near-zero marginal cost).
        Only computes what self.derived_names asked for; returns an array
        ordered per params.DERIVED_PARAMS (not self.derived_names' order),
        matching what get_derived_params/the "derived" graph node expose.
        """
        need_omega_m = "Omega_m" in self.derived_names or "S8" in self.derived_names
        need_sigma8  = "sigma8"  in self.derived_names or "S8" in self.derived_names

        omega_m = float(background.Omega_m(0.0)) if need_omega_m else None
        sigma8  = float(perturbations.sigma8_0()) if need_sigma8 else None
        values = {
            "sigma8":  sigma8,
            "Omega_m": omega_m,
            "S8":      sigma8 * (omega_m / 0.3) ** 0.5 if "S8" in self.derived_names else None,
        }
        return np.array([values[name] for name in self.derived_names], dtype=np.float64)

    def get_derived_params(self):
        return self._derived_values

    def get_sample_noise(self):
        return sample_correlated_noise(self.Lfid)

    def build(self, graph):
        z      = graph.node("z",      self.sample_z)
        C_ells = graph.node("C_ells", self.get_sample_Cls, z)
        if self.derived_names:
            derived = graph.node("derived", self.get_derived_params)
        noise  = graph.node("noise",  self.get_sample_noise)


# ============================================================
# Factory
# ============================================================

def build_simulator(config):
    """
    Construct a Simulator from a loaded YAML config dict.

    If config["PRIORS"]["use_Fisher_priors"] is true, loads the Fisher matrix
    from fisher.run_fisher (config["PRIORS"]["finv_file"] must already exist —
    see observation.generate_observation) and narrows every varied
    parameter's support to fiducial +/- sigma_scale*sigma: "uniform"
    parameters get those bounds directly; "normal" parameters keep their
    mean/sigma but get truncated to that window (PriorSampler/
    _TruncatedNormalRejection). Concentrates simulation density where the
    posterior actually has support. Raises via priors.load_fisher_sigmas if
    finv_file is missing or stale relative to the current PRIORS fixed/varied set.
    """
    fiducial, specs, varied_names, varied_indices = resolve_priors(config)

    if config["PRIORS"].get("use_Fisher_priors", False):
        sigmas = load_fisher_sigmas(config["PRIORS"]["finv_file"], varied_indices, varied_names)
        scale  = config["PRIORS"].get("sigma_scale", 5)
        specs  = apply_fisher_bounds(specs, fiducial, varied_indices, sigmas, scale)

    covmat     = np.load(config["CLOELIB_SETTINGS"]["covmat"])["Gauss"]
    n_bins     = config["CLOELIB_SETTINGS"]["Nbin_z"]
    ell_theory = load_array(config["CLOELIB_SETTINGS"]["ell"])
    dndz       = load_dndz(config["CLOELIB_SETTINGS"]["nz"])
    derived_names = resolve_derived_names(config["CLOELIB_SETTINGS"].get("add_derived"))

    return Simulator(
        fiducial=fiducial, covmat=covmat, n_bins=n_bins,
        specs=specs, ell_theory=ell_theory, dndz=dndz,
        derived_names=derived_names,
    )
