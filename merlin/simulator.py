import contextlib
import os

import numpy as np
import torch
from torch.distributions import Uniform, Normal

import swyft

from .io import load_array
from .params import ZS, N_COSMO, NUISANCE_KEYS, resolve_derived_names, resolve_derived_box_names
from .priors import (
    resolve_priors, apply_fisher_bounds, load_fisher_sigmas, load_derived_fisher_box,
    DERIVED_REJECTION_SIGMA_SCALE,
)
from .tracers import load_dndz, get_position_tracer, get_shear_tracer

# Safety cap on consecutive rejections within _sample_z_derived_rejection,
# for one row, before raising -- at the ~15-25% acceptance rates already
# measured empirically (tests/derived_prior_rejection_check.py), reaching
# this is astronomically unlikely for a correctly-configured box
# (0.85**3000 ~ 1e-212), so hitting it means a broken box (e.g. stale
# finv.npz), not bad luck.
_MAX_CONSECUTIVE_REJECTIONS = 3000


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

    def __init__(self, fiducial, covmat, n_bins, specs, ell_theory, dndz,
                 derived_names=(), derived_box=None):
        super().__init__()
        self.transform_samples = swyft.to_numpy
        self.fiducial = fiducial
        self.n_bins   = n_bins
        self.ells     = ell_theory
        self.dndz     = dndz
        self.sample_z = PriorSampler(specs, fiducial)
        self.Lfid     = np.linalg.cholesky(covmat)
        self.derived_names = tuple(derived_names)
        # CLOELIB_SETTINGS.restrict_prior_for_derived: {name: (lo, hi)} for
        # the subset of derived_names that gates acceptance in
        # _sample_z_derived_rejection (see params.resolve_derived_box_names
        # for which subset). Empty/falsy -> today's unrestricted behavior.
        self.derived_box = dict(derived_box) if derived_box else {}
        if not set(self.derived_box) <= set(self.derived_names):
            raise ValueError(
                f"derived_box names {sorted(self.derived_box)} must be a "
                f"subset of derived_names {self.derived_names}"
            )
        self._n_generated = 0
        self._n_accepted = 0
        self._cached_z = None
        self._cached_background = None
        self._cached_perturbations = None
        self._cached_nuisance = None
        self._cached_derived_values = None
        self._build_keys()

    def __getstate__(self):
        # CAMB result objects (self._cached_background/_cached_perturbations)
        # are explicitly not picklable (camb.results.CAMBdata.__getstate__
        # raises). They're only ever a same-process, same-call cache
        # (_sample_z_derived_rejection -> get_sample_Cls), so drop them on
        # pickling -- e.g. when joblib dispatches this Simulator to a worker,
        # or relays a worker exception back, which would otherwise try to
        # pickle a live cache and fail (masking the real error). Losing the
        # cache across a pickle boundary just means the next get_sample_Cls
        # call recomputes instead of reusing it -- harmless.
        state = self.__dict__.copy()
        state["_cached_z"] = None
        state["_cached_background"] = None
        state["_cached_perturbations"] = None
        state["_cached_nuisance"] = None
        state["_cached_derived_values"] = None
        return state

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

    def _background_perturbations(self, z):
        """
        CAMBBackground + HMemuLinearPerturbations + HMemuNonLinearPerturbations
        for a full PARAMS-ordered z (background/perturbations only, no
        Cls/tracers) -- single source of truth reused by get_sample_Cls,
        _sample_z_derived_rejection's accept/reject loop, and
        fisher.py's derived-quantity finite differencing.
        """
        with _suppress_stdout():
            from cloelib.cosmology.HMcode2020Emu_cosmology import HMemuLinearPerturbations, HMemuNonLinearPerturbations
        from cloelib.cosmology.camb_cosmology import CAMBBackground

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

        return background, perturbations, nuisance

    def get_sample_Cls(self, z):
        from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

        if self.derived_box and self._cached_z is z:
            # Already vetted (and background/perturbations/derived already
            # computed) by _sample_z_derived_rejection -- reuse rather than
            # re-running cloelib for the same z a second time.
            background, perturbations = self._cached_background, self._cached_perturbations
            nuisance = dict(self._cached_nuisance)
            self._derived_values = self._cached_derived_values
        else:
            background, perturbations, nuisance = self._background_perturbations(z)
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

    def _sample_z_derived_rejection(self):
        """
        z-node callback used instead of self.sample_z when self.derived_box
        is set (CLOELIB_SETTINGS.restrict_prior_for_derived): draws a
        candidate from the same Fisher-uniform COSMO box as always
        (self.sample_z, unchanged), computes its derived quantities via
        _background_perturbations/_compute_derived -- stopping short of
        Cls/tracers/AngularTwoPoint, the whole point -- and accepts only if
        every quantity in self.derived_box falls inside its (lo, hi). On
        rejection, discards and redraws a fresh z (including nuisance
        params) from scratch.

        Caches the accepted z's background/perturbations/nuisance/derived
        values as instance attributes so get_sample_Cls, invoked next on
        this exact z object, reuses them via `is` identity instead of
        recomputing. Only ever called with shape=() (one row at a time, per
        the swyft graph's per-sample call pattern) -- not the batched
        inference-time sample_z(shape=(n,)) path.

        Raises RuntimeError after _MAX_CONSECUTIVE_REJECTIONS straight
        rejections for one row with no acceptance (see that constant).
        """
        consecutive_rejections = 0
        while True:
            z = self.sample_z()
            self._n_generated += 1

            background, perturbations, nuisance = self._background_perturbations(z)
            derived_values = self._compute_derived(background, perturbations)
            derived = dict(zip(self.derived_names, derived_values))

            if all(lo <= derived[name] <= hi for name, (lo, hi) in self.derived_box.items()):
                self._n_accepted += 1
                self._cached_z = z
                self._cached_background = background
                self._cached_perturbations = perturbations
                self._cached_nuisance = nuisance
                self._cached_derived_values = derived_values
                return z

            consecutive_rejections += 1
            if consecutive_rejections >= _MAX_CONSECUTIVE_REJECTIONS:
                raise RuntimeError(
                    f"{_MAX_CONSECUTIVE_REJECTIONS} consecutive candidates rejected "
                    f"by the derived-quantity prior box {self.derived_box} with no "
                    "acceptance -- check FIDUCIAL matches the fiducial Fisher ran "
                    "at, and that CLOELIB_SETTINGS.add_derived hasn't changed since "
                    "finv.npz was last written (re-run Fisher if so)."
                )

    def build(self, graph):
        z_fn  = self._sample_z_derived_rejection if self.derived_box else self.sample_z
        z      = graph.node("z",      z_fn)
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

    use_fisher = config["PRIORS"].get("use_Fisher_priors", False)
    if use_fisher:
        sigmas = load_fisher_sigmas(config["PRIORS"]["finv_file"], varied_indices, varied_names)
        scale  = config["PRIORS"].get("sigma_scale", 5)
        specs  = apply_fisher_bounds(specs, fiducial, varied_indices, sigmas, scale)

    covmat     = np.load(config["CLOELIB_SETTINGS"]["covmat"])["Gauss"]
    n_bins     = config["CLOELIB_SETTINGS"]["Nbin_z"]
    ell_theory = load_array(config["CLOELIB_SETTINGS"]["ell"])
    dndz       = load_dndz(config["CLOELIB_SETTINGS"]["nz"])
    derived_names = resolve_derived_names(config["CLOELIB_SETTINGS"].get("add_derived"))

    # CLOELIB_SETTINGS.restrict_prior_for_derived: only meaningful with
    # Fisher priors and >=1 derived quantity requested -- silent no-op
    # (derived_box stays None) otherwise, see simulator.Simulator.build.
    derived_box = None
    if use_fisher and config["CLOELIB_SETTINGS"].get("restrict_prior_for_derived", False) and derived_names:
        box_names = resolve_derived_box_names(derived_names)
        sigma_scale_derived = config["CLOELIB_SETTINGS"].get(
            "sigma_scale_derived", DERIVED_REJECTION_SIGMA_SCALE)
        derived_box = load_derived_fisher_box(
            config["PRIORS"]["finv_file"], box_names, scale=sigma_scale_derived)

    return Simulator(
        fiducial=fiducial, covmat=covmat, n_bins=n_bins,
        specs=specs, ell_theory=ell_theory, dndz=dndz,
        derived_names=derived_names, derived_box=derived_box,
    )
