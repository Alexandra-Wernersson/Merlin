import os
import shutil
import time

import numpy as np
import torch
import swyft
from scipy.linalg import solve_triangular

from .io import format_duration, load_array
from .simulator import sample_correlated_noise


# ============================================================
# Scale cuts
# ============================================================

def make_scale_cut_mask(config):
    """
    Build a float mask (1.0 keep, 0.0 cut) that zeros out ell bins above the
    per-probe cutoffs in config["ANALYSIS_VARIANTS"]["SCALE CUTS"].

    Data vector order: [WL (SHE-SHE), GGL (POS-SHE), GCph (POS-POS)]
    """
    sc = config.get("ANALYSIS_VARIANTS", {}).get("SCALE CUTS") or {}
    lmax_wl  = sc.get("SHE_SHE", np.inf)
    lmax_ggl = sc.get("POS_SHE", np.inf)
    lmax_gc  = sc.get("POS_POS", np.inf)

    ell_theory = load_array(config["CLOELIB_SETTINGS"]["ell"])
    n_bins     = config["CLOELIB_SETTINGS"]["Nbin_z"]
    n_wl       = n_bins * (n_bins + 1) // 2
    n_ggl      = n_bins * n_bins
    n_gc       = n_bins * (n_bins + 1) // 2

    mask = np.concatenate([
        np.tile((ell_theory <= lmax_wl ).astype(float), n_wl),
        np.tile((ell_theory <= lmax_ggl).astype(float), n_ggl),
        np.tile((ell_theory <= lmax_gc ).astype(float), n_gc),
    ])

    print(f"Scale cuts: {int(mask.sum())}/{len(mask)} data points retained "
          f"(WL ℓ≤{lmax_wl}, GGL ℓ≤{lmax_ggl}, GCph ℓ≤{lmax_gc})")
    return mask


# ============================================================
# Probe selection
# ============================================================

def select_data_probes(config):
    """
    Integer index array selecting which entries of the flattened 3x2pt data
    vector to keep, per config["ANALYSIS_VARIANTS"]["train_on_data"]:
      "3x2pt" (default) -> everything; returns None (no-op).
      "2x2pt"           -> GGL + GCph only (drops WL/shear-shear).
      "WL"              -> WL only.

    Unlike make_scale_cut_mask (a soft 0/1 mask, fixed length), this is a
    real selection meant for np.take(..., axis=-1): it shrinks the data
    vector fed into PCA/the network, to train on a strict subset of probes
    (analogous to TRAINING.params_to_infer for parameters).

    Data vector order: [WL (SHE-SHE)][GGL (POS-SHE)][GCph (POS-POS)].

    Note: "2x2pt" here is this project's own convention (GGL+GCph), not the
    literature convention of GCph auto + WL auto without their cross-term.

    Returns
    -------
    idx : np.ndarray of int, or None for "3x2pt"
    """
    mode = config.get("ANALYSIS_VARIANTS", {}).get("train_on_data", "3x2pt")
    if mode == "3x2pt":
        return None

    ell_theory = load_array(config["CLOELIB_SETTINGS"]["ell"])
    n_bins     = config["CLOELIB_SETTINGS"]["Nbin_z"]
    n_ell      = len(ell_theory)
    n_wl       = n_bins * (n_bins + 1) // 2
    n_ggl      = n_bins * n_bins
    n_gc       = n_bins * (n_bins + 1) // 2
    len_wl, len_ggl, len_gc = n_wl * n_ell, n_ggl * n_ell, n_gc * n_ell

    if mode == "WL":
        idx = np.arange(0, len_wl)
    elif mode == "2x2pt":
        idx = np.arange(len_wl, len_wl + len_ggl + len_gc)
    else:
        raise ValueError(
            f"ANALYSIS_VARIANTS.train_on_data: unknown mode {mode!r} — "
            "expected one of '3x2pt', '2x2pt', 'WL'"
        )
    return idx


def _n_data(config):
    """Total length of the flattened [WL][GGL][GCph] 3x2pt data vector."""
    ell_theory = load_array(config["CLOELIB_SETTINGS"]["ell"])
    n_bins     = config["CLOELIB_SETTINGS"]["Nbin_z"]
    n_wl       = n_bins * (n_bins + 1) // 2
    n_ggl      = n_bins * n_bins
    n_gc       = n_bins * (n_bins + 1) // 2
    return len(ell_theory) * (n_wl + n_ggl + n_gc)


def combined_keep_indices(config):
    """
    Sorted int array of data-vector indices surviving BOTH
    ANALYSIS_VARIANTS.SCALE CUTS and .train_on_data, or None if neither
    restricts anything (equivalent to keeping the full vector).

    Both need to be resolved together, and BEFORE whitening (see
    compute_cut_lfid/apply_cholesky) rather than the vector being whitened
    in full and then zeroed/selected afterward: Cholesky whitening (L^-1 @
    data) is a triangular mixing transform, so a later kept entry's
    whitened value already depends on the raw value of any earlier entry
    -- cut or not. Zeroing entries post-whitening doesn't cleanly remove
    their contribution, and doesn't re-normalize the surviving entries for
    the covariance sub-block they actually now represent. Cutting the raw
    vector first and whitening with the Cholesky factor of the
    correctly-conditioned covariance sub-block is the statistically
    correct order.
    """
    variants = config.get("ANALYSIS_VARIANTS", {})
    n_data = _n_data(config)

    keep = None
    if variants.get("SCALE CUTS"):
        mask = make_scale_cut_mask(config).astype(bool)
        if not mask.all():  # lmax >= every ell value actually excludes nothing
            keep = mask

    probe_idx = select_data_probes(config)
    if probe_idx is not None:
        probe_mask = np.zeros(n_data, dtype=bool)
        probe_mask[probe_idx] = True
        keep = probe_mask if keep is None else (keep & probe_mask)

    return None if keep is None else np.where(keep)[0]


def compute_cut_lfid(config, keep_idx):
    """
    Cholesky factor of the covariance restricted to keep_idx -- the
    correct whitening basis for a scale-cut/probe-selected data vector.
    """
    cov = np.load(config["CLOELIB_SETTINGS"]["covmat"])["Gauss"]
    sub = cov[np.ix_(keep_idx, keep_idx)]
    return np.linalg.cholesky(sub)


def kept_segments(config):
    """
    [(n_blocks, block_size), ...] describing the kept data vector's block
    structure, in order over whichever of [WL, GGL, GCph] survive
    ANALYSIS_VARIANTS.train_on_data -- block_size = surviving ell count for
    that probe type under SCALE CUTS (uniform within a probe type, but not
    necessarily across types, since each has its own lmax), n_blocks =
    that type's tomographic-pair count. sum(n_blocks*block_size) equals
    the kept vector's actual length (matches combined_keep_indices), used
    by make_resampler to reshape/shuffle noise blocks correctly whether or
    not cuts are active.
    """
    variants = config.get("ANALYSIS_VARIANTS", {})
    mode = variants.get("train_on_data", "3x2pt")
    sc   = variants.get("SCALE CUTS") or {}

    ell_theory = load_array(config["CLOELIB_SETTINGS"]["ell"])
    n_bins     = config["CLOELIB_SETTINGS"]["Nbin_z"]
    n_wl       = n_bins * (n_bins + 1) // 2
    n_ggl      = n_bins * n_bins
    n_gc       = n_bins * (n_bins + 1) // 2

    probes = []
    if mode in ("3x2pt", "WL"):
        probes.append((n_wl, sc.get("SHE_SHE", np.inf)))
    if mode in ("3x2pt", "2x2pt"):
        probes.append((n_ggl, sc.get("POS_SHE", np.inf)))
        probes.append((n_gc,  sc.get("POS_POS", np.inf)))

    return [(n_blocks, int((ell_theory <= lmax).sum())) for n_blocks, lmax in probes]


# ============================================================
# Cholesky whitening
# ============================================================

def apply_cholesky(data, Lfid):
    """Cholesky-whiten a dict or store with keys 'C_ells' and 'noise'."""
    Cells_chol = solve_triangular(Lfid, data["C_ells"].T, lower=True, check_finite=False).T
    noise_chol = solve_triangular(Lfid, data["noise"].T,  lower=True, check_finite=False).T
    return Cells_chol, noise_chol


def load_or_precompute_cholesky(store, Lfid, cache_dir):
    """Load from cache or compute and cache Cholesky-whitened simulations."""
    os.makedirs(cache_dir, exist_ok=True)
    cells_path = os.path.join(cache_dir, "Cells_chol.npy")
    noise_path = os.path.join(cache_dir, "noise_chol.npy")

    if os.path.exists(cells_path) and os.path.exists(noise_path):
        print("Loading cached Cholesky data")
        return np.load(cells_path, mmap_mode="r"), np.load(noise_path, mmap_mode="r")

    print("Computing Cholesky transform (done once, then cached)")
    Cells_chol, noise_chol = apply_cholesky(store, Lfid)
    np.save(cells_path, Cells_chol)
    np.save(noise_path, noise_chol)
    print(f"Cached to {cache_dir}")
    return Cells_chol, noise_chol


# ============================================================
# PCA compression
# ============================================================

def load_or_compute_pca(Cells_chol, config):
    """
    Compute or load PCA projection matrix.

    config["PCA"]["recompute_pca"] controls the behaviour:
    - true: recompute from Cells_chol, save to PCA.SVD (train_<N>/aux_files/SVD.npy).
    - false, PCA.pca_file set: copy that file to PCA.SVD, then load it — reuse
      a PCA already computed for another train_<N> against the same store
      (valid only if nothing affecting the basis, e.g. ANALYSIS_VARIANTS or
      PCA.q/variance_cut, differs). Keeps each train_<N>'s own SVD.npy
      self-contained rather than pointing elsewhere.
    - false, PCA.pca_file unset: load directly from PCA.SVD (must exist).
    """
    pca_cfg  = config["PCA"]
    pca_file = pca_cfg["SVD"]
    pca_source = pca_cfg.get("pca_file")
    q        = pca_cfg["q"]
    var_cut  = pca_cfg["variance_cut"]

    if pca_cfg["recompute_pca"]:
        print("Doing PCA compression...")
        t0 = time.time()
        _, S, V = torch.pca_lowrank(torch.from_numpy(Cells_chol), q=q, center=True)
        V_proj  = V[:, (S / S.sum()) * 100 > var_cut]
        elapsed = time.time() - t0
        np.save(pca_file, V_proj.numpy(), allow_pickle=True)
        print(f"PCA compression finished in {format_duration(elapsed)} "
              f"on {Cells_chol.shape[0]} simulations (randomized low-rank SVD, CPU)")
        print(f"PCA stored to {pca_file}")
    elif pca_source:
        if os.path.abspath(pca_source) != os.path.abspath(pca_file):
            print(f"Copying PCA projection from {pca_source}")
            shutil.copy(pca_source, pca_file)
        else:
            print(f"PCA projection already at {pca_file} (pca_file points to itself)")
    else:
        print(f"Loading cached PCA projection from {pca_file}")

    V_proj = np.load(pca_file, allow_pickle=True)
    print(f"Using {V_proj.shape[1]} PCA components")
    return V_proj


# ============================================================
# Store preprocessing (formerly data.py)
# ============================================================

def preprocess(store, Lfid, config):
    """
    Cholesky-whiten simulations, apply scale cuts, select which 3x2pt probes
    to keep, compute PCA — all controlled by config["ANALYSIS_VARIANTS"] and
    applied here (not baked into the store) so different train_<N> runs
    against the same store can vary them independently without re-simulating.

    ANALYSIS_VARIANTS.train_on_data ("3x2pt"/"2x2pt"/"WL", default "3x2pt")
    and SCALE CUTS are combined into one set of surviving indices and cut
    from the RAW data vector before whitening (see
    combined_keep_indices/compute_cut_lfid) — the statistically correct
    order, rather than whitening the full vector and zeroing/selecting
    afterward.

    If ANALYSIS_VARIANTS.regenerate_noise_samples is true, the store's noise
    is discarded and redrawn from the current Lfid (may differ from the
    covmat active at simulation time), e.g. to retrain against a different
    noise covariance without re-running the expensive C_ells physics. Not
    persisted — cheap to redo, and reproducible from Lfid. Uses
    simulator.sample_correlated_noise (same function Simulator.get_sample_noise
    calls) so the distribution matches the simulator's exactly.

    Returns
    -------
    store_samples : swyft.Samples
    V_proj : np.ndarray
    """
    variants = config.get("ANALYSIS_VARIANTS", {})
    print(f"Training on {variants.get('train_on_data', '3x2pt')} data")

    keep_idx = combined_keep_indices(config)
    Lfid_use = compute_cut_lfid(config, keep_idx) if keep_idx is not None else Lfid
    C_ells   = store["C_ells"] if keep_idx is None else np.take(store["C_ells"], keep_idx, axis=-1)

    if variants.get("regenerate_noise_samples", False):
        n_sims = store["C_ells"].shape[0]
        t0 = time.time()
        noise = sample_correlated_noise(Lfid_use, shape=(n_sims,))
        elapsed = time.time() - t0
        print(f"Regenerated {n_sims} noise samples from the current Lfid "
              f"in {format_duration(elapsed)}")
    else:
        noise = store["noise"] if keep_idx is None else np.take(store["noise"], keep_idx, axis=-1)

    print("Whitening spectra (Cholesky transform)...")
    t0 = time.time()
    Cells_chol, noise_chol = apply_cholesky({"C_ells": C_ells, "noise": noise}, Lfid_use)
    elapsed = time.time() - t0
    print(f"Whitening finished in {format_duration(elapsed)} "
          f"on {Cells_chol.shape[0]} simulations (CPU)")

    if keep_idx is not None:
        print(f"Kept {len(keep_idx)}/{_n_data(config)} data points after "
              f"scale cuts + probe selection")

    V_proj = load_or_compute_pca(Cells_chol, config)

    samples = {
        "z": store["z"],
        "C_ells": Cells_chol,
        "noise": noise_chol,
    }
    # Present only if the store was simulated with CLOELIB_SETTINGS.add_derived
    # set; propagated so Network.forward can select derived-space
    # params_to_infer entries (priors.resolve_inference_params).
    if "derived" in store:
        samples["derived"] = store["derived"]

    store_samples = swyft.Samples(**samples)
    return store_samples, V_proj


def preprocess_obs(obs, Lfid, config=None):
    """
    Cholesky-whiten a single observation and return a noiseless swyft.Sample.
    Pass config to apply the same scale cuts / probe selection used during
    training (ANALYSIS_VARIANTS.SCALE CUTS / train_on_data), required for the
    result to match the width the trained network expects; see preprocess.
    """
    print("Whitening observation (Cholesky transform)...")
    keep_idx = combined_keep_indices(config) if config is not None else None

    if keep_idx is None:
        Cells_chol, noise_chol = apply_cholesky(obs, Lfid)
    else:
        Lfid_cut = compute_cut_lfid(config, keep_idx)
        cut_obs = {
            "C_ells": np.take(obs["C_ells"], keep_idx, axis=-1),
            "noise":  np.take(obs["noise"],  keep_idx, axis=-1),
        }
        Cells_chol, noise_chol = apply_cholesky(cut_obs, Lfid_cut)
        print(f"Kept {len(keep_idx)}/{_n_data(config)} data points after "
              f"scale cuts + probe selection")

    return swyft.Sample(dict(C_ells=Cells_chol, noise=0.0 * noise_chol))


# ============================================================
# Noise resampler
# ============================================================

def make_resampler(store_samples, N_sims, config):
    """
    Randomly mix noise blocks (per tomographic-pair "spectrum") across
    simulations to prevent overfitting to a specific noise realization.
    Block boundaries follow kept_segments, so this is correct whether or
    not ANALYSIS_VARIANTS.SCALE CUTS/train_on_data are active -- a scale
    cut can give WL a different surviving ell count than GGL/GCph, so
    blocks aren't necessarily uniform-length across the whole vector.
    """
    noise = store_samples["noise"]
    segments = kept_segments(config)
    assert sum(n * b for n, b in segments) == noise.shape[1], (
        f"kept_segments {segments} don't sum to noise width {noise.shape[1]} "
        "-- ANALYSIS_VARIANTS mismatch between preprocessing and training"
    )

    blocks_per_segment = []
    offset = 0
    for n_blocks, block_size in segments:
        length = n_blocks * block_size
        blocks_per_segment.append((offset, noise[:, offset:offset + length].reshape(N_sims, n_blocks, block_size)))
        offset += length

    def resampler(x):
        out = np.empty(noise.shape[1], dtype=noise.dtype)
        for (n_blocks, block_size), (seg_offset, blocks) in zip(segments, blocks_per_segment):
            i = np.random.randint(0, N_sims, size=n_blocks)
            out[seg_offset:seg_offset + n_blocks * block_size] = (
                blocks[i, np.arange(n_blocks), :].reshape(-1)
            )
        x["noise"] = out
        return x

    return resampler
