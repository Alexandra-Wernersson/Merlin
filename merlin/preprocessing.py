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
        print(f"Copying PCA projection from {pca_source}")
        shutil.copy(pca_source, pca_file)
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
    slices the data vector to the selected probe block(s) (select_data_probes),
    applied after SCALE CUTS so a cut in a dropped block is moot.

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

    if variants.get("regenerate_noise_samples", False):
        n_sims = store["C_ells"].shape[0]
        t0 = time.time()
        noise = sample_correlated_noise(Lfid, shape=(n_sims,))
        elapsed = time.time() - t0
        print(f"Regenerated {n_sims} noise samples from the current Lfid "
              f"in {format_duration(elapsed)}")
    else:
        noise = store["noise"]

    print("Whitening spectra (Cholesky transform)...")
    t0 = time.time()
    Cells_chol, noise_chol = apply_cholesky(
        {"C_ells": store["C_ells"], "noise": noise}, Lfid
    )
    elapsed = time.time() - t0
    print(f"Whitening finished in {format_duration(elapsed)} "
          f"on {Cells_chol.shape[0]} simulations (CPU)")

    if variants.get("SCALE CUTS"):
        mask       = make_scale_cut_mask(config)
        Cells_chol = Cells_chol * mask
        noise_chol = noise_chol * mask

    print(f"Training on {variants.get('train_on_data', '3x2pt')} data")
    data_idx = select_data_probes(config)
    if data_idx is not None:
        n_before   = Cells_chol.shape[-1]
        Cells_chol = np.take(Cells_chol, data_idx, axis=-1)
        noise_chol = np.take(noise_chol, data_idx, axis=-1)
        print(f"train_on_data={variants.get('train_on_data')!r}: "
              f"{Cells_chol.shape[-1]}/{n_before} data points kept")

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
    Cells_chol, noise_chol = apply_cholesky(obs, Lfid)

    if config is not None and config.get("ANALYSIS_VARIANTS", {}).get("SCALE CUTS"):
        mask       = make_scale_cut_mask(config)
        Cells_chol = Cells_chol * mask

    if config is not None:
        data_idx = select_data_probes(config)
        if data_idx is not None:
            Cells_chol = np.take(Cells_chol, data_idx, axis=-1)
            noise_chol = np.take(noise_chol, data_idx, axis=-1)

    return swyft.Sample(dict(C_ells=Cells_chol, noise=0.0 * noise_chol))


# ============================================================
# Noise resampler
# ============================================================

def make_resampler(store_samples, N_sims, N_spectra, Nbin_ell):
    """Randomly mix noise blocks across simulations to prevent overfitting."""
    noise_blocks = store_samples["noise"].reshape(N_sims, N_spectra, Nbin_ell)

    def resampler(x):
        i = np.random.randint(0, N_sims, size=N_spectra)
        x["noise"] = noise_blocks[i, np.arange(N_spectra), :].reshape(-1)
        return x

    return resampler
