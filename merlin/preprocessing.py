import os
import numpy as np
import torch
import swyft
from scipy.linalg import solve_triangular


# ============================================================
# Scale cuts
# ============================================================

def make_scale_cut_mask(config):
    """
    Build a float mask that zeros out ell bins above the per-probe cutoffs
    defined in config["SCALE CUTS"].

    Data vector order: [WL (SHE-SHE), GGL (POS-SHE), GCph (POS-POS)]

    Returns
    -------
    mask : np.ndarray, shape (N_data,)  — 1.0 to keep, 0.0 to cut
    """
    sc       = config.get("SCALE CUTS") or {}
    lmax_wl  = sc.get("SHE_SHE", np.inf)
    lmax_ggl = sc.get("POS_SHE", np.inf)
    lmax_gc  = sc.get("POS_POS", np.inf)

    ell_theory = np.load(config["AUX FILES"]["ell_file"])
    n_bins     = config["FINV"]["Nbin_z"]
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
    """Compute or load PCA projection matrix."""
    pca_cfg  = config["PCA"]
    pca_file = pca_cfg["SVD"]
    q        = pca_cfg["q"]
    var_cut  = pca_cfg["variance_cut"]

    if pca_cfg["store_pca"]:
        print("Running PCA compression...")
        _, S, V = torch.pca_lowrank(torch.from_numpy(Cells_chol), q=q, center=True)
        V_proj  = V[:, (S / S.sum()) * 100 > var_cut]
        np.save(pca_file, V_proj.numpy(), allow_pickle=True)
        print(f"PCA stored to {pca_file}")

    V_proj = np.load(pca_file, allow_pickle=True)
    print(f"Using {V_proj.shape[1]} PCA components")
    return V_proj


# ============================================================
# Store preprocessing (formerly data.py)
# ============================================================

def preprocess(store, Lfid, config):
    """
    Cholesky-whiten simulations, apply scale cuts, compute PCA.

    Returns
    -------
    store_samples : swyft.Samples
    V_proj : np.ndarray
    """
    Cells_chol, noise_chol = apply_cholesky(
        {"C_ells": store["C_ells"], "noise": store["noise"]}, Lfid
    )

    if config.get("SCALE CUTS"):
        mask       = make_scale_cut_mask(config)
        Cells_chol = Cells_chol * mask
        noise_chol = noise_chol * mask

    V_proj = load_or_compute_pca(Cells_chol, config)

    store_samples = swyft.Samples(
        z=store["z"],
        C_ells=Cells_chol,
        noise=noise_chol,
    )
    return store_samples, V_proj


def preprocess_obs(obs, Lfid, config=None):
    """
    Cholesky-whiten a single observation and return a noiseless swyft.Sample.
    Pass config to apply the same scale cuts used during training.
    """
    Cells_chol, noise_chol = apply_cholesky(obs, Lfid)

    if config is not None and config.get("SCALE CUTS"):
        mask       = make_scale_cut_mask(config)
        Cells_chol = Cells_chol * mask

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
