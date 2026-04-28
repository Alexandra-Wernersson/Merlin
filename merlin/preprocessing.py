"""
Preprocessing utilities: Cholesky whitening, PCA compression, and noise
resampling.

Corresponds to the transformation helpers in the original
``training_utils.py``.
"""

import os

import numpy as np
import torch
from scipy.linalg import solve_triangular


# ── Cholesky whitening ────────────────────────────────────────────────────────

def apply_cholesky_to_obs(obs, Lfid):
    """
    Apply the inverse Cholesky rotation to a single observation dict.

    Parameters
    ----------
    obs:
        Dict with keys ``'C_ells'`` and ``'noise'``, each of shape
        ``(N_spectra, N_bins)``.
    Lfid:
        Lower-triangular Cholesky factor of the fiducial covariance,
        shape ``(N_spectra, N_spectra)``.

    Returns
    -------
    oCells_chol, onoise_chol : ndarray
        Whitened arrays of shape ``(N_spectra, N_bins)``.
    """
    Linv = np.linalg.inv(Lfid)
    oCells_chol = (Linv @ obs["C_ells"].T).T
    onoise_chol = (Linv @ obs["noise"].T).T
    return oCells_chol, onoise_chol


def load_or_precompute_cholesky(store, Lfid, cache_dir):
    """
    Load cached Cholesky-whitened simulation arrays or compute and cache them.

    Parameters
    ----------
    store:
        Swyft sample store with arrays ``'C_ells'`` and ``'noise'``.
    Lfid:
        Cholesky factor of the fiducial covariance.
    cache_dir:
        Directory for caching ``Cells_chol.npy`` and ``noise_chol.npy``.

    Returns
    -------
    Cells_chol, noise_chol : ndarray
    """
    os.makedirs(cache_dir, exist_ok=True)
    cells_path = os.path.join(cache_dir, "Cells_chol.npy")
    noise_path = os.path.join(cache_dir, "noise_chol.npy")

    if os.path.exists(cells_path) and os.path.exists(noise_path):
        print("Loading cached Cholesky data")
        return (
            np.load(cells_path, mmap_mode="r"),
            np.load(noise_path, mmap_mode="r"),
        )

    print("Computing Cholesky transform (done once and cached)")
    assert store["C_ells"].shape[1] == Lfid.shape[0], (
        "Mismatch between simulation dimension and Lfid"
    )

    Cells_chol = solve_triangular(
        Lfid, store["C_ells"].T, lower=True, check_finite=False
    ).T
    noise_chol = solve_triangular(
        Lfid, store["noise"].T, lower=True, check_finite=False
    ).T

    np.save(cells_path, Cells_chol)
    np.save(noise_path, noise_chol)
    print(f"Saved to {cache_dir}")

    return Cells_chol, noise_chol


# ── PCA compression ───────────────────────────────────────────────────────────

def load_or_compute_pca(Cells_chol, config):
    """
    Compute a PCA projection matrix or load a previously saved one.

    The matrix is saved to ``config['PCA']['SVD']``.  Only modes whose
    variance fraction exceeds ``config['PCA']['variance_cut']`` are kept.

    Returns
    -------
    V_proj : ndarray, shape ``(n_data, n_components)``
    """
    pca_cfg = config["PCA"]
    store_PCA = pca_cfg.getboolean("store_pca", fallback=False)
    pca_file = pca_cfg["SVD"]
    q = int(pca_cfg["q"])
    var_cut = float(pca_cfg["variance_cut"])

    if store_PCA:
        print("Running PCA compression …")
        fCells_chol = torch.from_numpy(Cells_chol)
        _, S, V = torch.pca_lowrank(fCells_chol, q=q, center=True)

        Sn = (S / S.sum()) * 100
        V_proj = V[:, Sn > var_cut]

        np.save(pca_file, V_proj.numpy(), allow_pickle=True)
        print(f"PCA stored to: {pca_file}")

    V_proj = np.load(pca_file, allow_pickle=True)
    print(f"Kept {V_proj.shape[1]} PCA components")
    return V_proj


# ── Noise resampler ───────────────────────────────────────────────────────────

def make_resampler(store_samples, N_sims, N_spectra, Nbin_ell):
    """
    Return a resampler that draws independent noise realisations per spectrum.

    Using independent noise blocks across spectra decorrelates the noise from
    the signal during training, which improves generalisation.
    """
    noise_blocks = store_samples["noise"].reshape(N_sims, N_spectra, Nbin_ell)

    def resampler(x):
        i = np.random.randint(0, N_sims, size=N_spectra)
        sampled = noise_blocks[i, np.arange(N_spectra), :]
        x["noise"] = sampled.reshape(-1)
        return x

    return resampler
