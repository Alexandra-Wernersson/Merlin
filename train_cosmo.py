import os, sys, time, configparser
from datetime import datetime

import numpy as np
import torch
from scipy.linalg import solve_triangular
from scipy import stats
from training_utils import Network, make_resampler, apply_cholesky_to_obs, save_predictions
import swyft


# =========================
# Utilities
# =========================
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] | {msg}", flush=True)


def load_or_precompute_cholesky(store, Lfid, cache_dir):
    """Load precomputed data or compute once and save."""
    os.makedirs(cache_dir, exist_ok=True)

    cells_path = os.path.join(cache_dir, "Cells_chol.npy")
    noise_path = os.path.join(cache_dir, "noise_chol.npy")

    if os.path.exists(cells_path) and os.path.exists(noise_path):
        log("Loading cached Cholesky data")
        Cells_chol = np.load(cells_path, mmap_mode="r")
        noise_chol = np.load(noise_path, mmap_mode="r")
    else:
        log("Computing Cholesky transform (this is slow, done once)")

        Cells_chol = solve_triangular(
            Lfid, store['C_ells'].T, lower=True, check_finite=False
        ).T

        noise_chol = solve_triangular(
            Lfid, store['noise'].T, lower=True, check_finite=False
        ).T

        np.save(cells_path, Cells_chol)
        np.save(noise_path, noise_chol)

    return Cells_chol, noise_chol


def build_prior(fiducial, sigmas, n_params, n_samples=500_000, scale=6):
    lower = np.array([fiducial[i] - scale * sigmas[i] for i in range(n_params)])
    width = np.array([2 * scale * sigmas[i] for i in range(n_params)])

    samples = stats.uniform(lower, width).rvs(size=(n_samples, n_params))
    return swyft.Samples(z=samples)


# =========================
# Main
# =========================
if __name__ == "__main__":

    # === Config ===
    if len(sys.argv) < 2:
        print("Usage: python train_cosmo.py <config_file>")
        sys.exit(1)

    config_path = sys.argv[1]
    log(f"Reading config: {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path)

    # === Load store ===
    store_path = config['SIMULATION']['store_path']
    store = swyft.ZarrStore(store_path).get_sample_store()

    # === Load auxiliary ===
    Lfid = np.load(config['AUX FILES']['LFID'],allow_pickle=True)

    # === Precompute / load compressed data ===
    cache_dir = config['STORES'].get("cache_dir", "/gpfs/scratch1/shared/awernersson/swyft_cloelib/cache_newsim")
    Cells_chol, noise_chol = load_or_precompute_cholesky(store, Lfid, cache_dir)

    # === Build dataset ===
    log("Building dataset")
    store_samples = swyft.Samples(
        z=torch.tensor(store['z'], dtype=torch.float64),
        C_ells=torch.tensor(Cells_chol, dtype=torch.float64),
        noise=torch.tensor(noise_chol, dtype=torch.float64),
    )
    # === Network ===
    network = Network()

    # === DataModule ===
    log("Preparing dataloader")
    N_sims = len(store_samples['noise'])
    N_total = store_samples['noise'].shape[1]

    Nbin_ell = 32
    assert N_total % Nbin_ell == 0

    N_spectra = N_total // Nbin_ell
    dm = swyft.SwyftDataModule(
        store_samples,
        num_workers=12,  # increased
        batch_size=network.batch_size,
        val_fraction=0.2,
        on_after_load_sample=make_resampler(
            store_samples,
            N_sims,
            N_spectra,
            Nbin_ell
        )
    )

    # === Trainer ===
    trainer = swyft.SwyftTrainer(
        accelerator='gpu',
        devices=1,
        max_epochs=-1,
        precision=64
    )

    # === Train ===
    log("Starting training")
    t0 = time.time()
    trainer.fit(network, dm)
    log(f"Training done in {(time.time()-t0)/60:.1f} min")

    # =========================
    # Inference
    # =========================

    # === Load fiducial + covariance ===
    fiducial = [float(v) for v in config['FIDUCIAL VALUES'].values()]
    Finv = np.load(config['FINV']['finv_file'])
    sigmas = np.sqrt(np.diag(Finv))

    # === Observation ===
    log("Preparing observation")
    obs = np.load(config['AUX FILES']['OBS'], allow_pickle=True).item()
    oCells_chol = np.zeros((N_spectra*Nbin_ell))
    onoise_chol = np.zeros((N_spectra*Nbin_ell))

    oCells_chol = solve_triangular(Lfid, obs['C_ells'].T, lower=True, check_finite=False).T
    onoise_chol = solve_triangular(Lfid, obs['noise'].T, lower=True, check_finite=False).T
    
    #oCells_chol, onoise_chol = apply_cholesky_to_obs(obs, Lfid)

    obs_sample = swyft.Sample(
        dict(C_ells=oCells_chol, noise=0.0 * onoise_chol)
    )

    # === Prior ===
    log("Sampling prior")
    prior_samples = build_prior(
        fiducial,
        sigmas,
        network.num_params_show,
        n_samples=500_000,
        scale=6
    )

    # === Inference ===
    log("Running inference")
    predictions = trainer.infer(network, obs_sample, prior_samples)

    # === Save ===
    save_path = config['STORES']['predictions_cosmo']
    save_predictions(predictions, save_path)

    log(f"Saved predictions to {save_path}")
