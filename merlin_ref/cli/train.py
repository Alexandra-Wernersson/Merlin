"""
CLI entry point: train the network and run inference.

Usage::

    merlin-train <config.ini>
"""

import sys
import time
import configparser
from datetime import datetime

import numpy as np
import torch
from scipy.linalg import solve_triangular
from pytorch_lightning.callbacks import ModelCheckpoint
import swyft

from merlin.network import Network
from merlin.preprocessing import load_or_compute_pca, make_resampler, load_or_precompute_cholesky
from merlin.inference import build_prior, run_coverage_test
from merlin.io import save_predictions


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] | {msg}", flush=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: merlin-train <config.ini>")
        sys.exit(1)

    config_path = sys.argv[1]
    log(f"Reading config: {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path)

    # ── 1. load simulation store ──────────────────────────────────────────────
    store_path = config["SIMULATION"]["store_path"]
    store = swyft.ZarrStore(store_path).get_sample_store()

    # ── 2. Cholesky whitening ─────────────────────────────────────────────────
    Lfid = np.load(config["OBSERVATION"]["LFID"])

    cache_dir = config["STORES"].get("cache_dir", "cache")
    Cells_chol, noise_chol = load_or_precompute_cholesky(store, Lfid, cache_dir)

    # ── 3. PCA compression ────────────────────────────────────────────────────
    V_proj = load_or_compute_pca(Cells_chol, config)

    # ── 4. build dataset ──────────────────────────────────────────────────────
    log("Building dataset")
    store_samples = swyft.Samples(
        z=torch.tensor(store["z"], dtype=torch.float64),
        C_ells=torch.tensor(Cells_chol, dtype=torch.float64),
        noise=torch.tensor(noise_chol, dtype=torch.float64),
    )

    N_sims = len(store_samples["noise"])
    N_total = store_samples["noise"].shape[1]
    Nbin_ell = int(config["FINV"].get("N_ell", 32))
    assert N_total % Nbin_ell == 0, (
        f"noise shape {N_total} is not divisible by Nbin_ell={Nbin_ell}"
    )
    N_spectra = N_total // Nbin_ell

    # ── 5. network ────────────────────────────────────────────────────────────
    network = Network(V_proj)

    # ── 6. data module ────────────────────────────────────────────────────────
    log("Preparing dataloader")
    dm = swyft.SwyftDataModule(
        store_samples,
        num_workers=12,
        batch_size=network.batch_size,
        val_fraction=0.2,
        on_after_load_sample=make_resampler(store_samples, N_sims, N_spectra, Nbin_ell),
    )

    # ── 7. trainer ────────────────────────────────────────────────────────────
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["STORES"]["checkpoint_path"],
        filename="best_model",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    trainer = swyft.SwyftTrainer(
        accelerator="gpu",
        devices=1,
        max_epochs=int(config["TRAINING"]["max_epochs"]),
        precision=64,
        callbacks=[checkpoint_callback],
    )

    # ── 8. train ──────────────────────────────────────────────────────────────
    log("Starting training")
    t0 = time.time()
    trainer.fit(network, dm)
    log(f"Training done in {(time.time() - t0) / 60:.1f} min")
    log(f"Best model: {checkpoint_callback.best_model_path}")

    # ── 9. inference ──────────────────────────────────────────────────────────
    fiducial = [float(v) for v in config["FIDUCIAL VALUES"].values()]
    Finv = np.load(config["FINV"]["finv_file"])
    sigmas = np.sqrt(np.diag(Finv))

    log("Preparing observation")
    obs = np.load(
        config["OBSERVATION"]["OBS_CHOLESKY_NOISELESS"], allow_pickle=True
    ).item()
    obs_sample = swyft.Sample(obs)

    log("Sampling prior")
    prior_samples = build_prior(fiducial, sigmas, network.num_params_show)

    log("Running inference")
    predictions = trainer.infer(network, obs_sample, prior_samples)

    save_path = config["STORES"]["predictions_cosmo"]
    save_predictions(predictions, save_path)
    log(f"Saved predictions to {save_path}")

    # ── 10. optional coverage test ────────────────────────────────────────────
    if config["TRAINING"].getboolean("coverage_test", fallback=False):
        log("Running coverage test")
        coverage_path = config["STORES"]["coverage_plot"]
        run_coverage_test(trainer, network, store_samples, fiducial, sigmas, coverage_path)


if __name__ == "__main__":
    main()
