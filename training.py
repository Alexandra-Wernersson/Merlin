import os, sys, time, configparser
from datetime import datetime

import numpy as np
import torch
from scipy.linalg import solve_triangular
from scipy import stats
from pytorch_lightning.callbacks import ModelCheckpoint
from training_utils import Network, load_or_compute_pca, make_resampler, apply_cholesky_to_obs, save_predictions, load_or_precompute_cholesky, run_coverage_test
import swyft


# =========================
# Utilities
# =========================
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] | {msg}", flush=True)


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

    store_path = config["SIMULATION"]["store_path"]
    store = swyft.ZarrStore(
        store_path
    ).get_sample_store()


    # === Load auxiliary ===
    Lfid = np.load(config['OBSERVATION']['LFID'])

    # === Whitening ===

    Cells_chol = solve_triangular(
        Lfid,
        store["C_ells"].T,
        lower = True,
        check_finite = False
    ).T


    # === PCA ===

    V_proj = load_or_compute_pca(
        Cells_chol,
        config

    )

    # === Precompute / load compressed data ===
    cache_dir = config['STORES'].get("cache_dir", "/gpfs/scratch1/shared/awernersson/swyft_cloelib/cache_newgen")
    _, noise_chol = load_or_precompute_cholesky(store, Lfid, cache_dir)

    # === Build dataset ===
    log("Building dataset")
    store_samples = swyft.Samples(
        z=torch.tensor(store['z'], dtype=torch.float64),
        C_ells=torch.tensor(Cells_chol, dtype=torch.float64),
        noise=torch.tensor(noise_chol, dtype=torch.float64),
    )

    # === Network ===
    N_sims = len(store_samples['noise'])
    N_total = store_samples['noise'].shape[1]
    Nbin_ell = 32
    assert N_total % Nbin_ell == 0
    N_spectra = N_total // Nbin_ell
    C_ells = np.zeros((N_sims, N_spectra*Nbin_ell))
    #Cells_chol = np.zeros((N_sims,N_spectra*Nbin_ell))
    #Cells_chol = solve_triangular(Lfid, store_samples['C_ells'].T, lower=True, check_finite=False).T

    #Vh_proj = load_or_compute_pca(
    #    Cells_chol,
    #    config
    #)
    
    print(
        "max |V|:",
        np.abs(V_proj).max()
    )

    print(
        "min |V|:",
        np.abs(V_proj).min()
    )
    network = Network(V_proj)

    # === DataModule ===
    log("Preparing dataloader")

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
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['STORES']['checkpoint_path'],
        filename='best_model',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
    )

    trainer = swyft.SwyftTrainer(
        accelerator='gpu',
        devices=1,
        max_epochs=int(config["TRAINING"]["max_epochs"]),
        precision=64,
        callbacks=[checkpoint_callback],
    )

    # === Train ===
    log("Starting training")
    t0 = time.time()
    trainer.fit(network, dm)
    log(f"Training done in {(time.time()-t0)/60:.1f} min")
    log(f"Best model: {checkpoint_callback.best_model_path}")

    # =========================
    # Inference
    # =========================

    # === Load fiducial + covariance ===
    fiducial = [float(v) for v in config['FIDUCIAL VALUES'].values()]
    Finv = np.load(config['FINV']['finv_file'])
    sigmas = np.sqrt(np.diag(Finv))

    # === Observation ===
    log("Preparing observation")
    obs = np.load(config['OBSERVATION']['OBS_CHOLESKY_NOISELESS'],
              allow_pickle=True).item()
    print("\n--- consistency check ---")

    print("train Cells_chol mean:", np.mean(Cells_chol))
    print("train noise_chol mean:", np.mean(noise_chol))

    print("obs Cells_chol mean:", np.mean(obs["C_ells"]))
    print("obs noise_chol mean:", np.mean(obs["noise"]))

    print("-------------------------\n")
    print("train noise std:", np.std(noise_chol))
    print("obs noise std:", np.std(obs["noise"]))
    obs_sample = swyft.Sample(obs)
    
    # === Prior ===
    log("Sampling prior")
    prior_samples = build_prior(
        fiducial,
        sigmas,
        network.num_params_show,
        n_samples=500_000,
        scale=5
    )

    # === Inference ===
    log("Running inference")
    predictions = trainer.infer(network, obs_sample, prior_samples)

    # === Save ===
    save_path = config['STORES']['predictions_cosmo']
    save_predictions(predictions, save_path)

    log(f"Saved predictions to {save_path}")

    # === Coverage test ===
    if config["TRAINING"].get("coverage_test", "False") == "True":
        log("Running coverage test")
        coverage_path = config["STORES"]["coverage_plot"]
        run_coverage_test(trainer, network, store_samples, fiducial, sigmas, coverage_path)
