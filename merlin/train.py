import os
import shutil
import time
from pathlib import Path

import pytorch_lightning as pl
import swyft
from pytorch_lightning.loggers import CSVLogger

from .io import format_duration
from .network import Network
from .preprocessing import make_resampler
from .priors import resolve_inference_params


def _fmt_devices(trainer):
    kind = type(trainer.accelerator).__name__.replace("Accelerator", "") or "CPU"
    kind = {"CUDA": "GPU"}.get(kind, kind)
    n = trainer.num_devices
    return f"{n} {kind}{'s' if n != 1 else ''}"


class _EpochSummary(pl.Callback):
    """One print per epoch (train_loss/val_loss) instead of pytorch_lightning's
    default per-batch progress bar, which floods a non-interactive log file
    with a new line per step (its \\r-based live redraw only makes sense in a
    real terminal)."""

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            # PyTorch Lightning's pre-training sanity-check validation pass --
            # runs before any real training step, so train_loss doesn't exist
            # yet and val_loss is computed from untrained/lazily-uninitialized
            # weights (hence the harmless but confusing "Epoch 0 | val_loss=nan"
            # otherwise printed here). Not a real epoch; skip logging it.
            return
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")
        parts = [f"Epoch {trainer.current_epoch}"]
        if train_loss is not None:
            parts.append(f"train_loss={float(train_loss):.4g}")
        if val_loss is not None:
            parts.append(f"val_loss={float(val_loss):.4g}")
        print(" | ".join(parts), flush=True)


def train(store_samples, V_proj, config, num_workers=0, extra_callbacks=None):
    """
    Train the network on pre-processed store samples.

    The best checkpoint (by val_loss) is saved to config["STORES"]["checkpoint_path"]
    as "best.ckpt" (see Network.configure_callbacks) — reload it later with
    inference.predict_from_checkpoint to get predictions without retraining.
    Per-step/epoch train_loss/val_loss are logged to
    config["STORES"]["csv_logs"]/metrics.csv (pytorch-lightning's CSVLogger) for
    plotting the training curve later.

    Parameters
    ----------
    store_samples : swyft.Samples
    V_proj : np.ndarray
    config : dict
    num_workers : int — use 0 in notebooks, 12+ in slurm jobs
    extra_callbacks : list[pl.Callback] or None — appended after the built-in
        _EpochSummary callback, e.g. an Optuna pruning callback (see
        optuna_search.py) that needs the Trainer to report per-epoch val_loss.

    Returns
    -------
    network : Network
    trainer : swyft.SwyftTrainer
    """
    N_sims    = len(store_samples["noise"])
    N_total   = store_samples["noise"].shape[1]
    Nbin_ell  = config.get("Nbin_ell", 32)
    N_spectra = N_total // Nbin_ell

    network_cfg = config.get("NETWORK", {})
    _, param_indices = resolve_inference_params(config)
    network      = Network(
        V_proj, param_indices=param_indices,
        checkpoint_dir=config["STORES"]["checkpoint_path"],
        early_stopping_patience=config["TRAINING"].get("early_stopping", 5),
        learning_rate=config["TRAINING"]["learning_rate"],
        batch_size=config["TRAINING"]["batch_size"],
        num_feat_param=network_cfg.get("num_feat_param", 1),
        hidden_feat_compress=network_cfg.get("hidden_feat_compress", [2048, 512]),
        num_blocks_ratios=network_cfg.get("num_blocks_ratios", 6),
        dropout_ratios=network_cfg.get("dropout_ratios", 0.1),
        hidden_feat_ratios=network_cfg.get("hidden_feat_ratios", 64),
    )

    resampler = make_resampler(store_samples, N_sims, N_spectra, Nbin_ell)

    dm = swyft.SwyftDataModule(
        store_samples,
        num_workers=num_workers,
        batch_size=network.batch_size,
        val_fraction=config["TRAINING"]["val_fraction"],
        on_after_load_sample=resampler,
    )

    logger = CSVLogger(save_dir=config["STORES"]["csv_logs"], name="", version="")

    trainer = swyft.SwyftTrainer(
        accelerator=config["TRAINING"].get("accelerator", "auto"),
        devices=1,
        max_epochs=config["TRAINING"]["max_epochs"],
        precision=64,
        logger=logger,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[_EpochSummary(), *(extra_callbacks or [])],
    )

    params_to_infer = config["TRAINING"].get("params_to_infer", "COSMO")
    label = params_to_infer if isinstance(params_to_infer, str) else ", ".join(params_to_infer)
    print(f"Training network for {label} parameters...")
    t0 = time.time()
    trainer.fit(network, dm)
    elapsed = time.time() - t0

    print(f"Training finished after {trainer.current_epoch} epochs in "
          f"{format_duration(elapsed)} using {_fmt_devices(trainer)}")

    # CSVLogger always writes an hparams.yaml alongside metrics.csv, but
    # Network never registers any hyperparameters via save_hyperparameters(),
    # so it's always empty -- delete it rather than leave a useless, possibly
    # confusing file behind in train_<N>/.
    hparams_file = Path(config["STORES"]["csv_logs"]) / "hparams.yaml"
    hparams_file.unlink(missing_ok=True)

    # swyft.ZarrStore.__init__ unconditionally constructs a
    # zarr.ProcessSynchronizer, which recreates store_path+".sync" (a
    # write-lock coordination directory, not simulation data -- see the
    # matching cleanup in simulate.py) as a side effect of simply opening the
    # store, even for training's read-only access. Harmless to remove once
    # this training run is done reading; recreated automatically if the
    # store is opened again later.
    store_path = config["SIMULATION"]["store_path"]
    shutil.rmtree(store_path + ".sync", ignore_errors=True)
    lock_file = store_path + ".lock.file"
    if os.path.exists(lock_file):
        os.remove(lock_file)

    return network, trainer
