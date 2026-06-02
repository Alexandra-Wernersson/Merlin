import numpy as np
import torch
import swyft

from .network import Network
from .params import resolve_params
from .preprocessing import make_resampler


def train(store_samples, V_proj, config, num_workers=0):
    """
    Train the network on pre-processed store samples.

    Parameters
    ----------
    store_samples : swyft.Samples
    V_proj : np.ndarray
    config : dict
    num_workers : int — use 0 in notebooks, 12+ in slurm jobs

    Returns
    -------
    network : Network
    trainer : swyft.SwyftTrainer
    """
    N_sims    = len(store_samples["noise"])
    N_total   = store_samples["noise"].shape[1]
    Nbin_ell  = config.get("Nbin_ell", 32)
    N_spectra = N_total // Nbin_ell

    param_spec   = config.get("INFERENCE", {}).get("params", "COSMO")
    _, param_indices = resolve_params(param_spec)
    network      = Network(V_proj, param_indices=param_indices)

    resampler = make_resampler(store_samples, N_sims, N_spectra, Nbin_ell)

    dm = swyft.SwyftDataModule(
        store_samples,
        num_workers=num_workers,
        batch_size=network.batch_size,
        val_fraction=config["TRAINING"]["val_fraction"],
        on_after_load_sample=resampler,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = swyft.SwyftTrainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=config["TRAINING"]["max_epochs"],
        precision=64,
    )

    trainer.fit(network, dm)
    return network, trainer
