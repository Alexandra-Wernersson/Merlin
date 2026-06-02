import numpy as np
from scipy import stats
import swyft

from .io import save_predictions as _save
from .params import resolve_params


def infer(trainer, network, obs_sample, config, n_samples=500_000, scale=5):
    """
    Run inference given a trained network and a Cholesky-whitened observation.

    The parameters to infer are read from config["INFERENCE"]["params"]
    (defaults to "COSMO").
    """
    fiducial = list(config["FIDUCIAL VALUES"].values())
    Finv     = np.load(config["FINV"]["finv_file"])
    sigmas   = np.sqrt(np.diag(Finv))

    param_spec = config.get("INFERENCE", {}).get("params", "COSMO")
    _, param_indices = resolve_params(param_spec)

    lower = np.array([fiducial[i] - scale * sigmas[i] for i in param_indices])
    width = np.array([2 * scale * sigmas[i]            for i in param_indices])

    prior_samples = swyft.Samples(
        z=stats.uniform(lower, width).rvs(size=(n_samples, len(param_indices)))
    )

    predictions = trainer.infer(network, obs_sample, prior_samples)
    _save(predictions, config["STORES"]["predictions_cosmo"])
    return predictions
