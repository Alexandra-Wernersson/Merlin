"""
Inference utilities: prior sampling and coverage testing.
"""

import numpy as np
import swyft
from scipy import stats


def build_prior(fiducial, sigmas, n_params, n_samples=500_000, scale=5):
    """
    Build a swyft.Samples prior by drawing from a uniform hypercube centred
    on *fiducial* with half-width ``scale * sigmas``.

    Parameters
    ----------
    fiducial : array-like
        Fiducial parameter values.
    sigmas : array-like
        Marginal 1-sigma uncertainties (e.g., from Fisher matrix diagonal).
    n_params : int
        Number of parameters to include (first *n_params* entries).
    n_samples : int
        Number of prior samples to draw.
    scale : float
        Half-width of the uniform prior in units of *sigmas*.

    Returns
    -------
    swyft.Samples
    """
    lower = np.array([fiducial[i] - scale * sigmas[i] for i in range(n_params)])
    width = np.array([2 * scale * sigmas[i] for i in range(n_params)])
    samples = stats.uniform(lower, width).rvs(size=(n_samples, n_params))
    return swyft.Samples(z=samples)


def run_coverage_test(trainer, network, store_samples, fiducial, sigmas, save_path):
    """
    Run a frequentist coverage test on the trained network.

    .. note::
        This function is not yet implemented.  To skip the coverage test,
        set ``coverage_test = False`` in the ``[TRAINING]`` section of your
        config.
    """
    raise NotImplementedError(
        "run_coverage_test is not yet implemented. "
        "Set coverage_test = False in [TRAINING] to skip."
    )
