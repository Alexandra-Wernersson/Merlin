"""
Fisher-matrix utilities for deriving prior bounds.
"""

import numpy as np


def get_sigmas_bounds(fiducial, finv_file, N_pars):
    """
    Read an inverse Fisher matrix and return prior bounds for *N_pars*
    cosmological / nuisance parameters.

    Parameters
    ----------
    fiducial:
        List of fiducial parameter values.
    finv_file:
        Path to a ``.npy`` file containing the inverse Fisher matrix.
    N_pars:
        Number of parameters to extract bounds for.

    Returns
    -------
    sigmas : ndarray, shape (N_pars,)
    lower_bounds : list of float
    upper_bounds : list of float
    """
    Finv = np.load(finv_file)
    sigmas = np.sqrt(np.diag(Finv))
    lower_bounds = [fiducial[i] - 5 * sigmas[i] for i in range(N_pars)]
    upper_bounds = [fiducial[i] + 5 * sigmas[i] for i in range(N_pars)]
    return sigmas, lower_bounds, upper_bounds


def fisher_analysis(config):
    """
    Compute the inverse Fisher matrix for the configured survey setup.

    .. note::
        This function is not yet implemented.  Set ``run_fisher = False``
        in the ``[FINV]`` section of your config and provide a pre-computed
        ``finv_file``.

    Returns
    -------
    Finv : ndarray
    sigmas : ndarray
    """
    raise NotImplementedError(
        "fisher_analysis is not yet implemented. "
        "Set run_fisher = False in [FINV] and provide a precomputed finv_file."
    )
