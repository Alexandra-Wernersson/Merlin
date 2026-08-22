import time

import numpy as np

from .io import format_duration, load_array
from .priors import resolve_priors
from .simulator import Simulator
from .tracers import load_dndz


def run_fisher(config, eps=1e-2):
    """
    Compute the Fisher matrix via finite differences over every VARIED
    parameter in config["PRIORS"], add each "normal"-kind parameter's own
    Gaussian-prior curvature, and save Finv to config["PRIORS"]["finv_file"]
    as an .npz containing Finv, varied_indices, and varied_names (the latter
    two let build_simulator/load_fisher_sigmas detect a stale Finv if
    PRIORS's fixed/varied set changed since).

    Finv is the inverse of the POSTERIOR (not just likelihood) Fisher matrix:
    likelihood information (J^T Cinv J) plus each "normal"-kind parameter's
    prior precision (1/sigma^2) on the diagonal — Laplace-approximating
    likelihood x prior. Without the prior term, marginalizing over other
    Gaussian-prior parameters overstates how freely they compensate a shift
    in parameter a, inflating a's marginalized sigma.

    Parameters
    ----------
    config : dict
        Loaded YAML config.
    eps : float
        Finite-difference step size as a fraction of the fiducial value.
        For parameters with fiducial=0, a fixed step of 5e-4 is used.

    Returns
    -------
    Finv   : np.ndarray, shape (len(varied_indices), len(varied_indices))
    sigmas : np.ndarray, shape (len(varied_indices),) — sigmas[k] corresponds
             to varied_indices[k], NOT the full PARAMS vector.
    """
    fiducial, specs, varied_names, varied_indices = resolve_priors(config)

    sim = Simulator(
        fiducial=fiducial,
        covmat=np.load(config["PRIORS"]["covmat_Fisher"])["Gauss"],
        n_bins=config["CLOELIB_SETTINGS"]["Nbin_z"],
        specs=specs,
        ell_theory=load_array(config["CLOELIB_SETTINGS"]["ell"]),
        dndz=load_dndz(config["PRIORS"]["nz_Fisher"]),
    )

    fiducial_arr = np.array(fiducial)
    inv_cov = np.linalg.inv(sim.Lfid @ sim.Lfid.T)

    print(f"Computing {len(varied_indices)} derivatives "
          f"({len(fiducial) - len(varied_indices)} of {len(fiducial)} parameters fixed)...")
    t0 = time.time()
    derivatives = [_finite_difference(sim, fiducial_arr, i, eps) for i in varied_indices]
    elapsed = time.time() - t0
    print(f"Fisher derivatives computed in {format_duration(elapsed)} "
          f"({2 * len(varied_indices)} model evaluations, single process)")

    n = len(varied_indices)
    F = np.zeros((n, n))
    for a in range(n):
        for b in range(a, n):
            F[a, b] = derivatives[a] @ inv_cov @ derivatives[b]
            F[b, a] = F[a, b]

    # F is LIKELIHOOD-only so far. Add each "normal"-kind parameter's prior
    # curvature (1/sigma^2) to the diagonal before inverting; "uniform"-kind
    # parameters have no curvature to add (flat prior).
    for k, i in enumerate(varied_indices):
        if specs[i].kind == "normal":
            F[k, k] += 1.0 / specs[i].sigma ** 2

    Finv   = np.linalg.inv(F)
    sigmas = np.sqrt(np.diag(Finv))

    finv_path = config["PRIORS"]["finv_file"]
    np.savez(finv_path, Finv=Finv,
              varied_indices=np.array(varied_indices),
              varied_names=np.array(varied_names))
    print(f"Saved Finv to {finv_path}")

    return Finv, sigmas


def _finite_difference(sim, fiducial, i, eps):
    theta_p = fiducial.copy()
    theta_m = fiducial.copy()
    step = eps * fiducial[i] if fiducial[i] != 0 else 5e-4
    theta_p[i] += step
    theta_m[i] -= step

    C1 = sim.sample(conditions={"z": theta_p}, targets=["C_ells"])["C_ells"]
    C2 = sim.sample(conditions={"z": theta_m}, targets=["C_ells"])["C_ells"]
    return (C1 - C2) / (2 * step)
