import time

import numpy as np

from .io import format_duration, load_array
from .params import PARAMS, N_COSMO, resolve_derived_names
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

    If CLOELIB_SETTINGS.restrict_prior_for_derived is also set (with
    add_derived non-empty), additionally computes and saves delta-method
    Fisher sigmas for each requested derived quantity (see
    _derived_fisher_sigmas) into the same finv.npz, for
    simulator.build_simulator's derived-quantity rejection-sampling box.

    Returns
    -------
    Finv   : np.ndarray, shape (len(varied_indices), len(varied_indices))
    sigmas : np.ndarray, shape (len(varied_indices),) — sigmas[k] corresponds
             to varied_indices[k], NOT the full PARAMS vector.
    """
    fiducial, specs, varied_names, varied_indices = resolve_priors(config)

    # CLOELIB_SETTINGS.restrict_prior_for_derived: only meaningful with
    # Fisher priors and >=1 derived quantity requested. This Simulator is
    # never given a derived_box, so its own z-node stays the plain
    # sample_z -- Fisher must be computed from the unrestricted prior
    # (computing it from an already-restricted one would be circular).
    derived_names = resolve_derived_names(config["CLOELIB_SETTINGS"].get("add_derived"))
    restrict_derived = (
        config["PRIORS"].get("use_Fisher_priors", False)
        and config["CLOELIB_SETTINGS"].get("restrict_prior_for_derived", False)
        and bool(derived_names)
    )

    sim = Simulator(
        fiducial=fiducial,
        covmat=np.load(config["PRIORS"]["covmat_Fisher"])["Gauss"],
        n_bins=config["CLOELIB_SETTINGS"]["Nbin_z"],
        specs=specs,
        ell_theory=load_array(config["CLOELIB_SETTINGS"]["ell"]),
        dndz=load_dndz(config["PRIORS"]["nz_Fisher"]),
        derived_names=derived_names if restrict_derived else (),
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

    save_kwargs = dict(Finv=Finv, varied_indices=np.array(varied_indices),
                        varied_names=np.array(varied_names))

    if restrict_derived:
        print("Computing derived-parameter Fisher sigmas...")
        derived_sigma, derived_fid = _derived_fisher_sigmas(
            sim, fiducial_arr, Finv, varied_names, derived_names, eps
        )
        save_kwargs.update(
            derived_names=np.array(derived_names),
            derived_fisher_sigma=np.array([derived_sigma[name] for name in derived_names]),
            derived_fiducial=np.array([derived_fid[name] for name in derived_names]),
        )

    finv_path = config["PRIORS"]["finv_file"]
    np.savez(finv_path, **save_kwargs)
    print(f"Saved Finv to {finv_path}")

    return Finv, sigmas


def _derived_fisher_sigmas(sim, fiducial, Finv, varied_names, derived_names, eps):
    """
    Delta-method Fisher sigma for each of derived_names: sigma_d = sqrt(J^T
    @ Finv_cosmo @ J), J the central-difference Jacobian of the derived
    quantity w.r.t. the 5 COSMO params at the fiducial point (same eps
    convention as _finite_difference), Finv_cosmo the COSMO sub-block of
    Finv (already the correctly marginalized covariance for that subset,
    per the Fisher-submatrix property load_fisher_sigmas also relies on).

    Calls sim._background_perturbations/_compute_derived as plain methods
    (NOT through sim.sample()/the swyft graph, which would run the full
    Cls/tracers/AngularTwoPoint pipeline for every extra evaluation).

    Returns
    -------
    sigma    : dict {name: float}
    fiducial_vals : dict {name: float}
    """
    cosmo_idx = list(range(N_COSMO))
    sub_idx = [varied_names.index(PARAMS[i]) for i in cosmo_idx]
    Finv_cosmo = Finv[np.ix_(sub_idx, sub_idx)]

    background, perturbations, _ = sim._background_perturbations(fiducial)
    fiducial_vals = dict(zip(derived_names, sim._compute_derived(background, perturbations)))

    jac = {name: [] for name in derived_names}
    for i in cosmo_idx:
        step = eps * fiducial[i] if fiducial[i] != 0 else 5e-4
        theta_p, theta_m = fiducial.copy(), fiducial.copy()
        theta_p[i] += step
        theta_m[i] -= step
        bp, pp, _ = sim._background_perturbations(theta_p)
        bm, pm, _ = sim._background_perturbations(theta_m)
        vals_p = dict(zip(derived_names, sim._compute_derived(bp, pp)))
        vals_m = dict(zip(derived_names, sim._compute_derived(bm, pm)))
        for name in derived_names:
            jac[name].append((vals_p[name] - vals_m[name]) / (2 * step))

    sigma = {name: float(np.sqrt(np.array(j) @ Finv_cosmo @ np.array(j))) for name, j in jac.items()}
    return sigma, {n: float(v) for n, v in fiducial_vals.items()}


def _finite_difference(sim, fiducial, i, eps):
    theta_p = fiducial.copy()
    theta_m = fiducial.copy()
    step = eps * fiducial[i] if fiducial[i] != 0 else 5e-4
    theta_p[i] += step
    theta_m[i] -= step

    C1 = sim.sample(conditions={"z": theta_p}, targets=["C_ells"])["C_ells"]
    C2 = sim.sample(conditions={"z": theta_m}, targets=["C_ells"])["C_ells"]
    return (C1 - C2) / (2 * step)
