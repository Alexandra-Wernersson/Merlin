import numpy as np

from .simulator import Simulator
from .tracers import load_dndz


def run_fisher(config, eps=1e-2):
    """
    Compute the Fisher matrix via finite differences and save Finv.

    Parameters
    ----------
    config : dict
        Loaded YAML config.
    eps : float
        Finite-difference step size as a fraction of the fiducial value.
        For parameters with fiducial=0, a fixed step of 5e-4 is used.

    Returns
    -------
    Finv : np.ndarray, shape (N_pars, N_pars)
    sigmas : np.ndarray, shape (N_pars,)
    """
    fiducial = list(config["FIDUCIAL VALUES"].values())
    N_pars   = config["FINV"]["N_pars"]

    # Placeholder bounds — prior sampler is never called during Fisher
    lower_bounds = [0.0] * N_pars
    upper_bounds = [1.0] * N_pars

    sim = Simulator(
        fiducial=fiducial,
        covmat=np.load(config["FINV"]["covmat"])["Gauss"],
        n_bins=config["FINV"]["Nbin_z"],
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        zmean=np.load(config["AUX FILES"]["zmean_file"]),
        ell_theory=np.load(config["AUX FILES"]["ell_file"]),
        dndz=load_dndz(config["FINV"]["nz_example"]),
    )

    fiducial = np.array(fiducial)
    N_pars   = config["FINV"]["N_pars"]
    inv_cov  = np.linalg.inv(sim.Lfid @ sim.Lfid.T)

    print(f"Computing {N_pars} derivatives...")
    derivatives = [_finite_difference(sim, fiducial, i, eps) for i in range(N_pars)]

    F = np.zeros((N_pars, N_pars))
    for i in range(N_pars):
        for j in range(i, N_pars):
            F[i, j] = derivatives[i] @ inv_cov @ derivatives[j]
            F[j, i] = F[i, j]

    Finv   = np.linalg.inv(F)
    sigmas = np.sqrt(np.diag(Finv))

    finv_path = config["FINV"]["finv_file"]
    np.save(finv_path, Finv)
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
