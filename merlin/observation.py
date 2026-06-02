import numpy as np
from scipy.linalg import solve_triangular
import swyft

from .simulator import build_simulator


def generate_observation(config):
    """
    Generate a noiseless observation at the fiducial cosmology, apply
    Cholesky whitening, and save all outputs defined in config.

    Returns
    -------
    obs : dict
        Raw observation with keys 'z', 'C_ells', 'noise'.
    obs_chol_noiseless : dict
        Cholesky-whitened noiseless observation (what inference uses).
    Lfid : np.ndarray
        Cholesky factor of the fiducial covariance.
    """
    if config["FINV"].get("run_fisher", False):
        from .fisher import run_fisher
        print("Running Fisher analysis first...")
        run_fisher(config)

    sim = build_simulator(config)
    obs = sim.generate_observation()
    Lfid = sim.Lfid

    oCells_chol = solve_triangular(
        Lfid, obs["C_ells"].T, lower=True, check_finite=False
    ).T

    onoise_chol = solve_triangular(
        Lfid, obs["noise"].T, lower=True, check_finite=False
    ).T

    obs_chol_noiseless = dict(C_ells=oCells_chol, noise=0.0 * onoise_chol)

    # save
    obs_path             = config["OBSERVATION"]["OBS"]
    obs_chol_noiseless_path = config["OBSERVATION"]["OBS_CHOLESKY_NOISELESS"]
    lfid_path            = config["OBSERVATION"]["LFID"]

    for p in [obs_path, obs_chol_noiseless_path, lfid_path]:
        import os
        os.makedirs(os.path.dirname(p), exist_ok=True)

    np.save(obs_path, obs)
    np.save(obs_chol_noiseless_path, obs_chol_noiseless)
    np.save(lfid_path, Lfid)

    print(f"Saved observation to      {obs_path}")
    print(f"Saved noiseless chol obs  {obs_chol_noiseless_path}")
    print(f"Saved Lfid to             {lfid_path}")

    return obs, obs_chol_noiseless, Lfid
