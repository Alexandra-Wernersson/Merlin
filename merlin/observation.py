import numpy as np
from scipy.linalg import solve_triangular
import swyft

from .simulator import build_simulator


def generate_observation(config):
    """
    Obtain an observation, apply Cholesky whitening, and save all outputs
    defined in config.

    If config["PRIORS"]["use_Fisher_priors"] is true, finv_file must already
    exist (built by fisher.run_fisher); this function only reads it via
    build_simulator, never recomputes it. Safe to call repeatedly (e.g. once
    per merlin-train run) — always writes fresh to the train_<N>-scoped
    OBSERVATION.OBS/OBS_CHOLESKY/LFID paths.

    config["MOCK_OBS"] controls the source:
      - generate_from_fiducial: true (default) — simulate C_ells and noise
        separately from the fiducial cosmology ("path" ignored). obs["z"]
        holds the fiducial parameter vector; obs["noise"] is a real sampled
        realization, folded into obs_chol only if add_noise_fid_obs is true
        (default false, i.e. noiseless observation for inference).
      - generate_from_fiducial: false — load an already-noisy data vector
        from the .npy array at "path" (e.g. real data with no clean/noise
        split). obs has no "z"; "noise" is zero and the loaded array is
        stored as "C_ells" directly. add_noise_fid_obs is ignored here.

    Returns
    -------
    obs : dict
        Raw observation with keys 'C_ells', 'noise' (plus 'z' only when
        generate_from_fiducial is true).
    obs_chol : dict
        Cholesky-whitened observation for inference (keys 'C_ells', 'noise';
        'noise' is always zero, a placeholder). 'C_ells' includes noise
        whenever there's nothing left to split out or add_noise_fid_obs
        opted in, so it isn't always literally noiseless.
    Lfid : np.ndarray
        Cholesky factor of the fiducial covariance.
    """
    sim = build_simulator(config)
    Lfid = sim.Lfid

    mock_cfg = config.get("MOCK_OBS", {})
    if mock_cfg.get("generate_from_fiducial", True):
        print("Generating fiducial observation...")
        obs = sim.generate_observation()
        if mock_cfg.get("add_noise_fid_obs", False):
            data_for_chol = obs["C_ells"] + obs["noise"]
        else:
            data_for_chol = obs["C_ells"]
    else:
        mock_path = mock_cfg.get("path")
        if not mock_path:
            raise ValueError(
                "MOCK_OBS.generate_from_fiducial is false but MOCK_OBS.path is not set"
            )
        data = np.load(mock_path)
        if data.shape[-1] != sim.n_data:
            raise ValueError(
                f"Mock observation at {mock_path!r} has {data.shape[-1]} data "
                f"points, expected {sim.n_data} (WL + GGL + GCph combined)"
            )
        obs = dict(C_ells=data, noise=np.zeros_like(data))
        data_for_chol = obs["C_ells"]
        print(f"Loaded mock observation from {mock_path}")

    Cells_chol = solve_triangular(
        Lfid, data_for_chol.T, lower=True, check_finite=False
    ).T

    obs_chol = dict(C_ells=Cells_chol, noise=np.zeros_like(Cells_chol))

    # save
    obs_path      = config["OBSERVATION"]["OBS"]
    obs_chol_path = config["OBSERVATION"]["OBS_CHOLESKY"]
    lfid_path     = config["OBSERVATION"]["LFID"]

    for p in [obs_path, obs_chol_path, lfid_path]:
        import os
        os.makedirs(os.path.dirname(p), exist_ok=True)

    np.save(obs_path, obs)
    np.save(obs_chol_path, obs_chol)
    np.save(lfid_path, Lfid)

    print(f"Saved observation to      {obs_path}")
    print(f"Saved chol obs            {obs_chol_path}")
    print(f"Saved Lfid to             {lfid_path}")

    return obs, obs_chol, Lfid
