import numpy as np
from scipy.linalg import solve_triangular
import swyft

from .simulator import build_simulator


def generate_observation(config):
    """
    Obtain an observation, apply Cholesky whitening, and save all outputs
    defined in config.

    Independent of Fisher: if config["PRIORS"]["use_Fisher_priors"] is true,
    config["PRIORS"]["finv_file"] must already exist (see simulate.simulate,
    which runs fisher.run_fisher first if enabled, before this) — this
    function only reads it via build_simulator, it never (re)computes it.
    Safe to call repeatedly (e.g. once per merlin-train run, to pick up a
    changed MOCK_OBS/FIDUCIAL without re-running Fisher or
    re-simulating) — every call writes fresh to OBSERVATION.OBS/OBS_CHOLESKY/
    LFID, which are train_<N>-scoped (config.populate_train_dir must be
    called first to point them at a specific training run's subfolder).

    config["MOCK_OBS"] controls where the observation comes from:
      - generate_from_fiducial: true (default if MOCK_OBS is absent) —
        simulate C_ells and noise SEPARATELY from the fiducial cosmology, as
        before ("path" is ignored, can be null). obs["z"] holds the true
        fiducial parameter vector. obs["noise"] is a real sampled noise
        realization (sim.get_sample_noise()) — always saved as part of the raw
        obs, but only folded into obs_chol if add_noise_fid_obs is true
        (default false, matching prior behavior — a noiseless observation for
        inference).
      - generate_from_fiducial: false — load an ALREADY-noisy data vector
        (C_ells + noise combined — e.g. real data, where no clean/noise split
        exists) from the plain .npy array at "path". obs has no "z" (no true
        parameter vector for real data) and "noise" is zero (the loaded array
        is stored as "C_ells" directly, since there's nothing left to split
        out). add_noise_fid_obs is ignored in this branch.

    Returns
    -------
    obs : dict
        Raw observation with keys 'C_ells', 'noise' (plus 'z' only when
        generate_from_fiducial is true). 'noise' here is always the true,
        un-added noise realization (or zero, in the loaded-file branch) —
        unaffected by add_noise_fid_obs.
    obs_chol : dict
        Cholesky-whitened observation used for inference (keys 'C_ells',
        'noise' — 'noise' is always zero here, a placeholder to match the
        dict shape elsewhere; 'C_ells' includes noise whenever there's
        nothing left to split out (generate_from_fiducial: false) or the
        caller opted in via add_noise_fid_obs (generate_from_fiducial: true),
        so it is not always literally noiseless — hence the plain
        "cholesky" name rather than "cholesky_noiseless").
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
