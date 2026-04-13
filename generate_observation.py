import sys
import os
import numpy as np
import configparser
from datetime import datetime
from scipy.linalg import solve_triangular

from simulator_utils import Simulator, get_sigmas_bounds, fisher_analysis


# ============================================================
# helper
# ============================================================

def log(msg):

    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] | {msg}",
        flush=True
    )


# ============================================================
# main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. config
    # --------------------------------------------------------

    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "python generate_observation.py example_config.ini"
        )
        sys.exit(1)

    config_path = sys.argv[1]
    log(f"Reading config file: {config_path}")
    config = configparser.ConfigParser()
    config.read(config_path)


    # --------------------------------------------------------
    # 2. paths
    # --------------------------------------------------------

    store_path = config["SIMULATION"]["store_path"]
    os.makedirs(store_path, exist_ok=True)


    # --------------------------------------------------------
    # 3. load inputs
    # --------------------------------------------------------

    log("Loading cosmology inputs")
    fiducial = [
        float(v)
        for v in config["FIDUCIAL VALUES"].values()
    ]

    covmat = np.load(
        config["FINV"]["covmat"]
    )["Gauss"]


    n_bins = int(
        config["FINV"]["Nbin_z"]
    )

    ell_theory = np.load(
        config["AUX FILES"]["ell_file"]
    )

    zmean = np.load(
        config["AUX FILES"]["zmean_file"]
    )

    if config["FINV"]["run_fisher"] == "True":

        log("Running Fisher analysis")

        Finv, sigmas = fisher_analysis(config)

        np.save(config["FINV"]["finv_file"], Finv)

        log("Saved Fisher matrix")
        log(config["FINV"]["finv_file"])
    # --------------------------------------------------------
    # 4. prior bounds
    # --------------------------------------------------------

    log("Computing prior bounds")

    _, lower_bounds, upper_bounds = get_sigmas_bounds(
        fiducial,
        config["FINV"]["finv_file"],
        int(config["FINV"]["N_pars"])
    )


    # --------------------------------------------------------
    # 5. build simulator
    # --------------------------------------------------------

    log("Initializing simulator")

    simulator = Simulator(
        fiducial=fiducial,
        covmat=covmat,
        n_bins=n_bins,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        zmean=zmean,
        ell_theory=ell_theory,
    )


    # --------------------------------------------------------
    # 6. generate observation
    # --------------------------------------------------------

    log("Generating observation")
    obs = simulator.generate_observation()


    # --------------------------------------------------------
    # 7. rotate observation using Cholesky
    # --------------------------------------------------------

    log("Applying Cholesky rotation")

    Lfid = simulator.Lfid

    oCells_chol = solve_triangular(
        Lfid,
        obs["C_ells"].T,
        lower=True,
        check_finite=False
    ).T


    onoise_chol = solve_triangular(
        Lfid,
        obs["noise"].T,
        lower=True,
        check_finite=False
    ).T


    obs_chol = dict(
        C_ells=oCells_chol,
        noise=onoise_chol
    )


    # noiseless version used for inference
    obs_chol_noiseless = dict(
        C_ells=oCells_chol,
        noise=0.0 * onoise_chol
    )


    # --------------------------------------------------------
    # 8. save outputs
    # --------------------------------------------------------

    log("Saving outputs")
    obs_path = config["OBSERVATION"]["OBS"]
    obs_chol_path = config["OBSERVATION"]["OBS_CHOLESKY"]
    obs_chol_noiseless_path = config["OBSERVATION"]["OBS_CHOLESKY_NOISELESS"]
    Lfid_path = config["OBSERVATION"]["LFID"]
    metadata_path = config["OBSERVATION"]["metadata_path"]

    for p in [
        obs_path,
        obs_chol_path,
        obs_chol_noiseless_path,
        Lfid_path,
        metadata_path

    ]:

        os.makedirs(
            os.path.dirname(p),
            exist_ok=True
        )

    # save original observation
    np.save(

        obs_path,
        obs
    )

    # save rotated observation
    np.save(
        obs_chol_path,
        obs_chol
    )

    # save noiseless rotated observation
    np.save(
        obs_chol_noiseless_path,
        obs_chol_noiseless
    )

    # save cholesky
    np.save(
        Lfid_path,
        Lfid
    )

    # save metadata
    np.savez(
        metadata_path,
        fiducial=fiducial,
        n_bins=n_bins
    )


    log("Saved:")
    log(obs_path)
    log(obs_chol_path)
    log(obs_chol_noiseless_path)
    log(Lfid_path)
    log(metadata_path)
