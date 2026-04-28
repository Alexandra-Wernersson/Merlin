"""
CLI entry point: generate a fiducial observation.

Usage::

    merlin-generate-obs <config.ini>
"""

import os
import sys
import logging
import configparser
from datetime import datetime

import numpy as np
from scipy.linalg import solve_triangular

from merlin.simulator import Simulator
from merlin.fisher import get_sigmas_bounds, fisher_analysis
from merlin.tracers import load_dndz


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] | {msg}", flush=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: merlin-generate-obs <config.ini>")
        sys.exit(1)

    config_path = sys.argv[1]
    log(f"Reading config: {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path)

    # ── 1. load cosmology inputs ──────────────────────────────────────────────
    log("Loading cosmology inputs")

    fiducial = [float(v) for v in config["FIDUCIAL VALUES"].values()]

    covmat = np.load(config["AUX FILES"]["covmat"])["Gauss"]
    n_bins = int(config["FINV"]["Nbin_z"])
    ell_theory = np.load(config["AUX FILES"]["ell_file"])
    zmean = np.load(config["AUX FILES"]["zmean_file"])
    dndz = load_dndz(config["AUX FILES"]["nz_example"])

    # ── 2. optionally run Fisher analysis ─────────────────────────────────────
    if config["FINV"].getboolean("run_fisher", fallback=False):
        log("Running Fisher analysis")
        Finv, sigmas = fisher_analysis(config)
        np.save(config["FINV"]["finv_file"], Finv)
        log(f"Saved Fisher matrix to {config['FINV']['finv_file']}")

    # ── 3. prior bounds ───────────────────────────────────────────────────────
    log("Computing prior bounds")
    _, lower_bounds, upper_bounds = get_sigmas_bounds(
        fiducial,
        config["FINV"]["finv_file"],
        int(config["FINV"]["N_pars"]),
    )

    # ── 4. build simulator ────────────────────────────────────────────────────
    log("Initialising simulator")
    simulator = Simulator(
        fiducial=fiducial,
        covmat=covmat,
        n_bins=n_bins,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        zmean=zmean,
        ell_theory=ell_theory,
        dndz=dndz,
    )

    # ── 5. generate observation ───────────────────────────────────────────────
    log("Generating observation")
    obs = simulator.generate_observation()

    # ── 6. Cholesky-whiten the observation ────────────────────────────────────
    log("Applying Cholesky rotation")
    Lfid = simulator.Lfid

    oCells_chol = solve_triangular(
        Lfid, obs["C_ells"].T, lower=True, check_finite=False
    ).T
    onoise_chol = solve_triangular(
        Lfid, obs["noise"].T, lower=True, check_finite=False
    ).T

    # Noiseless version is used for inference
    obs_chol_noiseless = {
        "C_ells": oCells_chol,
        "noise": np.zeros_like(onoise_chol),
    }

    # ── 7. save outputs ───────────────────────────────────────────────────────
    log("Saving outputs")
    obs_path = config["OBSERVATION"]["OBS"]
    obs_chol_noiseless_path = config["OBSERVATION"]["OBS_CHOLESKY_NOISELESS"]
    Lfid_path = config["OBSERVATION"]["LFID"]

    for p in [obs_path, obs_chol_noiseless_path, Lfid_path]:
        os.makedirs(os.path.dirname(p), exist_ok=True)

    np.save(obs_path, obs)
    np.save(obs_chol_noiseless_path, obs_chol_noiseless)
    np.save(Lfid_path, Lfid)

    log(f"Saved observation  → {obs_path}")
    log(f"Saved obs (chol)   → {obs_chol_noiseless_path}")
    log(f"Saved Lfid         → {Lfid_path}")


if __name__ == "__main__":
    main()
