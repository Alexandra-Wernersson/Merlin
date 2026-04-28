"""
CLI entry point: run forward simulations and fill a Zarr store.

Usage::

    merlin-simulate <config.ini>
"""

import os
import sys
import time
import logging
import configparser
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed
import swyft

from merlin.simulator import Simulator
from merlin.fisher import get_sigmas_bounds
from merlin.tracers import load_dndz


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] | {msg}", flush=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: merlin-simulate <config.ini>")
        sys.exit(1)

    config_path = sys.argv[1]
    log(f"Reading config: {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path)

    # ── 1. set up logging ─────────────────────────────────────────────────────
    store_path = config["SIMULATION"]["store_path"]
    run_id = config["SIMULATION"].get("run_id", "run")
    os.makedirs(store_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(store_path, f"log_{run_id}.log"),
        filemode="w",
        format="%(asctime)s | %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logging.info("Simulation run started")

    # ── 2. load cosmology inputs ──────────────────────────────────────────────
    log("Loading cosmology inputs")

    fiducial = [float(v) for v in config["FIDUCIAL VALUES"].values()]
    covmat = np.load(config["FINV"]["covmat"])["Gauss"]
    n_bins = int(config["FINV"]["Nbin_z"])
    finv_file = config["FINV"]["finv_file"]
    N_pars = int(config["FINV"]["N_pars"])
    ell_theory = np.load(config["AUX FILES"]["ell_file"])
    zmean = np.load(config["AUX FILES"]["zmean_file"])
    dndz = load_dndz(config["AUX FILES"]["nz_example"])

    # ── 3. prior bounds ───────────────────────────────────────────────────────
    log("Computing prior bounds")
    _, lower_bounds, upper_bounds = get_sigmas_bounds(fiducial, finv_file, N_pars)

    # ── 4. build simulator ────────────────────────────────────────────────────
    log("Initialising simulator")
    sim = Simulator(
        fiducial=fiducial,
        covmat=covmat,
        n_bins=n_bins,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        zmean=zmean,
        ell_theory=ell_theory,
        dndz=dndz,
    )
    shapes, dtypes = sim.get_shapes_and_dtypes()

    # ── 5. simulation settings ────────────────────────────────────────────────
    N_sims = int(config["SIMULATION"]["N_sims"])
    batch_size = int(config["SIMULATION"]["batch_size"])
    chunk_size = int(config["SIMULATION"]["chunk_size"])
    n_workers = int(config["SIMULATION"]["n_workers"])

    # ── 6. initialise Zarr store ──────────────────────────────────────────────
    log("Initialising Zarr store")
    store = swyft.ZarrStore(store_path)
    store.init(N_sims, batch_size, shapes, dtypes)

    # ── 7. avoid thread oversubscription ─────────────────────────────────────
    for var in (
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
    ):
        os.environ[var] = "-1" if var == "CUDA_VISIBLE_DEVICES" else "1"

    # ── 8. simulation loop ────────────────────────────────────────────────────
    log("Starting simulations")
    start_time = time.time()

    def simulate_chunk(n):
        store.simulate(sim, max_sims=n, batch_size=n)

    while store.sims_required > 0:
        remaining = store.sims_required
        jobs = min(n_workers, remaining // chunk_size + 1)
        Parallel(n_jobs=jobs)(
            delayed(simulate_chunk)(chunk_size) for _ in range(jobs)
        )
        log(f"Remaining simulations: {store.sims_required}")

    elapsed = (time.time() - start_time) / 60
    log(f"Finished {len(store)} simulations in {elapsed:.1f} minutes")
    logging.info("Simulation run complete")


if __name__ == "__main__":
    main()
