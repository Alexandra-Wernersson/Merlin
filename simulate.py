import sys
import os
import time
import logging
from datetime import datetime
import configparser
import numpy as np

from joblib import Parallel, delayed

import swyft

from simulator_utils import Simulator, get_sigmas_bounds


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
            "python simulate.py example_config.ini"
        )

        sys.exit(1)

    config_path = sys.argv[1]

    log(f"Reading config file: {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path)


    # --------------------------------------------------------
    # 2. logging
    # --------------------------------------------------------

    store_path = config["SIMULATION"]["store_path"]
    run_id = config["SIMULATION"].get("run_id", "test")
    os.makedirs(store_path, exist_ok=True)

    log_path = os.path.join(

        store_path,

        f"log_{run_id}.log"
    )

    logging.basicConfig(
        filename=log_path,
        filemode="w",
        format="%(asctime)s | %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    logging.info("Simulation run started")


    # --------------------------------------------------------
    # 3. read cosmology inputs
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

    finv_file = config["FINV"]["finv_file"]

    N_pars = int(
        config["FINV"]["N_pars"]
    )

    ell_theory = np.load(
        config["AUX FILES"]["ell_file"]
    )

    zmean = np.load(
        config["AUX FILES"]["zmean_file"]
    )


    # --------------------------------------------------------
    # 4. compute prior bounds
    # --------------------------------------------------------

    log("Computing prior bounds")

    _, lower_bounds, upper_bounds = get_sigmas_bounds(
        fiducial,
        finv_file,
        N_pars
    )


    # --------------------------------------------------------
    # 5. build simulator
    # --------------------------------------------------------

    log("Initializing simulator")

    sim = Simulator(

        fiducial=fiducial,
        covmat=covmat,
        n_bins=n_bins,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        zmean=zmean,
        ell_theory=ell_theory,
    )

    shapes, dtypes = sim.get_shapes_and_dtypes()

    # --------------------------------------------------------
    # 6. simulationsettings
    # --------------------------------------------------------

    N_sims = int(config["SIMULATION"]["N_sims"])

    batch_size = int(config["SIMULATION"]["batch_size"])

    chunk_size = int(config["SIMULATION"]["chunk_size"])

    n_workers = int(config["SIMULATION"]["n_workers"])


    # --------------------------------------------------------
    # 7. initialize Zarr store
    # --------------------------------------------------------

    log("Initializing Zarr store")

    store = swyft.ZarrStore(store_path)

    store.init(

        N_sims,

        batch_size,

        shapes,

        dtypes
    )


    # --------------------------------------------------------
    # 8. avoid thread oversubscription
    # --------------------------------------------------------

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    os.environ["OMP_NUM_THREADS"] = "1"

    os.environ["MKL_NUM_THREADS"] = "1"

    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    os.environ["NUMEXPR_MAX_THREADS"] = "1"


    # --------------------------------------------------------
    # 9. simulation loop
    # --------------------------------------------------------

    log("Starting simulations")

    start_time = time.time()


    def simulate_chunk(n):

        store.simulate(
            sim,
            max_sims=n,
            batch_size=n
        )


    while store.sims_required > 0:

        remaining = store.sims_required

        jobs = min(

            n_workers,

            remaining // chunk_size + 1
        )

        Parallel(

            n_jobs=jobs

        )(

            delayed(simulate_chunk)(chunk_size)

            for _ in range(jobs)
        )

        log(

            f"remaining simulations: {store.sims_required}"
        )


    # --------------------------------------------------------
    # 10. done
    # --------------------------------------------------------

    end_time = time.time()

    log(f"Finished {len(store)} simulations")

    log(

        f"Total runtime: {(end_time-start_time)/60:.1f} minutes"
    )

    logging.info(

        f"Finished {len(store)} simulations"
    )
