import os
import time
import logging
from datetime import datetime

import swyft
from joblib import Parallel, delayed

from .simulator import build_simulator


def simulate(config):
    """
    Fill a ZarrStore with simulations using parallel joblib workers.

    Config keys used:
        SIMULATION.store_path, N_sims, batch_size, chunk_size, n_workers
    """
    def _log(msg):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] | {msg}", flush=True)

    store_path = config["SIMULATION"]["store_path"]
    run_id     = config["SIMULATION"].get("run_id", "run")
    N_sims     = config["SIMULATION"]["N_sims"]
    batch_size = config["SIMULATION"]["batch_size"]
    chunk_size = config["SIMULATION"]["chunk_size"]
    n_workers  = config["SIMULATION"]["n_workers"]

    os.makedirs(store_path, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(store_path, f"log_{run_id}.log"),
        filemode="w",
        format="%(asctime)s | %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    _log("Building simulator")
    sim = build_simulator(config)
    shapes, dtypes = sim.get_shapes_and_dtypes()

    store = swyft.ZarrStore(store_path)
    try:
        remaining = store.sims_required
        if remaining == 0:
            _log("Store already complete")
            return store
        _log(f"Resuming store ({len(store)} done, {remaining} remaining)")
    except KeyError:
        _log("Initialising new Zarr store")
        store.init(N_sims, batch_size, shapes, dtypes)

    os.environ.update({
        "CUDA_VISIBLE_DEVICES": "-1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_MAX_THREADS": "1",
    })

    def _chunk(n):
        store.simulate(sim, max_sims=n, batch_size=n)

    _log("Starting simulations")
    t0 = time.time()
    while store.sims_required > 0:
        remaining = store.sims_required
        jobs = min(n_workers, remaining // chunk_size + 1)
        Parallel(n_jobs=jobs)(delayed(_chunk)(chunk_size) for _ in range(jobs))
        _log(f"Remaining: {store.sims_required}")

    _log(f"Finished {len(store)} simulations in {(time.time()-t0)/60:.1f} min")
    logging.info(f"Finished {len(store)} simulations")
    return store
