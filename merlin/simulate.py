import os
import shutil
import time
import logging
from datetime import datetime

import swyft
from joblib import Parallel, delayed
from tqdm import tqdm

from .io import format_duration
from .simulator import build_simulator


def simulate(config):
    """
    Fill a ZarrStore with simulations using parallel joblib workers.

    If config["PRIORS"]["use_Fisher_priors"] is true, runs fisher.run_fisher
    first (a no-op otherwise) — Fisher only ever feeds prior bounds for THIS
    step (via build_simulator), so it's run here rather than as a separate
    CLI step; the Zarr store is the only artifact this produces that anything
    downstream (observation.generate_observation, inference, coverage) reads
    back, and none of them re-trigger Fisher themselves.

    Config keys used:
        SIMULATION.store_path, N_sims, batch_size, chunk_size, n_workers,
        add_extra_sims

    SIMULATION.add_extra_sims (default false): if true, grow an EXISTING store
    at store_path to the new (larger) N_sims via ZarrStore.reset_length, then
    fill the newly added slots — instead of creating a separate store. Raises
    ValueError if no store exists yet at store_path, or if N_sims is not
    strictly larger than the store's current length. Leave false for the
    normal create-or-resume-up-to-N_sims behaviour.
    """
    if config["PRIORS"].get("use_Fisher_priors", False):
        from .fisher import run_fisher
        print("Running Fisher analysis first...")
        run_fisher(config)

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

    if config["SIMULATION"].get("add_extra_sims", False):
        current_len = len(store)
        if current_len == 0:
            raise ValueError(
                f"SIMULATION.add_extra_sims is true but no existing store was "
                f"found at {store_path!r} to grow."
            )
        if N_sims <= current_len:
            raise ValueError(
                f"SIMULATION.add_extra_sims is true but N_sims ({N_sims}) must "
                f"be larger than the existing store's current length ({current_len})."
            )
        _log(f"Growing store from {current_len} to {N_sims} simulations")
        store.reset_length(N_sims)

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
        # swyft.Simulator.sample() wraps its per-sample loop in tqdm with no
        # way to disable it — replace the module-level tqdm reference it uses
        # with a no-op passthrough. Applied inside the worker function itself
        # (not just once in the parent process) since joblib workers may be
        # separate processes that re-import swyft fresh.
        import swyft.lightning.simulator as _swyft_simulator_mod
        _swyft_simulator_mod.tqdm = lambda it, *a, **kw: it
        store.simulate(sim, max_sims=n, batch_size=n)

    _log("Starting simulations")
    start_len = len(store) - store.sims_required  # already-filled slots, excluded from the count below
    t0 = time.time()
    while store.sims_required > 0:
        remaining = store.sims_required
        jobs = min(n_workers, remaining // chunk_size + 1)
        Parallel(n_jobs=jobs)(delayed(_chunk)(chunk_size) for _ in range(jobs))
        done = N_sims - store.sims_required
        # A static tqdm-formatted bar string printed as a normal log line (not
        # tqdm's own live \r-redrawn display, which collapses into one unreadable
        # line when the log is viewed as a plain file rather than a live terminal).
        bar = tqdm.format_meter(n=done, total=N_sims, elapsed=time.time() - t0,
                                 unit="sim", prefix="Simulating")
        _log(bar)

    elapsed    = time.time() - t0
    n_generated = len(store) - start_len
    summary = (f"Generated {n_generated} simulations in {format_duration(elapsed)} "
               f"using {n_workers} workers")
    _log(summary)
    logging.info(summary)

    # store.sync/store.lock.file are swyft/zarr's own inter-process write
    # coordination artifacts (ProcessSynchronizer / fasteners.InterProcessLock),
    # not simulation data — safe to remove once no more workers are writing;
    # they're recreated automatically on demand if the store is read or
    # written to again later.
    shutil.rmtree(store_path + ".sync", ignore_errors=True)
    lock_file = store_path + ".lock.file"
    if os.path.exists(lock_file):
        os.remove(lock_file)

    return store
