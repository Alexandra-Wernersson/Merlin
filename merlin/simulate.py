import os
import shutil
import time
import logging
from datetime import datetime
from pathlib import Path

import swyft
from joblib import Parallel, delayed
from tqdm import tqdm

from .io import format_duration
from .params import resolve_derived_names
from .priors import load_derived_fisher_sigmas
from .simulator import build_simulator


_GROWTH_CONSISTENCY_SECTIONS = ("FIDUCIAL", "PRIORS", "CLOELIB_SETTINGS")


def _check_growth_consistency(config, run_dir):
    """
    Verify this config's FIDUCIAL/PRIORS/CLOELIB_SETTINGS (everything
    build_simulator reads) match the config that produced the store's
    existing rows, saved at run_dir/config.yaml -- growing with different
    settings would silently mix incompatible simulations. NETWORK/TRAINING/
    PCA/PLOTTING vary independently per train_<N> and are skipped.
    """
    from .config import load_config

    old_config_path = run_dir / "config.yaml"
    if not old_config_path.exists():
        raise ValueError(
            f"SIMULATION.add_extra_sims is true but no saved config.yaml was "
            f"found at {old_config_path} to check consistency against."
        )
    old_config = load_config(str(old_config_path))

    for section in _GROWTH_CONSISTENCY_SECTIONS:
        old_val = old_config.get(section, {}) or {}
        new_val = config.get(section, {}) or {}
        if old_val == new_val:
            continue
        for key in set(old_val) | set(new_val):
            if old_val.get(key) != new_val.get(key):
                raise ValueError(
                    f"SIMULATION.add_extra_sims is true but {section}.{key} "
                    f"differs from the config that produced this store's "
                    f"existing simulations: was {old_val.get(key)!r}, now "
                    f"{new_val.get(key)!r}. Growing a store with different "
                    f"simulation settings would silently mix incompatible "
                    f"rows -- simulate a fresh store instead."
                )


def simulate(config):
    """
    Fill a ZarrStore with simulations using parallel joblib workers.

    Runs fisher.run_fisher first if PRIORS.use_Fisher_priors is set (a no-op
    otherwise), since Fisher only feeds prior bounds for this step.

    Config keys: SIMULATION.store_path, N_sims, batch_size, chunk_size,
    n_workers, add_extra_sims.

    If SIMULATION.add_extra_sims is true, grows an existing store at
    store_path to the new N_sims via ZarrStore.reset_length instead of
    creating a new one. Raises ValueError if no store exists, N_sims isn't
    strictly larger than the current length, or FIDUCIAL/PRIORS/
    CLOELIB_SETTINGS differ from what produced the existing rows (see
    _check_growth_consistency). Default false creates or resumes a store
    up to N_sims.

    If CLOELIB_SETTINGS.restrict_prior_for_derived is also set (with
    add_derived non-empty), each row is drawn via rejection sampling on a
    second, tighter Fisher box over the derived quantities themselves (see
    simulator.Simulator._sample_z_derived_rejection) -- as many candidate
    cosmologies as needed are generated to fill the store with N_sims
    *accepted* rows; the total generated/accepted counts are logged at the end.
    """
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

    def _log(msg):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] | {msg}", flush=True)

    if config["PRIORS"].get("use_Fisher_priors", False):
        from .fisher import run_fisher
        _log("Running Fisher analysis first...")
        run_fisher(config)

        derived_names = resolve_derived_names(config["CLOELIB_SETTINGS"].get("add_derived"))
        if config["CLOELIB_SETTINGS"].get("restrict_prior_for_derived", False) and derived_names:
            sigma_d, fid_d = load_derived_fisher_sigmas(config["PRIORS"]["finv_file"], derived_names)
            for name in derived_names:
                msg = f"Fisher sigma [{name}] = {sigma_d[name]:.6g}  (fiducial {fid_d[name]:.6g})"
                _log(msg)
                logging.info(msg)

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
        _check_growth_consistency(config, Path(config["RUN"]["run_dir"]))
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
        # Disable swyft's per-sample tqdm (no off-switch) by patching its module
        # reference; done per-worker since joblib processes re-import swyft.
        import swyft.lightning.simulator as _swyft_simulator_mod
        _swyft_simulator_mod.tqdm = lambda it, *a, **kw: it
        # Deltas, not raw counts -- defensive against a loky worker process
        # being reused across dispatches within the same Parallel() pool.
        before_gen, before_acc = sim._n_generated, sim._n_accepted
        store.simulate(sim, max_sims=n, batch_size=n)
        return sim._n_generated - before_gen, sim._n_accepted - before_acc

    _log("Starting simulations")
    start_len = len(store) - store.sims_required  # already-filled slots, excluded from the count below
    total_generated = 0
    total_accepted  = 0
    t0 = time.time()
    while store.sims_required > 0:
        remaining = store.sims_required
        jobs = min(n_workers, remaining // chunk_size + 1)
        results = Parallel(n_jobs=jobs)(delayed(_chunk)(chunk_size) for _ in range(jobs))
        for gen, acc in results:
            total_generated += gen
            total_accepted  += acc
        done = N_sims - store.sims_required
        # Static tqdm-formatted bar as a log line, not tqdm's live \r-redraw
        # (unreadable when the log is viewed as a plain file).
        bar = tqdm.format_meter(n=done, total=N_sims, elapsed=time.time() - t0,
                                 unit="sim", prefix="Simulating")
        _log(bar)

    if sim.derived_box:
        frac = total_accepted / total_generated if total_generated else float("nan")
        rejection_summary = (
            f"Derived-quantity rejection sampling: {total_generated} cosmologies "
            f"generated, kept {frac:.1%} of these"
        )
        _log(rejection_summary)
        logging.info(rejection_summary)

    elapsed    = time.time() - t0
    n_generated = len(store) - start_len
    summary = (f"Generated {n_generated} simulations in {format_duration(elapsed)} "
               f"using {n_workers} workers")
    _log(summary)
    logging.info(summary)

    # store.sync/store.lock.file are zarr's inter-process write-coordination
    # artifacts, not simulation data -- safe to remove; recreated on demand.
    shutil.rmtree(store_path + ".sync", ignore_errors=True)
    lock_file = store_path + ".lock.file"
    if os.path.exists(lock_file):
        os.remove(lock_file)

    return store
