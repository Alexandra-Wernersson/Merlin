import shutil
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# (section, key) pairs whose values are file/directory paths. Relative values
# are resolved against REPO_ROOT; absolute values are left untouched.
_PATH_KEYS = {
    "RUN": ["run_dir"],
    "PRIORS": ["finv_file", "covmat_Fisher", "nz_Fisher"],
    "MOCK_OBS": ["path"],
    "OBSERVATION": ["OBS", "OBS_CHOLESKY", "LFID"],
    # ell may hold an inline array instead of a file path (see io.load_array);
    # non-string values are left untouched by the resolution loop below.
    "CLOELIB_SETTINGS": ["covmat", "nz", "ell"],
    "PCA": ["SVD", "pca_file"],
    "STORES": ["checkpoint_path", "plots_dir", "csv_logs"],
    "SIMULATION": ["store_path"],
}


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)

    for section, keys in _PATH_KEYS.items():
        if section not in config:
            continue
        for key in keys:
            value = config[section].get(key)
            if isinstance(value, str) and not Path(value).is_absolute():
                config[section][key] = str(REPO_ROOT / value)

    if "RUN" in config:
        _populate_run_dir(config)

    return config


def _populate_run_dir(config):
    """
    Derive paths shared across every training run under RUN.run_dir: the
    Zarr store and Fisher matrix, both produced once by `merlin-simulate`
    and only ever read downstream. Creates run_dir/store on disk;
    run_dir/finv.npz is written later by fisher.run_fisher.

        <run_dir>/store            SIMULATION.store_path (swyft ZarrStore)
        <run_dir>/finv.npz         PRIORS.finv_file

    Per-run artifacts (checkpoint, csv logs, PCA, mock obs, plots) live
    under <run_dir>/train_<N>/ instead -- see populate_train_dir, which
    must be called explicitly before any training/observation/plotting.
    """
    run_dir = Path(config["RUN"]["run_dir"])
    store_dir = run_dir / "store"
    store_dir.mkdir(parents=True, exist_ok=True)

    config.setdefault("SIMULATION", {})["store_path"] = str(store_dir)
    config.setdefault("PRIORS", {})["finv_file"] = str(run_dir / "finv.npz")


def _train_ids(run_dir):
    """Sorted list of N for every existing run_dir/train_<N> subfolder."""
    if not run_dir.is_dir():
        return []
    ids = []
    for p in run_dir.iterdir():
        if p.is_dir() and p.name.startswith("train_") and p.name[len("train_"):].isdigit():
            ids.append(int(p.name[len("train_"):]))
    return sorted(ids)


def populate_train_dir(config, config_path, train_id=None):
    """
    Create (train_id=None) or select (train_id=<int>) a run_dir/train_<N>
    subfolder, and point STORES.checkpoint_path/csv_logs/plots_dir,
    OBSERVATION.OBS/OBS_CHOLESKY/LFID, and PCA.SVD inside it. Must be called
    after load_config and before any training/observation/plotting call
    that needs those paths -- unlike store_path/finv_file, load_config
    can't populate them since it doesn't know if the call is meant to
    create a new training run, target an existing one, or neither
    (e.g. merlin-simulate).

    train_id=None: creates the next free train_<N> and copies config_path
    there as train_<N>/config.yaml. Returns the new train_id.
    train_id=<int>: selects an existing train_<train_id>, leaving its saved
    config.yaml untouched. Raises FileNotFoundError if it doesn't exist.
    """
    run_dir = Path(config["RUN"]["run_dir"])

    if train_id is None:
        train_id = max(_train_ids(run_dir), default=0) + 1
        train_dir = run_dir / f"train_{train_id}"
        train_dir.mkdir(parents=True)
        shutil.copy(config_path, train_dir / "config.yaml")
    else:
        train_dir = run_dir / f"train_{train_id}"
        if not train_dir.is_dir():
            raise FileNotFoundError(
                f"{train_dir} does not exist — run `merlin-train <config>` first, "
                f"or check --train-id (existing: {_train_ids(run_dir)})"
            )

    # checkpoint_path/csv_logs point directly at train_dir since each only
    # ever holds one relevant file (best.ckpt; metrics.csv).
    plots_dir    = train_dir / "plots"
    aux_dir      = train_dir / "aux_files"
    mock_obs_dir = train_dir / "mock_obs"

    for d in [plots_dir, aux_dir, mock_obs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    config.setdefault("STORES", {}).update({
        "checkpoint_path": str(train_dir),
        "csv_logs": str(train_dir),
        "plots_dir": str(plots_dir),
    })
    config.setdefault("OBSERVATION", {}).update({
        "OBS": str(mock_obs_dir / "obs.npy"),
        "OBS_CHOLESKY": str(mock_obs_dir / "obs_cholesky.npy"),
        "LFID": str(aux_dir / "Lfid.npy"),
    })
    config.setdefault("PCA", {})["SVD"] = str(aux_dir / "SVD.npy")

    return train_id
