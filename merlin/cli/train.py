import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: merlin-train <config.yaml>")
        sys.exit(1)
    import swyft
    from ..config import load_config, populate_train_dir
    from ..observation import generate_observation
    from ..preprocessing import preprocess
    from ..train import train

    config = load_config(sys.argv[1])

    # Every merlin-train run gets its own numbered train_<N>/ subfolder under
    # RUN.run_dir (checkpoint, logs, PCA projection, mock observation, plots)
    # — training varies independently of the shared simulation store.
    train_id = populate_train_dir(config, sys.argv[1])
    print(f"Training run: train_{train_id}")

    # Regenerated fresh (not reloaded) so a changed MOCK_OBS/FIDUCIAL takes
    # effect without re-running Fisher or re-simulating.
    _, _, Lfid = generate_observation(config)

    store = swyft.ZarrStore(config["SIMULATION"]["store_path"]).get_sample_store()

    # TRAINING.train_on_frac (default 1.0): train on only the first
    # frac*N_sims simulations, mainly for quick testing. Sliced with an
    # explicit copy rather than a view — a plain slice keeps the full
    # underlying array alive, which doubled memory use and OOM-killed
    # frac<1.0 jobs against a 300k-sim store.
    frac = config["TRAINING"].get("train_on_frac", 1.0)
    if not (0 < frac <= 1.0):
        raise ValueError(f"TRAINING.train_on_frac must be in (0, 1], got {frac}")
    n_sims = int(len(store) * frac)
    if frac < 1.0:
        store = swyft.Samples({k: v[:n_sims].copy() for k, v in store.items()})
    print(f"Training on {n_sims} simulations")

    store_samples, V_proj = preprocess(store, Lfid, config)
    train(store_samples, V_proj, config,
          num_workers=config["TRAINING"].get("num_workers", 12))
