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
    # RUN.run_dir (checkpoint, csv logs, PCA projection, mock observation,
    # plots) — training can vary independently of the shared simulation store
    # (architecture, scale cuts, which observation to evaluate against, ...).
    train_id = populate_train_dir(config, sys.argv[1])
    print(f"Training run: train_{train_id}")

    # Regenerated fresh on every run (not reloaded from disk) so a changed
    # MOCK_OBS/FIDUCIAL VALUES takes effect without re-running Fisher (part
    # of cli/simulate.py) or re-simulating — see observation.generate_observation.
    _, _, Lfid = generate_observation(config)

    store = swyft.ZarrStore(config["SIMULATION"]["store_path"]).get_sample_store()

    # TRAINING.train_on_frac (default 1.0): train on only the first
    # frac*N_sims simulations of the store, mainly for quick testing — the
    # store on disk is never touched, only this in-memory Samples object is
    # truncated before preprocessing/training. Sliced with an explicit copy
    # (not a plain `store[:n_sims]` view) so the discarded portion's buffer
    # can actually be freed — a numpy slice is a VIEW that keeps the full
    # underlying array (and its memory) alive for the rest of the run
    # regardless of frac. Confirmed the hard way: without the copy, this
    # cost ~2x the intended memory and OOM-killed several frac<1.0 jobs
    # against a 300k-sim store even at frac=0.5 (150k effective sims — well
    # within budget on its own).
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
