import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: merlin-simulate <config.yaml>")
        sys.exit(1)
    import shutil
    from pathlib import Path
    from ..config import load_config
    from ..simulate import simulate

    config = load_config(sys.argv[1])
    simulate(config)

    # Record of exactly what produced this run_dir's store/finv.npz — kept
    # separate from each training run's own train_<N>/config.yaml copy, since
    # training varies independently of the simulations.
    if "RUN" in config:
        run_dir = Path(config["RUN"]["run_dir"])
        shutil.copy(sys.argv[1], run_dir / "config.yaml")
