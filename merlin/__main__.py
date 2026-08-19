import argparse
import shutil
import sys
from pathlib import Path

from .config import load_config
from .simulate import simulate


def main():
    parser = argparse.ArgumentParser(prog="merlin_module")
    sub = parser.add_subparsers(dest="command")

    sim_parser = sub.add_parser("simulate", help="Run the simulation loop and fill a ZarrStore")
    sim_parser.add_argument("config", help="Path to YAML config file")
    sim_parser.add_argument("--store-path", help="Override SIMULATION.store_path in the config")
    sim_parser.add_argument("--n-sims", type=int, help="Override SIMULATION.N_sims")
    sim_parser.add_argument("--n-workers", type=int, help="Override SIMULATION.n_workers")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "simulate":
        config = load_config(args.config)
        if args.store_path:
            config["SIMULATION"]["store_path"] = args.store_path
        if args.n_sims:
            config["SIMULATION"]["N_sims"] = args.n_sims
        if args.n_workers:
            config["SIMULATION"]["n_workers"] = args.n_workers
        simulate(config)
        if "RUN" in config:
            run_dir = Path(config["RUN"]["run_dir"])
            shutil.copy(args.config, run_dir / "config.yaml")


if __name__ == "__main__":
    main()
