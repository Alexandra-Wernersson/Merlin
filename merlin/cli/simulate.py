import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: merlin-simulate <config.yaml>")
        sys.exit(1)
    from ..config import load_config
    from ..simulate import simulate
    config = load_config(sys.argv[1])
    simulate(config)
