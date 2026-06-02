import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: merlin-generate-obs <config.yaml>")
        sys.exit(1)
    from ..config import load_config
    from ..observation import generate_observation
    config = load_config(sys.argv[1])
    generate_observation(config)
