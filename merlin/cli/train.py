import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: merlin-train <config.yaml>")
        sys.exit(1)
    import numpy as np
    import torch
    import swyft
    from ..config import load_config
    from ..preprocessing import preprocess, preprocess_obs
    from ..train import train
    from ..inference import infer

    config = load_config(sys.argv[1])

    store = swyft.ZarrStore(config["SIMULATION"]["store_path"]).get_sample_store()
    Lfid  = np.load(config["OBSERVATION"]["LFID"])

    store_samples, V_proj = preprocess(store, Lfid, config)
    network, trainer      = train(store_samples, V_proj, config,
                                  num_workers=config["TRAINING"].get("num_workers", 12))

    obs        = np.load(config["OBSERVATION"]["OBS"], allow_pickle=True).item()
    obs_sample = preprocess_obs(obs, Lfid, config=config)

    infer(trainer, network, obs_sample, config)

    if config["TRAINING"].get("coverage_test", False):
        fiducial = list(config["FIDUCIAL VALUES"].values())
        Finv     = np.load(config["FINV"]["finv_file"])
        sigmas   = np.sqrt(np.diag(Finv))
        from training_utils import run_coverage_test
        run_coverage_test(trainer, network, store_samples, fiducial, sigmas,
                          config["STORES"]["coverage_plot"])
