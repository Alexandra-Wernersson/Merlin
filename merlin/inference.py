import swyft

from .network import Network
from .priors import resolve_inference_params
from .simulator import build_simulator


def infer(trainer, network, obs_sample, config, n_samples=500_000):
    """
    Run inference given a trained network and a Cholesky-whitened observation.

    Prior samples are drawn directly from the same Simulator/PriorSampler used
    to generate the training data (see simulator.build_simulator), so the
    inference prior is guaranteed to match the simulation prior exactly for
    every parameter — including the uniform-bounded ones (fiducial ±
    sigma_scale*sigma) and the Gaussian-distributed photo-z nuisance
    dimensions — rather than re-deriving bounds by hand, which could silently
    drift out of sync with simulator.py. Network.forward already knows how to
    pick config["TRAINING"]["params_to_infer"]'s subset out of the full prior
    vector (the same way it does during training/validation).

    Note: sim.sample_z(shape=(n_samples,)) is used instead of
    sim.sample(N=n_samples, targets=["z"]) — the latter samples one at a time
    in a plain Python loop (swyft's Simulator.sample loops per-sample even when
    only the cheap "z" root node is requested) and takes ~13 minutes for
    n_samples=500_000; sample_z's vectorized `shape` argument gives the exact
    same samples in well under a second.

    Predictions are returned, not persisted to disk — they're cheap to recompute
    from a saved checkpoint (see predict_from_checkpoint) so there's no need to
    store them.
    """
    print("Running inference...")
    sim = build_simulator(config)
    prior_samples = swyft.Samples(z=sim.sample_z(shape=(n_samples,)))

    return trainer.infer(network, obs_sample, prior_samples)


def load_network_from_checkpoint(checkpoint_path, V_proj, config):
    """
    Reload a trained Network from a saved checkpoint (see train.train), without
    needing the original (network, trainer) objects in memory — e.g. for
    plotting after training finished in a separate job.

    V_proj must match what the checkpoint was trained with (e.g. reloaded from
    config["PCA"]["SVD"]), and config["NETWORK"]["num_feat_param"]/
    "hidden_feat_compress"/"num_blocks_ratios"/"hidden_feat_ratios" must match
    what was used at training time — the checkpoint's state_dict keys depend on
    the exact layer structure, so a mismatch here will fail to load.
    "dropout_ratios" doesn't affect the state_dict (no learnable params) and has
    no effect on predictions either way — network.eval() (called below) always
    disables dropout — but is passed through anyway for consistency.
    """
    network_cfg = config.get("NETWORK", {})
    _, param_indices = resolve_inference_params(config)

    network = Network.load_from_checkpoint(
        checkpoint_path, V_proj=V_proj, param_indices=param_indices,
        num_feat_param=network_cfg.get("num_feat_param", 1),
        hidden_feat_compress=network_cfg.get("hidden_feat_compress", [2048, 512]),
        num_blocks_ratios=network_cfg.get("num_blocks_ratios", 6),
        hidden_feat_ratios=network_cfg.get("hidden_feat_ratios", 64),
        dropout_ratios=network_cfg.get("dropout_ratios", 0.1),
    )
    network.eval()
    return network


def predict_from_checkpoint(checkpoint_path, V_proj, obs_sample, config, n_samples=500_000):
    """
    Reload a trained Network from a saved checkpoint and run inference on
    obs_sample — see load_network_from_checkpoint for the checkpoint-reload
    details this delegates to.
    """
    network = load_network_from_checkpoint(checkpoint_path, V_proj, config)

    trainer = swyft.SwyftTrainer(
        accelerator=config.get("TRAINING", {}).get("accelerator", "auto"), devices=1, precision=64,
        logger=False, enable_checkpointing=False, enable_model_summary=False,
        enable_progress_bar=False,
    )

    return infer(trainer, network, obs_sample, config, n_samples=n_samples)
