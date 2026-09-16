import swyft

from .network import Network
from .params import DERIVED_PARAMS
from .priors import resolve_inference_params
from .simulator import build_simulator


def _derived_prior_samples(config, n_samples):
    """
    Build prior_samples by reusing the training store's own simulated
    ("z", "derived") rows, rather than drawing fresh from sim.sample_z --
    sigma8/Omega_m/S8 have no closed-form prior (a deterministic pushforward
    of the sampled cosmology, see Simulator._compute_derived), and
    re-running cloelib per sample would take hours at n_samples=500_000.

    Reusing store rows preserves every z-derived correlation exactly, and is
    at least as statistically valid as a fresh draw. Takes the first
    min(n_samples, len(store)) rows (order doesn't matter, i.i.d.) -- note
    this silently caps prior_samples below n_samples whenever the store
    itself has fewer than n_samples rows.

    Raises
    ------
    ValueError — if the store has no "derived" array, i.e. it was simulated
    without CLOELIB_SETTINGS.add_derived covering the requested name(s).
    """
    store_path = config["SIMULATION"]["store_path"]
    store = swyft.ZarrStore(store_path).get_sample_store()
    if "derived" not in store:
        raise ValueError(
            f"TRAINING.params_to_infer requests a derived parameter, but the "
            f"store at {store_path!r} has no \"derived\" array — it was "
            f"simulated without CLOELIB_SETTINGS.add_derived covering it. "
            f"Re-simulate (or grow) the store with add_derived set."
        )
    n = min(n_samples, len(store["z"]))
    return swyft.Samples(z=store["z"][:n], derived=store["derived"][:n])


def infer(trainer, network, obs_sample, config, n_samples=500_000):
    """
    Run inference given a trained network and a Cholesky-whitened observation.

    Prior samples are drawn from the same Simulator/PriorSampler used to
    generate the training data (see simulator.build_simulator), so the
    inference prior matches the simulation prior exactly for every
    parameter, rather than re-deriving bounds by hand.

    Uses sim.sample_z(shape=(n_samples,)) rather than
    sim.sample(N=n_samples, targets=["z"]) -- the latter loops per-sample
    (swyft's Simulator.sample does even for the cheap "z" root node) and
    takes ~13 min at n_samples=500_000; sample_z is vectorized and takes
    under a second. "z"/"derived" pairs are reused from the training store
    instead (see _derived_prior_samples) whenever:
    - any inferred parameter is derived (sigma8/Omega_m/S8) -- required,
      these have no closed-form prior to draw fresh from;
    - CLOELIB_SETTINGS.restrict_prior_for_derived was active for this store
      (sim.derived_box non-empty), even for COSMO-only params_to_infer:
      sample_z(shape=(n,)) draws from the *unrestricted* Fisher box (that
      flag never touches PriorSampler itself, only simulate.py's rejection
      loop), which would silently mismatch the network's actual training
      distribution otherwise.

    Predictions are returned, not persisted -- cheap to recompute from a
    checkpoint (see predict_from_checkpoint).
    """
    print("Running inference...")
    names, _ = resolve_inference_params(config)
    sim = build_simulator(config)
    if any(name in DERIVED_PARAMS for name in names) or sim.derived_box:
        prior_samples = _derived_prior_samples(config, n_samples)
    else:
        prior_samples = swyft.Samples(z=sim.sample_z(shape=(n_samples,)))

    return trainer.infer(network, obs_sample, prior_samples)


def load_network_from_checkpoint(checkpoint_path, V_proj, config):
    """
    Reload a trained Network from a saved checkpoint without needing the
    original (network, trainer) objects in memory.

    V_proj and config["NETWORK"]["num_feat_param"]/"hidden_feat_compress"/
    "num_blocks_ratios"/"hidden_feat_ratios" must match training time -- the
    checkpoint's state_dict keys depend on the exact layer structure.
    "dropout_ratios" has no effect on predictions (network.eval() disables
    dropout) but is passed through for consistency.
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
