import os
import shutil

import numpy as np
import swyft
from torch.distributions import Normal, Uniform

from .params import PARAM_LABELS, MCMC_KEY_MAP
from .priors import resolve_priors, resolve_inference_params
from .io import load_array
from .network import Network
from .inference import load_network_from_checkpoint, predict_from_checkpoint
from .preprocessing import preprocess, preprocess_obs
from .coverage import run_coverage_test
from .simulator import build_simulator, _TruncatedNormalRejection
from .swyft_patches import apply_patches


def setup_latex_style():
    """
    Best-effort LaTeX styling matching the notebooks' corner-plot appearance;
    silently falls back to matplotlib defaults if LaTeX isn't available (e.g.
    on a machine without texlive).
    """
    import matplotlib as mpl
    from matplotlib import rc

    texlive_bin = "/home/abellan/texlive/2025/bin/x86_64-linux"
    if os.path.isdir(texlive_bin) and texlive_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = texlive_bin + ":" + os.environ.get("PATH", "")

    if shutil.which("latex") is not None:
        rc("text", usetex=True)
        rc("font", **{"family": "serif", "serif": ["cmr10"]})
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{{amsmath}}"


def _checkpoint_path(config):
    return os.path.join(config["STORES"]["checkpoint_path"], "best.ckpt")


def _hdi_thresholds(counts, cred_level=(0.68268, 0.95450)):
    """
    Density thresholds enclosing the given highest-density-interval credible
    masses — same sorted-cumulative-mass algorithm swyft's own (private)
    plot.plot._get_HDI_thresholds uses internally, reimplemented here (rather
    than importing swyft's underscore-prefixed function) so the filled-band
    contours below don't depend on swyft's private API.
    """
    flat = np.sort(counts.flatten())[::-1]
    total_mass = flat.sum()
    enclosed_mass = np.cumsum(flat)
    idx = [np.argmax(enclosed_mass >= total_mass * f) for f in cred_level]
    return sorted(flat[idx])


def load_mcmc_overlay(config):
    """
    Load the nested-sampling chain configured for corner-plot overlay, if any.

    Returns (chain_dict, weights) or (None, None) if PLOTTING.mcmc_path
    is unset/null. chain_dict maps physics parameter names (params.MCMC_KEY_MAP
    values, e.g. "multiplicative_bias_3") to 1d sample arrays; weights are
    linear (already normalized to sum to 1), converted from the file's LOG
    nested-sampling weights — no burn-in to discard (unlike MCMC chains,
    nested sampling has none).

    Expects a Nautilus-format .npz: a 0-d object array "chain" (.item() gives
    the dict above) and a "weights" array of log-weights. If a sibling 0-d
    object array "derived" is present (confirmed present in every Nautilus
    chain generated so far, holding {"sigma8_0": array} — see
    params.MCMC_KEY_MAP), its entries are merged into the same returned
    dict, so an overlay for a derived params_to_infer entry (currently only
    sigma8 has a chain-side equivalent) works via the same
    MCMC_KEY_MAP[name]/chain_dict[key] lookup mcmc_samples_for already uses
    for PARAMS-space names — no separate code path needed there.
    """
    chain_path = config.get("PLOTTING", {}).get("mcmc_path")
    if not chain_path:
        return None, None

    data = np.load(chain_path, allow_pickle=True)
    chain_dict = data["chain"].item()
    if "derived" in data:
        chain_dict = {**chain_dict, **data["derived"].item()}
    logw = np.asarray(data["weights"], dtype=np.float64)
    weights = np.exp(logw - logw.max())
    weights /= weights.sum()
    return chain_dict, weights


def mcmc_samples_for(param_names, chain_dict, weights):
    """
    Build a getdist MCSamples over param_names (merlin params.PARAMS names),
    mapped to chain_dict's own key names via params.MCMC_KEY_MAP. Raises
    KeyError naming the parameter if MCMC_KEY_MAP or chain_dict is missing
    one of param_names — e.g. TRAINING.params_to_infer includes a parameter
    the configured chain doesn't have.
    """
    from getdist import MCSamples

    mcmc_keys = [MCMC_KEY_MAP[name] for name in param_names]
    samples = np.column_stack([chain_dict[k] for k in mcmc_keys])
    labels = [PARAM_LABELS.get(name, name) for name in param_names]
    return MCSamples(samples=samples, weights=weights, names=mcmc_keys, labels=labels)


def plot_corner_mode(config, output_path, smooth=None, bins=None):
    """
    Corner plot of every TRAINING.params_to_infer parameter (LaTeX labels from
    params.PARAM_LABELS, fiducial truth lines from FIDUCIAL), evaluated
    on the mock observation using the best checkpoint in STORES.checkpoint_path.

    smooth, bins : Gaussian-kernel smoothing and bin count passed to
        swyft.plot_corner and get_pdf for Merlin's density estimate -- higher
        smooth smooths more. Default to config["PLOTTING"]["smooth_swyft"]/
        ["nbins_swyft"] (falling back to 1.0/100 if unset); pass explicitly
        to override the config for a one-off plot.

    If PLOTTING.mcmc_path is set, overlays that nested-sampling chain in
    green (see load_mcmc_overlay) — every parameter in TRAINING.params_to_infer
    must have an entry in params.MCMC_KEY_MAP and in the chain itself, or this
    raises a KeyError naming the missing one. The overlay is drawn from
    getdist density estimates (get1DDensity/get2DDensity) plotted directly via
    matplotlib, rather than getdist's own plot_1d/plot_2d axis-drawing methods
    — those inject their own ticks/labels into the shared axes, which would
    otherwise need stripping back out afterward.
    """
    import matplotlib.pyplot as plt

    setup_latex_style()
    apply_patches()

    plotting_cfg = config.get("PLOTTING", {})
    if smooth is None:
        smooth = plotting_cfg.get("smooth_swyft", 1.0)
    if bins is None:
        bins = plotting_cfg.get("nbins_swyft", 100)

    param_names, param_indices = resolve_inference_params(config)
    fiducial, _, _, _ = resolve_priors(config)
    n_params = len(fiducial)
    N_plot = len(param_names)
    marginals = Network._get_marginals(N_plot)

    V_proj = load_array(config["PCA"]["SVD"])
    Lfid   = np.load(config["OBSERVATION"]["LFID"])
    obs    = np.load(config["OBSERVATION"]["OBS"], allow_pickle=True).item()
    obs_sample = preprocess_obs(obs, Lfid, config=config)

    # Truth/axvline value for each inferred parameter. PARAMS-space indices
    # (< n_params) come from FIDUCIAL as before; derived-space indices
    # (>= n_params, see priors.resolve_inference_params's concatenated index
    # space) have no FIDUCIAL entry at all (they're computed, not sampled) —
    # their truth instead comes from the observation's own "derived" array,
    # which generate_observation() already populated at the fiducial
    # cosmology for free (see simulator.Simulator.generate_observation).
    truth = [
        fiducial[i] if i < n_params else obs["derived"][i - n_params]
        for i in param_indices
    ]

    predictions = predict_from_checkpoint(_checkpoint_path(config), V_proj, obs_sample, config)

    chain_dict, weights = load_mcmc_overlay(config)
    mcmc_samples = mcmc_samples_for(param_names, chain_dict, weights) if chain_dict is not None else None
    mcmc_keys = [MCMC_KEY_MAP[name] for name in param_names] if mcmc_samples is not None else None

    parnames = [f"z[{i}]" for i in range(N_plot)]
    labels   = [PARAM_LABELS.get(name, name) for name in param_names]
    # Figure size scales linearly with N_plot (figsize=2.2*N_plot per side),
    # so fixed label/tick fontsizes read proportionally smaller on larger
    # triangle plots (e.g. 13x13 PHOTOZ) than on smaller ones (e.g. 5x5
    # COSMO) -- scale up to compensate, clamped at the original fontsize for
    # N_plot<=5 so small corner plots look exactly as before. Sqrt (not
    # linear like legend_fontsize below) -- a full linear scale-up reads as
    # oversized once there are many more, smaller panels crowded together.
    label_fontsize = max(16, 16 * np.sqrt(N_plot / 5))
    fontsize_tick = max(13, 13 * np.sqrt(N_plot / 5))
    color_merlin = "tab:blue"
    color_mcmc   = "tab:orange"

    from matplotlib.colors import to_rgba
    from swyft.lightning.utils import get_pdf

    fig, axes = plt.subplots(N_plot, N_plot, figsize=(2.2 * N_plot, 2.2 * N_plot))
    swyft.plot_corner(
        predictions, parnames, color=color_merlin, cmap=None, fig=fig, bins=bins, smooth=smooth,
        contours_1d=False, labels=labels,
        label_args={"fontsize": label_fontsize, "labelpad": 10}, linewidth=1.5,
    )

    # Filled credible-region bands for Merlin's 2D panels, layered underneath
    # (zorder) the crisp contour lines swyft.plot_corner already drew above —
    # swyft's own cmap= option gives a continuous density heatmap instead of
    # discrete bands, which reads as barely-there against a white background,
    # so the bands are recomputed here from swyft's own public get_pdf utility
    # (same bins/smooth swyft.plot_corner was called with, for consistency).
    merlin_band_colors = [to_rgba(color_merlin, 0.3), to_rgba("darkblue", 0.75)]
    for i, j in marginals:
        ax = axes[j, i]
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        counts, xy = get_pdf(predictions, [parnames[i], parnames[j]], bins=bins, smooth=smooth)
        xbins, ybins = xy[:, 0], xy[:, 1]
        levels = _hdi_thresholds(counts)
        ax.contourf(xbins, ybins, counts.T, levels=levels + [counts.max() + 1e-12],
                    colors=merlin_band_colors, zorder=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # Force axis ranges to match the configured PRIORS bounds (post-Fisher-
    # overlay, i.e. what sim.sample_z actually drew from — the same object
    # infer()/predict_from_checkpoint used above) rather than swyft's default
    # auto-scaling to the posterior sample spread — which for Normal-type
    # priors (e.g. the m_i shear-calibration nuisance params) auto-scales to
    # whatever the raw posterior samples happen to span, routinely far wider
    # than where the marginal actually has support. Uniform priors and
    # Fisher-truncated Normal priors (PRIORS.use_Fisher_priors — see
    # simulator._TruncatedNormalRejection) both use their exact hard bounds directly;
    # a plain (untruncated) Normal gets an explicit mean +/- N*sigma window
    # instead, since it has no hard bounds of its own (fixed-type parameters
    # keep swyft's auto range — they have no meaningful "prior range", since
    # they're never sampled/varied at all). Derived-space indices (>= n_params)
    # are skipped entirely — a deterministic, computed quantity (sigma8/
    # Omega_m/S8) has no torch.distributions prior object at all, so those
    # axes just keep swyft's own auto-range, same as any other
    # no-override parameter.
    NORMAL_RANGE_NSIGMA = 5
    sim = build_simulator(config)
    prior_ranges = {}
    for i, idx in enumerate(param_indices):
        if idx >= n_params:
            continue
        prior = sim.sample_z.priors[idx]
        if isinstance(prior, Uniform):
            prior_ranges[i] = (prior.low.item(), prior.high.item())
        elif isinstance(prior, _TruncatedNormalRejection):
            prior_ranges[i] = (prior.lower, prior.upper)
        elif isinstance(prior, Normal):
            mean, std = prior.loc.item(), prior.scale.item()
            prior_ranges[i] = (mean - NORMAL_RANGE_NSIGMA * std, mean + NORMAL_RANGE_NSIGMA * std)
    for i, (lo, hi) in prior_ranges.items():
        axes[i, i].set_xlim(lo, hi)
    for i, j in marginals:
        if i in prior_ranges:
            axes[j, i].set_xlim(*prior_ranges[i])
        if j in prior_ranges:
            axes[j, i].set_ylim(*prior_ranges[j])

    if mcmc_samples is not None:
        # Grab swyft's axis limits first, so the MCMC densities are evaluated
        # and drawn on the same scale as the SBI posterior.
        xlims = [axes[i, i].get_xlim() for i in range(N_plot)]
        ylims_diag = [axes[i, i].get_ylim() for i in range(N_plot)]
        xlims_off = {(j, i): axes[j, i].get_xlim() for i, j in marginals}
        ylims_off = {(j, i): axes[j, i].get_ylim() for i, j in marginals}
        for i in range(N_plot):
            density = mcmc_samples.get1DDensity(mcmc_keys[i])
            xs = np.linspace(*xlims[i], 500)
            ys = density.Prob(xs)
            # Rescale to Merlin's own curve height (already drawn on this ax
            # by swyft.plot_corner above) rather than swyft's default ylim
            # (typically ~1.3x the Merlin peak, for its own headroom) -- so
            # both curves' peaks land at the same height instead of Nautilus
            # reading taller than Merlin purely from the axis's own margin.
            merlin_peak = axes[i, i].lines[0].get_ydata().max()
            if ys.max() > 0:
                ys = ys / ys.max() * merlin_peak
            axes[i, i].plot(xs, ys, color=color_mcmc, lw=2.0)
            axes[i, i].set_xlim(xlims[i])
            # Headroom above the shared peak height (both curves now peak at
            # merlin_peak) instead of swyft's own (now-irrelevant) ylim.
            axes[i, i].set_ylim(ylims_diag[i][0], merlin_peak * 1.15)
        for i, j in marginals:
            if xlims_off[(j, i)][0] == xlims_off[(j, i)][1] or ylims_off[(j, i)][0] == ylims_off[(j, i)][1]:
                # Degenerate (zero-width) swyft posterior on this panel — e.g. a
                # near-delta-function posterior for a very tightly-constrained
                # parameter — nothing meaningful to overlay here.
                continue
            density2d = mcmc_samples.get2DDensity(mcmc_keys[i], mcmc_keys[j], normalized=True)
            xs = np.linspace(*xlims_off[(j, i)], 200)
            ys = np.linspace(*ylims_off[(j, i)], 200)
            XX, YY = np.meshgrid(xs, ys)
            # NOT density2d.Prob(XX, YY) — installed getdist==1.4's Density2D.Prob
            # calls self.__call__(...) but never returns it, so it always gives
            # None. Call the (callable) density object directly instead; pass
            # the 1D xs/ys with grid=True (not the meshgridded XX/YY) to get a
            # proper (len(xs), len(ys)) grid back — then transpose to match
            # meshgrid's (ny, nx) convention for ax.contour.
            ZZ = density2d(xs, ys, grid=True).T
            levels = sorted(density2d.getContourLevels([0.68, 0.95]))
            axes[j, i].contour(XX, YY, ZZ, levels=levels, colors=color_mcmc, linewidths=2.0)
            axes[j, i].set_xlim(xlims_off[(j, i)])
            axes[j, i].set_ylim(ylims_off[(j, i)])

    # Rotate x-tick labels only (not y) -- unrotated, they clump together
    # into an unreadable smear once there are many closely-spaced bottom-row
    # panels (e.g. 13x13 PHOTOZ). Harmless to apply to every panel here
    # rather than just the bottom row: swyft.plot_corner already hides tick
    # labels on all but the bottom/left edge panels, so rotating a hidden
    # label has no visible effect.
    XTICK_ROTATION = 45
    for i in range(N_plot):
        axes[i, i].axvline(truth[i], color="black", lw=1., linestyle="dashed")
        axes[i, i].minorticks_on()
        axes[i, i].tick_params(axis="both", which="major", direction="in",
                                labelsize=fontsize_tick, size=4)
        axes[i, i].tick_params(axis="both", which="minor", direction="in",
                                labelsize=fontsize_tick, size=2, left=False)
        axes[i, i].tick_params(axis="x", rotation=XTICK_ROTATION)
    for i, j in marginals:
        axes[j, i].axvline(truth[i], color="black", lw=1., linestyle="dashed")
        axes[j, i].axhline(truth[j], color="black", lw=1., linestyle="dashed")
        axes[j, i].minorticks_on()
        axes[j, i].tick_params(axis="both", which="major", direction="in",
                                labelsize=fontsize_tick, size=4)
        axes[j, i].tick_params(axis="both", which="minor", direction="in",
                                labelsize=fontsize_tick, size=2)
        axes[j, i].tick_params(axis="x", rotation=XTICK_ROTATION)
    for i, j in marginals:
        [x.set_linewidth(1.5) for x in axes[j, i].spines.values()]
    for i in range(N_plot):
        [x.set_linewidth(1.5) for x in axes[i, i].spines.values()]

    if mcmc_samples is not None:
        from matplotlib.lines import Line2D
        legend_lines = [
            Line2D([0], [0], color=color_merlin, linewidth=4, linestyle="-"),
            Line2D([0], [0], color=color_mcmc, linewidth=4, linestyle="-"),
        ]
        # Same adaptive-fontsize reasoning as label_fontsize/fontsize_tick above.
        legend_fontsize = max(20, 20 * N_plot / 5)
        axes[0, N_plot - 1].legend(
            legend_lines, ["Merlin", "Nautilus"], loc="upper right",
            fontsize=legend_fontsize, frameon=False,
        )

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_mode(config, output_path, n_test=1000):
    """
    Coverage/calibration plot for the best checkpoint in
    STORES.checkpoint_path, evaluated against the last n_test simulations in
    the store. Delegates the actual diagnostic + plotting to
    coverage.run_coverage_test (same one used during interactive/notebook use)
    so both stay visually and behaviorally identical.

    Unlike the notebook flow (which reuses a store_samples already
    preprocessed once for training), this is a fresh process with nothing to
    reuse — so it slices the RAW store down to the last n_test simulations
    (all run_coverage_test's default n_test=1000 ever uses) BEFORE whitening,
    and reuses the checkpoint's own already-trained V_proj (recompute_pca
    forced off) instead of re-deriving PCA from scratch. Whitening + PCA over
    the full store otherwise costs several minutes and tens of GB of memory
    for no benefit, since 99%+ of it would just be discarded.
    """
    setup_latex_style()

    V_proj = load_array(config["PCA"]["SVD"])
    network = load_network_from_checkpoint(_checkpoint_path(config), V_proj, config)

    store = swyft.ZarrStore(config["SIMULATION"]["store_path"]).get_sample_store()
    Lfid  = np.load(config["OBSERVATION"]["LFID"])

    n_test = min(n_test, len(store["z"]))
    store_tail = {k: store[k][-n_test:] for k in store.keys()}
    coverage_config = {**config, "PCA": {**config["PCA"], "recompute_pca": False}}
    store_samples, _ = preprocess(store_tail, Lfid, coverage_config)

    trainer = swyft.SwyftTrainer(
        accelerator=config.get("TRAINING", {}).get("accelerator", "auto"), devices=1,
        precision=64, logger=False, enable_checkpointing=False, enable_model_summary=False,
        enable_progress_bar=False,
    )

    run_coverage_test(trainer, network, store_samples, config, output_path, n_test=n_test)


def plot_loss_mode(config, output_path):
    """
    Training/validation loss vs. epoch, read from STORES.csv_logs/metrics.csv
    (written by pytorch-lightning's CSVLogger during train.train). Train is
    logged per-step only (swyft.SwyftModule.training_step), so it's averaged
    down to one point per epoch to match the once-per-epoch val_loss.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    setup_latex_style()

    csv_path = os.path.join(config["STORES"]["csv_logs"], "metrics.csv")
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(6, 4))
    if "train_loss" in df.columns:
        series = df.dropna(subset=["train_loss"]).groupby("epoch")["train_loss"].mean()
        ax.plot(series.index, series.values, color="green", linewidth=1.5,
                linestyle="-", label="Training")
    if "val_loss" in df.columns:
        series = df.dropna(subset=["val_loss"])
        ax.plot(series["epoch"], series["val_loss"], color="purple", linewidth=1.5,
                linestyle="--", label="Validation")

    ax.set_xlabel("Epoch", fontsize=16, labelpad=10)
    ax.set_ylabel("Loss", fontsize=16, labelpad=10)
    ax.legend(frameon=False, fontsize=14)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", direction="in", labelsize=13, size=4)
    ax.tick_params(axis="both", which="minor", direction="in", labelsize=13, size=2)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
