import os
import shutil

import numpy as np
import swyft
from torch.distributions import Normal, Uniform

from .params import PARAM_LABELS, MCMC_KEY_MAP, resolve_derived_names
from .priors import (
    resolve_priors, resolve_inference_params, load_derived_fisher_sigmas, DERIVED_REJECTION_SIGMA_SCALE,
)
from .io import load_array
from .network import Network
from .observation import generate_observation_for_fiducial
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
    masses. Reimplements swyft's private plot._get_HDI_thresholds so the
    filled-band contours below don't depend on swyft's private API.
    """
    flat = np.sort(counts.flatten())[::-1]
    total_mass = flat.sum()
    enclosed_mass = np.cumsum(flat)
    idx = [np.argmax(enclosed_mass >= total_mass * f) for f in cred_level]
    return sorted(flat[idx])


# sum(m_nu)[eV] = Omega_nu * h^2 * _NEUTRINO_H2_EV, standard relation for the
# relic neutrino background (N_mnu=1 here, so sum(m_nu) is just FIDUCIAL.mnu).
_NEUTRINO_H2_EV = 93.14


def _add_mcmc_derived(chain_dict, config):
    """
    Compute Omega_m/S8 from the chain's own H0/Omega_b0/Omega_cdm0 (+ the
    merged-in sigma8_0) when CLOELIB_SETTINGS.add_derived requests them --
    Nautilus chains have no such columns. mnu is fixed at FIDUCIAL.mnu since
    it isn't varied in the Nautilus run either.
    """
    requested = resolve_derived_names(config.get("CLOELIB_SETTINGS", {}).get("add_derived"))
    if "Omega_m" not in requested and "S8" not in requested:
        return chain_dict

    h = chain_dict["H0"] / 100.0
    Omega_nu = config["FIDUCIAL"]["mnu"] / (_NEUTRINO_H2_EV * h**2)
    Omega_m = chain_dict["Omega_b0"] + chain_dict["Omega_cdm0"] + Omega_nu
    chain_dict = {**chain_dict, "Omega_m": Omega_m}
    if "S8" in requested:
        chain_dict["S8"] = chain_dict["sigma8_0"] * np.sqrt(Omega_m / 0.3)
    return chain_dict


def _weighted_quantile_range(v, w, lo_q=0.001, hi_q=0.999, pad=0.3):
    """
    [lo_q, hi_q] weighted-quantile range of v under normalized weights w,
    padded by `pad` of its width. Used to zoom derived-parameter axes
    (sigma8/Omega_m/S8) to where the posterior actually has support, since
    they have no prior object to fall back on (see the PRIORS-bounds block
    in plot_corner_mode) and swyft's own auto-range spans the full store,
    not the posterior.
    """
    order = np.argsort(v)
    v_sorted, w_sorted = v[order], w[order]
    cw = np.cumsum(w_sorted)
    cw /= cw[-1]
    lo, hi = np.interp([lo_q, hi_q], cw, v_sorted)
    margin = pad * (hi - lo)
    return lo - margin, hi + margin


def load_mcmc_overlay(config):
    """
    Load the nested-sampling chain configured for corner-plot overlay, if any.

    Returns (chain_dict, weights) or (None, None) if PLOTTING.mcmc_path is
    unset/null. chain_dict maps physics parameter names (params.MCMC_KEY_MAP
    values) to 1d sample arrays; weights are linear (normalized to sum to 1),
    converted from the file's log nested-sampling weights (no burn-in, unlike
    MCMC chains).

    Expects a Nautilus-format .npz: a 0-d object array "chain" (.item() gives
    the dict above) and a "weights" log-weight array. A sibling "derived"
    0-d object array, if present (e.g. {"sigma8_0": array}), is merged into
    the same dict so a derived params_to_infer entry overlays through the
    same MCMC_KEY_MAP lookup as PARAMS-space names. Omega_m/S8, if requested
    via CLOELIB_SETTINGS.add_derived, are computed from the chain itself
    (see _add_mcmc_derived) since Nautilus never stored them directly.
    """
    chain_path = config.get("PLOTTING", {}).get("mcmc_path")
    if not chain_path:
        return None, None

    data = np.load(chain_path, allow_pickle=True)
    chain_dict = data["chain"].item()
    if "derived" in data:
        chain_dict = {**chain_dict, **data["derived"].item()}
    chain_dict = _add_mcmc_derived(chain_dict, config)
    logw = np.asarray(data["weights"], dtype=np.float64)
    weights = np.exp(logw - logw.max())
    weights /= weights.sum()
    return chain_dict, weights


def mcmc_samples_for(param_names, chain_dict, weights):
    """
    Build a getdist MCSamples over param_names, mapped to chain_dict's key
    names via params.MCMC_KEY_MAP. Raises KeyError if MCMC_KEY_MAP or
    chain_dict is missing a param_names entry (e.g. TRAINING.params_to_infer
    includes a parameter the configured chain doesn't have).
    """
    from getdist import MCSamples

    mcmc_keys = [MCMC_KEY_MAP[name] for name in param_names]
    samples = np.column_stack([chain_dict[k] for k in mcmc_keys])
    labels = [PARAM_LABELS.get(name, name) for name in param_names]
    return MCSamples(samples=samples, weights=weights, names=mcmc_keys, labels=labels)


def plot_corner_mode(config, output_path, smooth=None, bins=None, fiducial_override=None):
    """
    Corner plot of every TRAINING.params_to_infer parameter (LaTeX labels from
    params.PARAM_LABELS, fiducial truth lines from FIDUCIAL), evaluated on
    the mock observation using the best checkpoint in STORES.checkpoint_path.

    smooth, bins : Gaussian-kernel smoothing and bin count passed to
        swyft.plot_corner and get_pdf (higher smooth = smoother). Default to
        config["PLOTTING"]["smooth_swyft"]/["nbins_swyft"] (1.0/100 if
        unset); pass explicitly to override for a one-off plot.

    fiducial_override : optional {param_name: value} dict. When set (or
        when PLOTTING.eval_fiducial is set in config and this arg is None),
        evaluates the checkpoint against a freshly-generated observation at
        FIDUCIAL merged with these overrides, instead of the train_<N>'s own
        saved mock observation -- see observation.generate_observation_for_fiducial.
        Truth lines/axvlines use the overridden fiducial too. Results are
        only meaningful if the override stays inside the trained prior box.

    If PLOTTING.mcmc_path is set, overlays that nested-sampling chain (see
    load_mcmc_overlay) — every TRAINING.params_to_infer parameter must have
    an entry in params.MCMC_KEY_MAP and in the chain, or this raises
    KeyError. Drawn from getdist density estimates plotted directly via
    matplotlib rather than getdist's own plot_1d/plot_2d, which would inject
    their own ticks/labels into the shared axes.
    """
    import matplotlib.pyplot as plt

    setup_latex_style()
    apply_patches()

    plotting_cfg = config.get("PLOTTING", {})
    if smooth is None:
        smooth = plotting_cfg.get("smooth_swyft", 1.0)
    if bins is None:
        bins = plotting_cfg.get("nbins_swyft", 100)
    if fiducial_override is None:
        fiducial_override = plotting_cfg.get("eval_fiducial")

    active_config = config
    if fiducial_override:
        active_config = {**config, "FIDUCIAL": {**config["FIDUCIAL"], **fiducial_override}}

    param_names, param_indices = resolve_inference_params(config)
    fiducial, _, _, _ = resolve_priors(active_config)
    n_params = len(fiducial)
    N_plot = len(param_names)
    marginals = Network._get_marginals(N_plot)

    V_proj = load_array(config["PCA"]["SVD"])
    Lfid   = np.load(config["OBSERVATION"]["LFID"])
    if fiducial_override:
        print(f"Evaluating at overridden fiducial: {fiducial_override}")
        obs = generate_observation_for_fiducial(active_config)
    else:
        obs = np.load(config["OBSERVATION"]["OBS"], allow_pickle=True).item()
    obs_sample = preprocess_obs(obs, Lfid, config=config)

    # Truth/axvline value per inferred parameter: PARAMS-space indices
    # (< n_params) come from FIDUCIAL; derived-space indices (>= n_params,
    # see priors.resolve_inference_params) have no FIDUCIAL entry, so their
    # truth comes from the observation's own "derived" array instead.
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
    # Figure size scales linearly with N_plot, so fixed fontsizes read
    # proportionally smaller on larger triangle plots; scale up to
    # compensate, clamped at the original size for N_plot<=5. Sqrt (not
    # linear) since a full linear scale-up looks oversized with many panels.
    label_fontsize = max(16, 16 * np.sqrt(N_plot / 5))
    fontsize_tick = max(13, 13 * np.sqrt(N_plot / 5))
    color_merlin = "tab:blue"
    color_mcmc   = "tab:orange"

    from matplotlib.colors import to_rgba
    from swyft.lightning.utils import get_pdf, get_weighted_samples

    fig, axes = plt.subplots(N_plot, N_plot, figsize=(2.2 * N_plot, 2.2 * N_plot))
    swyft.plot_corner(
        predictions, parnames, color=color_merlin, cmap=None, fig=fig, bins=bins, smooth=smooth,
        contours_1d=False, labels=labels,
        label_args={"fontsize": label_fontsize, "labelpad": 10}, linewidth=1.5,
    )

    # Filled credible-region bands for Merlin's 2D panels, drawn underneath
    # swyft.plot_corner's contour lines (its own cmap= option gives a
    # continuous heatmap that reads as barely-there on white). Recomputed
    # via swyft's public get_pdf, same bins/smooth as plot_corner above.
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

    # Force axis ranges to the configured PRIORS bounds (post-Fisher-overlay,
    # what sim.sample_z actually drew from) rather than swyft's default
    # auto-scaling to the posterior sample spread — which for Normal-type
    # priors (e.g. m_i) can span far wider than where the marginal has
    # support. Uniform and Fisher-truncated Normal priors use their exact
    # bounds; a plain Normal gets mean +/- N*sigma instead. Derived-space
    # parameters (sigma8/Omega_m/S8) have no prior object: if
    # restrict_prior_for_derived was active for this store, its rows are
    # already confined to fiducial +/- CLOELIB_SETTINGS.sigma_scale_derived*
    # sigma_Fisher (see simulator._sample_z_derived_rejection), so that's the correct,
    # non-circular range to show -- using the posterior's own spread here
    # would just reproduce whatever the network happened to learn on already-
    # restricted training data. Otherwise (unrestricted store), swyft's own
    # auto-range spans the whole store, far wider than the posterior, so
    # those axes are zoomed to a weighted-quantile range of the posterior
    # samples themselves instead.
    NORMAL_RANGE_NSIGMA = 5
    sim = build_simulator(config)
    restrict_derived = config["CLOELIB_SETTINGS"].get("restrict_prior_for_derived", False)
    if restrict_derived:
        derived_plot_names = [param_names[i] for i, idx in enumerate(param_indices) if idx >= n_params]
        derived_sigma, derived_fid = load_derived_fisher_sigmas(config["PRIORS"]["finv_file"], derived_plot_names)
        sigma_scale_derived = config["CLOELIB_SETTINGS"].get(
            "sigma_scale_derived", DERIVED_REJECTION_SIGMA_SCALE)
    # PLOTTING.derived_axis_ranges: optional {param_name: [lo, hi]} override
    # for derived-space (sigma8/Omega_m/S8) axis ranges, taking precedence
    # over both the restrict_prior_for_derived and auto weighted-quantile
    # cases below -- lets a fixed range be pinned across runs/train_<N>s
    # (e.g. matching a wider-posterior run) rather than each plot
    # auto-scaling to its own, possibly narrower, posterior spread.
    fixed_derived_ranges = plotting_cfg.get("derived_axis_ranges", {})
    prior_ranges = {}
    for i, idx in enumerate(param_indices):
        if idx >= n_params:
            name = param_names[i]
            if name in fixed_derived_ranges:
                lo, hi = fixed_derived_ranges[name]
                prior_ranges[i] = (lo, hi)
            elif restrict_derived:
                half_width = sigma_scale_derived * derived_sigma[name]
                prior_ranges[i] = (derived_fid[name] - half_width, derived_fid[name] + half_width)
            else:
                v, w = get_weighted_samples(predictions, parnames[i])
                prior_ranges[i] = _weighted_quantile_range(v.numpy().flatten(), w.numpy().flatten())
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
        # Grab swyft's axis limits first so MCMC densities are evaluated and
        # drawn on the same scale as the SBI posterior.
        xlims = [axes[i, i].get_xlim() for i in range(N_plot)]
        ylims_diag = [axes[i, i].get_ylim() for i in range(N_plot)]
        xlims_off = {(j, i): axes[j, i].get_xlim() for i, j in marginals}
        ylims_off = {(j, i): axes[j, i].get_ylim() for i, j in marginals}
        for i in range(N_plot):
            density = mcmc_samples.get1DDensity(mcmc_keys[i])
            xs = np.linspace(*xlims[i], 500)
            ys = density.Prob(xs)
            # Rescale to Merlin's curve height (not swyft's default ylim, which
            # has its own headroom) so both curves' peaks land at the same height.
            merlin_peak = axes[i, i].lines[0].get_ydata().max()
            if ys.max() > 0:
                ys = ys / ys.max() * merlin_peak
            axes[i, i].plot(xs, ys, color=color_mcmc, lw=3.0, linestyle="dashed")
            axes[i, i].set_xlim(xlims[i])
            axes[i, i].set_ylim(ylims_diag[i][0], merlin_peak * 1.15)
        for i, j in marginals:
            if xlims_off[(j, i)][0] == xlims_off[(j, i)][1] or ylims_off[(j, i)][0] == ylims_off[(j, i)][1]:
                # Degenerate (zero-width) swyft posterior on this panel — nothing to overlay.
                continue
            density2d = mcmc_samples.get2DDensity(mcmc_keys[i], mcmc_keys[j], normalized=True)
            xs = np.linspace(*xlims_off[(j, i)], 200)
            ys = np.linspace(*ylims_off[(j, i)], 200)
            XX, YY = np.meshgrid(xs, ys)
            # NOT density2d.Prob(XX, YY): getdist==1.4's Density2D.Prob calls
            # __call__ but doesn't return it, so it always gives None. Call the
            # density object directly with 1D xs/ys and grid=True instead, then
            # transpose to match meshgrid's (ny, nx) convention.
            ZZ = density2d(xs, ys, grid=True).T
            levels = sorted(density2d.getContourLevels([0.68, 0.95]))
            axes[j, i].contour(XX, YY, ZZ, levels=levels, colors=color_mcmc, linewidths=3.0, linestyles="dashed")
            axes[j, i].set_xlim(xlims_off[(j, i)])
            axes[j, i].set_ylim(ylims_off[(j, i)])

    # Rotate x-tick labels only: unrotated, they clump into an unreadable
    # smear with many closely-spaced panels. Safe to apply to every panel —
    # swyft.plot_corner already hides ticks on non-edge panels.
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
            Line2D([0], [0], color=color_mcmc, linewidth=4, linestyle="dashed"),
        ]
        # Same adaptive scaling as label_fontsize/fontsize_tick above, but
        # with a lower floor -- small triangle plots (e.g. 2-3 derived
        # params) should get a smaller legend than the N_plot=5 size, just
        # not shrunk all the way down proportionally (unreadable at N_plot=2).
        legend_fontsize = max(12, 20 * N_plot / 5)
        axes[0, N_plot - 1].legend(
            legend_lines, ["Merlin", "Nautilus"], loc="upper right",
            fontsize=legend_fontsize, frameon=False,
        )

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_mode(config, output_path, n_test=1000, n_cols=None):
    """
    Coverage/calibration plot for the best checkpoint in
    STORES.checkpoint_path, evaluated against the last n_test simulations in
    the store. Delegates to coverage.run_coverage_test so this stays
    identical to interactive/notebook use.

    n_cols : optional panel-grid column count override (default: up to 5,
        see run_coverage_test), e.g. to avoid a sparsely-filled last row for
        a specific parameter count.

    Slices the raw store down to the last n_test simulations before
    whitening, and reuses the checkpoint's own trained V_proj
    (recompute_pca forced off) rather than re-deriving PCA — whitening/PCA
    over the full store would cost minutes and tens of GB for no benefit.
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

    run_coverage_test(trainer, network, store_samples, config, output_path, n_test=n_test, n_cols=n_cols)


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
