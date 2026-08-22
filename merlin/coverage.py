import swyft

from .inference import _derived_prior_samples
from .params import DERIVED_PARAMS, PARAM_LABELS
from .priors import resolve_inference_params
from .simulator import build_simulator


def run_coverage_test(trainer, network, store_samples, config, output_path,
                       n_prior=10_000, n_test=1000):
    """
    Run swyft's built-in coverage/calibration test and save a zz-plot (nominal
    vs. empirical credibility) per inferred parameter to output_path.

    This checks whether the network's posterior credible intervals are
    correctly calibrated — e.g. the true parameter should fall inside the
    reported 68% credible region ~68% of the time across many held-out
    simulations — a standard SBI diagnostic, since over/under-confident
    posteriors are a known failure mode of neural ratio/posterior estimators.

    The test set is the last `n_test` simulations in store_samples (a cheap
    stand-in for a proper held-out split, matching swyft's test_coverage
    convention); the prior samples used for comparison are drawn from the same
    Simulator/PriorSampler as training (see inference.infer for why — this
    keeps the coverage test's prior consistent with the training prior for
    every parameter, including the Gaussian-distributed photo-z dimensions).
    If any inferred parameter is derived (sigma8/Omega_m/S8), the prior
    samples instead reuse the training store's own ("z", "derived") pairs
    (see inference._derived_prior_samples) rather than a fresh draw — there
    is no closed-form/independently-samplable prior for a deterministic,
    correlated pushforward quantity like sigma8.
    """
    import matplotlib.pyplot as plt

    param_names, param_indices = resolve_inference_params(config)

    if any(name in DERIVED_PARAMS for name in param_names):
        prior_samples = _derived_prior_samples(config, n_prior)
    else:
        sim = build_simulator(config)
        prior_samples = swyft.Samples(z=sim.sample_z(shape=(n_prior,)))

    n_test     = min(n_test, len(store_samples["noise"]))
    test_store = store_samples[-n_test:]

    network.eval()
    coverage_samples = trainer.test_coverage(network, test_store, prior_samples)

    # Wrap into multiple rows of up to 5 panels each — a single row gets
    # unreadably long/squeezed past ~5 parameters (e.g. the 13 shear-calibration
    # m_i's). n_cols stays at N_plot (a single row) for N_plot <= 5, matching
    # the previous one-row-only layout exactly.
    N_plot  = len(param_indices)
    n_cols  = min(N_plot, 5)
    n_rows  = -(-N_plot // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 4.0 * n_rows), sharex=True, sharey=True)
    axes = axes.reshape(n_rows, n_cols) if N_plot > 1 else [[axes]]

    for i in range(N_plot):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]
        swyft.plot_zz(coverage_samples, f"z[{i}]", ax=ax, bins=50)
        # swyft.plot_zz hardcodes fontsize=14 for the in-panel credibility
        # percentage annotations (swyft/plot/mass.py) — bump those post-hoc,
        # no fontsize parameter is exposed to pass through.
        for txt in ax.texts:
            txt.set_fontsize(14)
        ax.set_title(PARAM_LABELS.get(param_names[i], param_names[i]), fontsize=21, pad=8)
        ax.tick_params(axis="both", which="major", labelsize=13)
        # Bottom-most VISIBLE panel in this column (not just "last row" —
        # the last row may be partially filled, e.g. 13 params over 5 cols
        # leaves the last row's final 2 slots empty) gets the x-label;
        # leftmost column gets the y-label — mirrors label_outer()'s intent
        # but geometry-aware of the ragged last row, which label_outer()
        # itself isn't (it would wrongly blank tick labels on the row above
        # a column's empty slot).
        if i + n_cols >= N_plot:
            ax.set_xlabel("Nominal credibility ($z_p$)", fontsize=16)
        else:
            # swyft.plot_zz already set its own (small-font) default xlabel
            # on this ax -- tick_params only hides the tick VALUE numbers,
            # not that label text, so clear it explicitly too.
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", labelbottom=False)
        if col == 0:
            ax.set_ylabel("Empirical coverage ($z_p$)", fontsize=16)
        else:
            # Same issue as above, for swyft.plot_zz's default ylabel.
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", labelleft=False)

    # Hide unused trailing panels in a ragged last row.
    for i in range(N_plot, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
