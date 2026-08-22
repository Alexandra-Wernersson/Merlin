import matplotlib.pyplot as plt
import swyft.plot.plot as _swyft_plot_module
from swyft.plot.plot import _get_HDI_thresholds, get_pdf

_applied = False
_swyft_filled = False


def set_swyft_filled(filled: bool):
    global _swyft_filled
    _swyft_filled = filled


def _plot_2d(
    lrs_coll,
    parname1,
    parname2,
    ax=None,
    bins=100,
    color="k",
    cmap="gray_r",
    linewidth=1,
    smooth=0.0,
    cred_level=[0.68268, 0.95450, 0.99730],
    truth=None,
    smooth_prior=False,
):
    """
    Drop-in replacement for swyft.plot.plot._plot_2d.

    swyft's own version fills via a cmap and ignores the `color` arg on 2D
    panels, so color="black" has no effect. This version restores
    color-driven rendering: unfilled contour lines by default
    (set_swyft_filled(True) to also shade the two inner regions), and a red
    truth marker matching merlin's dashed-red fiducial lines.
    """
    counts, xy = get_pdf(
        lrs_coll,
        [parname1, parname2],
        bins=bins,
        smooth=smooth,
        smooth_prior=smooth_prior,
    )
    xbins = xy[:, 0]
    ybins = xy[:, 1]

    if ax is None:
        ax = plt.gca()

    levels = sorted(_get_HDI_thresholds(counts, cred_level=cred_level))
    if _swyft_filled:
        ax.contourf(
            counts.T,
            extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
            levels=[levels[1], counts.max()],
            colors=[color],
            alpha=0.35,
        )
        ax.contourf(
            counts.T,
            extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
            levels=[levels[2], counts.max()],
            colors=[color],
            alpha=0.35,
        )
    ax.contour(
        counts.T,
        extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
        levels=levels[1:],
        colors=color,
        linewidths=linewidth,
    )
    ax.set_xlim([xbins.min(), xbins.max()])
    ax.set_ylim([ybins.min(), ybins.max()])

    if truth is not None:
        if parname1 in truth.keys():
            ax.axvline(truth[parname1], color="r", lw=1.5, zorder=10, ls=(1, (5, 1)))
        if parname2 in truth.keys():
            ax.axhline(truth[parname2], color="r", lw=1.5, zorder=10, ls=(1, (5, 1)))
        if parname1 in truth.keys() and parname2 in truth.keys():
            ax.scatter([truth[parname1]], [truth[parname2]], c="r", marker=".", s=100)


def _contour1d(z, v, levels, ax=plt, linestyles=None, color=None, **kwargs):
    """
    Drop-in replacement for swyft.plot.plot._contour1d: fills only the two
    inner credible levels, dropping the outermost band. Kept for portability
    to older swyft versions that fill all three.
    """
    y0 = -1.0 * v.max()
    y1 = 5.0 * v.max()
    ax.fill_between(z, y0, y1, where=v >= levels[0], color=color, alpha=0.1)
    ax.fill_between(z, y0, y1, where=v >= levels[1], color=color, alpha=0.1)


def apply_patches():
    """
    Monkeypatch swyft.plot.plot to use the _plot_2d/_contour1d above.

    swyft.plot_corner resolves these as bare names from swyft.plot.plot's
    module globals at call time, so reassigning the module attributes here
    redirects every later call site — no fork/reinstall needed. Call once
    before plotting; afterwards always access via swyft.plot.plot._plot_2d
    (module attribute), never `from swyft.plot.plot import _plot_2d`, which
    would copy the pre-patch reference at import time.
    """
    global _applied
    if _applied:
        return
    _swyft_plot_module._plot_2d = _plot_2d
    _swyft_plot_module._contour1d = _contour1d
    _applied = True
