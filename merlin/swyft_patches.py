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

    The installed swyft's own _plot_2d already drops the outermost (3σ)
    credible band from the *fill* (levels[1:]), but always fills via a
    cmap-based contourf and silently ignores the `color` argument on 2D
    panels — so color="black" (as used throughout merlin's plotting code) has
    no visible effect there. This version restores color-driven rendering:
    unfilled contour lines by default (set_swyft_filled(True) to also shade
    the two inner regions), and a red (not black) truth marker to match
    merlin's own dashed-red fiducial lines.
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
    inner credible levels (drops the outermost band's fill_between). Matches
    what the installed swyft's own _contour1d already does (kept here for
    portability to older swyft versions where it didn't).
    """
    y0 = -1.0 * v.max()
    y1 = 5.0 * v.max()
    ax.fill_between(z, y0, y1, where=v >= levels[0], color=color, alpha=0.1)
    ax.fill_between(z, y0, y1, where=v >= levels[1], color=color, alpha=0.1)


def apply_patches():
    """
    Monkeypatch swyft.plot.plot to use the hand-edited 2-contour-level
    _plot_2d/_contour1d (drops the 3σ credible band; restores color-driven,
    unfilled-by-default 2D contour lines instead of swyft's cmap-filled
    default — see _plot_2d above for why that matters).

    swyft.plot_corner resolves _plot_2d/_contour1d as bare names from
    swyft.plot.plot's own module globals at call time, so reassigning the
    module attributes here redirects every later call site — no fork or
    reinstall of swyft required. Call this once before plotting; afterwards
    always call the patched functions via swyft.plot.plot._plot_2d(...)
    (module-attribute access), never via `from swyft.plot.plot import
    _plot_2d`, since that copies the (possibly pre-patch) reference at import
    time instead of resolving it dynamically.

    Adapted from ~/swyft_projects/Sireeni/src/sireeni/swyft_patches.py.
    """
    global _applied
    if _applied:
        return
    _swyft_plot_module._plot_2d = _plot_2d
    _swyft_plot_module._contour1d = _contour1d
    _applied = True
