"""Figures for the evaluation report.

A trajectory plot is not decoration. An RMSE number collapses the whole
trajectory into one scalar and hides *where* and *how* it went wrong: a
mirrored axis, an alignment that latched onto the wrong section, a single
catastrophic relocalization failure among hundreds of good ones, and uniform
mediocrity can all produce the same RMSE. The plot separates them instantly.

Palette is fixed and colourblind-safe (checked for protan/deutan/tritan
separation, not chosen by eye). Ground truth is deliberately a recessive
neutral: it is the reference the eye compares against, not a competing series.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

C_GT = "#444444"      # ground truth: neutral, recessive, dashed
C_EST = "#1F6FEB"     # estimate
C_ERR = "#E8590C"     # error / residual
C_ALT = "#7048E8"     # a third series when needed
C_GRID = "#D9D9D9"
C_INK = "#1A1A1A"
C_MUTED = "#6B6B6B"


def _style(ax, title=None, xlabel=None, ylabel=None):
    """Recessive axes and grid, so the data is the darkest thing on the plot."""
    ax.grid(True, color=C_GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    if title:
        ax.set_title(title, color=C_INK, fontsize=11, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=C_MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=C_MUTED, fontsize=9)
    return ax


def plot_trajectory_xy(estimate, groundtruth, ax=None, title="Trajectory (top-down)"):
    """Top-down XY overlay of an **already-aligned** estimate against ground truth.

    Pass the aligned estimate, not the raw one -- plotting the raw estimate
    against ground truth shows the difference between two arbitrary coordinate
    frames, which is not information about your system.

    Equal aspect is enforced. Without it matplotlib stretches one axis to fill
    the figure and a trajectory that drifted 5 cm sideways over 2 m looks like
    a violent swerve.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))

    g, e = groundtruth.positions, estimate.positions
    ax.plot(g[:, 0], g[:, 1], color=C_GT, lw=2.0, ls="--", label="ground truth", zorder=2)
    ax.plot(e[:, 0], e[:, 1], color=C_EST, lw=2.0, label="estimate", zorder=3)
    ax.scatter(*g[0, :2], s=42, color=C_GT, zorder=4, edgecolor="white", linewidth=1.2)
    ax.annotate("start", g[0, :2], textcoords="offset points", xytext=(8, 6),
                fontsize=9, color=C_MUTED)

    _style(ax, title, "x (m)", "y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(frameon=False, fontsize=9, labelcolor=C_INK, loc="best")
    return ax


def plot_error_over_time(result, ax=None, title="Absolute trajectory error over time"):
    """ATE per frame against time, with the RMSE drawn as a reference line.

    The shape carries the diagnosis. A slow upward ramp is drift. A flat line
    with isolated spikes is intermittent relocalization failure -- a completely
    different problem with a completely different fix, and one that RMSE alone
    would report as "moderately bad throughout".
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 3.2))

    t = result.timestamps - result.timestamps[0]
    rmse_cm = result.translation["rmse"] * 100
    ax.plot(t, result.errors * 100, color=C_ERR, lw=1.4, zorder=3)
    ax.axhline(rmse_cm, color=C_INK, lw=1.2, ls=":", zorder=4)
    ax.fill_between(t, 0, result.errors * 100, color=C_ERR, alpha=0.12, zorder=2)
    _style(ax, title, "time (s)", "error (cm)")
    ax.set_ylim(bottom=0)
    # Direct-labelled at the right edge instead of a legend box: one series
    # needs no legend, and a box in the corner collided with the curve.
    ax.annotate(f"RMSE {rmse_cm:.2f} cm", xy=(t[-1], rmse_cm),
                xytext=(6, 3), textcoords="offset points",
                fontsize=9, color=C_INK, va="bottom", annotation_clip=False)
    ax.set_xlim(t[0], t[-1] * 1.02)
    return ax


def plot_rpe_histogram(result, axes=None, title="Relative pose error"):
    """Distribution of per-interval translation and rotation error.

    Two panels rather than two y-axes on one: metres and degrees are different
    quantities and overlaying them on a shared axis invites a comparison that
    means nothing.

    A long right tail here is the signature of a few bad intervals rather than
    uniformly noisy tracking, and the median-vs-RMSE gap quantifies it.
    """
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    for ax, data, colour, unit, label in (
        (axes[0], result.errors_trans * 100, C_ERR, "cm", "translation"),
        (axes[1], result.errors_rot_deg, C_ALT, "deg", "rotation"),
    ):
        ax.hist(data, bins=40, color=colour, alpha=0.85, edgecolor="white", linewidth=0.5)
        med = float(np.median(data))
        ax.axvline(med, color=C_INK, lw=1.2, ls=":")
        ax.annotate(f"median {med:.2f} {unit}", (med, ax.get_ylim()[1] * 0.92),
                    textcoords="offset points", xytext=(6, 0), fontsize=9, color=C_INK)
        _style(ax, f"{title} — {label}", f"error per {result.delta:g}"
               f"{'s' if result.delta_unit == 's' else ' frames'} ({unit})", "count")
    return axes


def evaluation_figure(estimate_aligned, groundtruth, ate_result, rpe_result=None,
                      path=None, suptitle=None):
    """The report figure: trajectory, error over time, and RPE distributions.

    Saves to ``path`` if given and returns the figure either way.
    """
    import matplotlib
    if path is not None:
        matplotlib.use("Agg")           # no display needed when writing to disk
    import matplotlib.pyplot as plt

    nrows = 2 if rpe_result is None else 3
    fig = plt.figure(figsize=(11, 3.6 * nrows), facecolor="white")
    gs = fig.add_gridspec(nrows, 2, height_ratios=[1.5] + [1] * (nrows - 1),
                          hspace=0.55, wspace=0.25)

    plot_trajectory_xy(estimate_aligned, groundtruth, ax=fig.add_subplot(gs[0, 0]))
    ax_txt = fig.add_subplot(gs[0, 1])
    ax_txt.axis("off")
    text = ate_result.summary()
    if rpe_result is not None:
        text += "\n\n" + rpe_result.summary()
    text += (f"\n\npath length   {groundtruth.path_length():.2f} m"
             f"\nduration      {groundtruth.duration:.1f} s")
    ax_txt.text(0, 1, text, family="monospace", fontsize=9.5, va="top", color=C_INK)

    plot_error_over_time(ate_result, ax=fig.add_subplot(gs[1, :]))
    if rpe_result is not None:
        plot_rpe_histogram(rpe_result,
                           axes=[fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])])

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, color=C_INK, x=0.02, ha="left")

    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig
