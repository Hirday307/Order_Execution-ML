"""Matplotlib style for the notebooks: one validated palette, thin marks, quiet chrome.

Categorical slots are used in fixed order (never cycled past the list). Every
chart in the notebooks is printed next to its table, so no value is read from
color alone.
"""
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Diverging: blue and red poles, neutral gray midpoint
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#2a78d6", "#f0efec",
                                                                "#e34948", "#a8302f"])


def apply_style() -> None:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 9.5, "text.color": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.labelcolor": INK_2, "axes.labelsize": 9,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 8, "lines.markeredgewidth": 2, "lines.markeredgecolor": SURFACE,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK_2,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "patch.edgecolor": SURFACE, "patch.linewidth": 2,   # 2px surface gap between touching bars
    })


def small_multiples(n: int, ncols: int = 3, panel=(3.4, 2.3), sharex=True, sharey=False):
    """A grid of n panels; unused cells are removed. Returns (fig, list_of_axes)."""
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows),
                             sharex=sharex, sharey=sharey, squeeze=False, layout="constrained")
    flat = list(axes.ravel())
    for ax in flat[n:]:
        ax.remove()
    for i, ax in enumerate(flat[:n]):         # bottom panel of each column shows the x ticks
        if i + ncols >= n:
            ax.xaxis.set_tick_params(labelbottom=True)
    return fig, flat[:n]


def zero_line(ax, axis: str = "y") -> None:
    (ax.axhline if axis == "y" else ax.axvline)(0, color=AXIS, linewidth=1, zorder=1)


def ci_dots(ax, labels, means, lo, hi, color=SERIES[0]) -> None:
    """Horizontal point estimates with 95% intervals, against a zero reference."""
    y = np.arange(len(labels))[::-1]
    ax.hlines(y, lo, hi, color=color, linewidth=2, zorder=2)
    ax.plot(means, y, "o", color=color, zorder=3)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    zero_line(ax, "x")
