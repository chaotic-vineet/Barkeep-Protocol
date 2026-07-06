# barkeep_style.py
# The Barkeep Protocol — Matplotlib Style
# Dark, warm, atmospheric. The light above a bar counter at 11pm.

import matplotlib.pyplot as plt
import matplotlib as mpl

# === Core Palette ===
COLORS = {
    "bg":           "#0C0C0E",
    "bg_panel":     "#141210",
    "grid":         "#1E1C17",
    "border":       "#2A2820",
    "text_dim":     "#4A4438",
    "text_mid":     "#8C8070",
    "text_main":    "#D4C9B8",
    "text_bright":  "#F0E8D8",

    # Accent palette — use for data series
    "amber":        "#E8A838",
    "green":        "#1D9E75",
    "violet":       "#7C71D8",
    "red":          "#DC4E3A",
    "orange":       "#D85A30",
    "teal":         "#2E9E7A",
    "gold":         "#BA7517",
    "blue":         "#5B9BD5",
}

# Ordered accent cycle for multi-series plots
ACCENT_CYCLE = [
    COLORS["amber"],
    COLORS["green"],
    COLORS["violet"],
    COLORS["red"],
    COLORS["teal"],
    COLORS["orange"],
    COLORS["blue"],
    COLORS["gold"],
]


def apply_style():
    """Apply the Barkeep Protocol style globally."""
    mpl.rcParams.update({
        # Figure
        "figure.facecolor":     COLORS["bg"],
        "figure.edgecolor":     COLORS["bg"],
        "figure.figsize":       (10, 6),
        "figure.dpi":           100,

        # Axes
        "axes.facecolor":       COLORS["bg"],
        "axes.edgecolor":       COLORS["border"],
        "axes.labelcolor":      COLORS["text_main"],
        "axes.titlecolor":      COLORS["text_bright"],
        "axes.grid":            True,
        "axes.linewidth":       0.6,
        "axes.titlesize":       14,
        "axes.titleweight":     "normal",
        "axes.labelsize":       11,
        "axes.prop_cycle":      mpl.cycler(color=ACCENT_CYCLE),

        # Grid
        "grid.color":           COLORS["grid"],
        "grid.linewidth":       0.4,
        "grid.alpha":           1.0,

        # Ticks
        "xtick.color":          COLORS["text_dim"],
        "ytick.color":          COLORS["text_dim"],
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,
        "xtick.direction":      "out",
        "ytick.direction":      "out",

        # Legend
        "legend.facecolor":     COLORS["bg_panel"],
        "legend.edgecolor":     COLORS["border"],
        "legend.fontsize":      10,
        "legend.labelcolor":    COLORS["text_main"],

        # Text
        "text.color":           COLORS["text_main"],
        "font.family":          "serif",
        "font.serif":           ["Georgia", "DejaVu Serif", "Times New Roman"],
        "font.size":            11,

        # Lines
        "lines.linewidth":      1.5,
        "lines.antialiased":    True,

        # Savefig
        "savefig.facecolor":    COLORS["bg"],
        "savefig.edgecolor":    COLORS["bg"],
        "savefig.bbox":         "tight",
        "savefig.dpi":          150,
    })


def barkeep_fig(nrows=1, ncols=1, figsize=None, **kwargs):
    """Create a styled figure. Shortcut for common use."""
    apply_style()
    if figsize is None:
        w = 10 if ncols == 1 else 6 * ncols
        h = 5 if nrows == 1 else 4.5 * nrows
        figsize = (w, h)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes


def annotate_region(ax, x_start, x_end, label=None, color=None):
    """Highlight a vertical region on an axis."""
    c = color or COLORS["amber"]
    ax.axvspan(x_start, x_end, alpha=0.1, color=c)
    if label:
        mid = (x_start + x_end) / 2
        ymin, ymax = ax.get_ylim()
        ax.text(mid, ymax * 0.92, label, ha="center", fontsize=9,
                color=c, alpha=0.8)


def annotate_point(ax, x, y, label, color=None):
    """Annotate a specific point with an arrow."""
    c = color or COLORS["amber"]
    ax.annotate(label, xy=(x, y), fontsize=9, color=c,
                xytext=(15, 15), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=c, lw=0.8))
