"""Matplotlib style shared by all experiments.

The categorical palette is the validated default from the data-viz reference
(colour-vision-deficiency safe in the listed order). Use SERIES[i] in order;
never cycle past the list.
"""
import matplotlib as mpl

PALETTE = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#8a8985",
    "line": "#dcdad4",
    "surface": "#fcfcfb",
}
SERIES = [PALETTE[k] for k in ("blue", "orange", "aqua", "violet", "red", "yellow")]
DIVERGING = "RdBu_r"      # two hues + neutral midpoint, for signed fields
SEQUENTIAL = "Blues"      # one hue, for magnitudes such as |E|^2


def use_style(dpi: int = 130) -> None:
    mpl.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "figure.facecolor": PALETTE["surface"],
        "axes.facecolor": PALETTE["surface"],
        "axes.edgecolor": PALETTE["line"],
        "axes.labelcolor": PALETTE["ink"],
        "axes.titlecolor": PALETTE["ink"],
        "axes.titleweight": "medium",
        "axes.grid": True,
        "grid.color": PALETTE["line"],
        "grid.linewidth": 0.6,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "lines.linewidth": 2.0,
        "xtick.color": PALETTE["ink2"],
        "ytick.color": PALETTE["ink2"],
        "text.color": PALETTE["ink"],
        "font.size": 10,
        "legend.frameon": False,
        "mathtext.fontset": "dejavusans",
        "animation.writer": "ffmpeg",
    })
