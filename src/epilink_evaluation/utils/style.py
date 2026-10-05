"""Publication-grade plotting defaults.

Usage
-----
>>> from analysis.utils import style
>>> fig, ax = style.new_figure("single")

>>> # Temporary override without leaking global state:
>>> with style.theme(context="poster"):
...     fig, ax = style.new_figure("slide")
"""

from __future__ import annotations

import string
import warnings
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "CONTEXTS",
    "DEFAULT_HEIGHT_IN",
    "FIG_WIDTHS_IN",
    "WIDTHS",
    "add_panel_labels",
    "new_figure",
    "save_figure",
    "set_theme",
    "theme",
]

# ---------------------------------------------------------------------------
# Figure size presets (inches). Aligned with journal guidance.
# ---------------------------------------------------------------------------

FIG_WIDTHS_IN = {
    "single": 3.5,  # single column
    "onehalf": 5.2,  # 1.5-column
    "double": 7.2,  # full-page double column
    "slide": 10.0,  # for talks
}

WIDTHS = Literal["single", "onehalf", "double", "slide"]
CONTEXTS = Literal["paper", "talk", "poster", "notebook"]

DEFAULT_HEIGHT_IN = 2.6

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

# Context -> base element scale (relative to "paper").
_CONTEXT_SCALES: dict[str, float] = {
    "paper": 1.0,
    "notebook": 1.0,
    "talk": 1.3,
    "poster": 1.6,
}

# Preferred sans-serif stack, in fallback order. DejaVu Sans is always present
# with matplotlib and acts as the guaranteed fallback.
_SANS_SERIF_STACK: list[str] = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]


def _check_font_availability(stack: list[str]) -> None:
    """Warn if none of the preferred fonts are installed.

    Matplotlib silently falls back to DejaVu Sans when a requested font is
    missing, which can make figures render differently on a laptop vs. CI or a
    journal's print service. This surfaces that situation early.
    """
    available = {f.name for f in fm.fontManager.ttflist}
    if not any(font in available for font in stack[:-1]):  # ignore guaranteed fallback
        warnings.warn(
            f"None of {stack[:-1]} are installed; falling back to "
            f"'{stack[-1]}'. Figures may not match camera-ready output.",
            stacklevel=3,
        )


def _build_rc(scale: float) -> dict[str, Any]:
    """Construct the rcParams dict for a given element scale factor."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": _SANS_SERIF_STACK,
        # --- scaled elements ---
        "font.size": 9 * scale,
        "axes.titlesize": 10 * scale,
        "axes.labelsize": 9 * scale,
        "xtick.labelsize": 8 * scale,
        "ytick.labelsize": 8 * scale,
        "legend.fontsize": 8 * scale,
        "legend.title_fontsize": 8 * scale,
        "axes.linewidth": 0.8 * scale,
        "lines.linewidth": 1.5 * scale,
        "lines.markersize": 5 * scale,
        "patch.linewidth": 0.8 * scale,
        "xtick.major.width": 0.8 * scale,
        "ytick.major.width": 0.8 * scale,
        "xtick.major.size": 3.5 * scale,
        "ytick.major.size": 3.5 * scale,
        "xtick.minor.width": 0.6 * scale,
        "ytick.minor.width": 0.6 * scale,
        "xtick.minor.size": 2.0 * scale,
        "ytick.minor.size": 2.0 * scale,
        # --- unscaled (categorical / boolean) ---
        "mathtext.fontset": "dejavusans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": False,
        "legend.frameon": False,
        # Type 42 = embedded TrueType. Avoids Type 3 fonts, which some
        # publishers/print services reject for camera-ready PDFs.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # keep text as text in SVG, not outlined paths
        "image.cmap": "viridis",
    }


def set_theme(
    context: CONTEXTS = "paper",
    font_scale: float = 1.0,
) -> None:
    """Apply the project matplotlib/seaborn theme for figures.

    Parameters
    ----------
    context:
        Rendering context controlling the base element scale.
    font_scale:
        Additional multiplicative factor applied on top of the context scale.
    """
    _check_font_availability(_SANS_SERIF_STACK)
    scale = _CONTEXT_SCALES[context] * font_scale

    sns.set_theme(
        style="white",
        context="paper",  # neutral baseline; scaling handled explicitly in rc
        font_scale=1.0,
        rc=_build_rc(scale),
    )


@contextmanager
def theme(
    context: CONTEXTS = "paper",
    font_scale: float = 1.0,
) -> Generator[None, None, None]:
    """Temporarily apply the project theme, restoring rcParams on exit.

    Useful for producing multiple renderings (e.g. paper and poster) in a
    single script without leaking global style state.

    Notes
    -----
    ``mpl.rc_context`` restores matplotlib's rcParams on exit, but seaborn's
    module-level palette/style state is not captured. In practice this is fine
    as long as every figure-producing block wraps itself in this manager.

    Examples
    --------
    >>> with theme(context="poster"):
    ...     fig, ax = new_figure("slide")
    ...     ax.plot(x, y)
    ...     save_figure(fig, Path("figure_poster"))
    """
    _check_font_availability(_SANS_SERIF_STACK)
    scale = _CONTEXT_SCALES[context] * font_scale
    rc = _build_rc(scale)
    with mpl.rc_context(rc=rc):
        sns.set_theme(style="white", context="paper", font_scale=1.0, rc=rc)
        yield


def save_figure(
    fig: Figure,
    out_path: Path,
    width: WIDTHS = "single",
    *,
    width_in: float | None = None,
    height_in: float | None = None,
    dpi: int = 600,
    save_pdf: bool = True,
    save_png: bool = False,
    save_tiff: bool = False,
    save_eps: bool = False,
    save_svg: bool = False,
) -> dict[str, Path]:
    """Save a figure to the requested raster/vector formats."""
    if not (300 <= dpi <= 600):
        raise ValueError("dpi should usually be between 300 and 600 for print scripts.")
    if width_in is not None and width_in <= 0:
        raise ValueError("width_in must be positive when provided.")
    if height_in is not None and height_in <= 0:
        raise ValueError("height_in must be positive when provided.")

    current_w, current_h = fig.get_size_inches()

    w = width_in if width_in is not None else FIG_WIDTHS_IN[width]
    if height_in is None:
        aspect = current_h / current_w
        h = w * aspect
    else:
        h = height_in

    fig.set_size_inches(w, h, forward=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}

    if save_pdf:
        pdf_path = out_path.with_suffix(".pdf")
        fig.savefig(pdf_path, dpi=dpi, transparent=False)
        saved_paths["pdf"] = pdf_path

    if save_png:
        png_path = out_path.with_suffix(".png")
        fig.savefig(png_path, dpi=dpi, transparent=False)
        saved_paths["png"] = png_path

    if save_eps:
        eps_path = out_path.with_suffix(".eps")
        fig.savefig(eps_path, format="eps", dpi=dpi, transparent=False)
        saved_paths["eps"] = eps_path

    if save_svg:
        svg_path = out_path.with_suffix(".svg")
        fig.savefig(svg_path, format="svg", dpi=dpi, transparent=False)
        saved_paths["svg"] = svg_path

    if save_tiff:
        tif_path = out_path.with_suffix(".tif")
        fig.savefig(
            tif_path,
            format="tiff",
            dpi=dpi,
            transparent=False,
            pil_kwargs={"compression": "tiff_lzw"},
        )
        saved_paths["tiff"] = tif_path

    plt.close(fig)

    return saved_paths


def add_panel_labels(
    axes: Axes | Sequence[Axes] | np.ndarray,
    *,
    x: float = 0.0,
    y: float = 1.0,
    offset_points: tuple[float, float] = (-6, 6),
    label: str | Sequence[str] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> None:
    """Label axes in row-major order, accepting a single axis or subplot grid.

    Labels sit a fixed number of points above and left of the axes corner,
    independent of subplot height. They use the shared title size and bold
    weight. Explicit labels must match the number of axes; text properties
    can be overridden in ``kwargs``.
    """

    flat_axes = (
        [axes]
        if isinstance(axes, Axes)
        else np.asarray(axes, dtype=object).ravel().tolist()
    )
    if not all(isinstance(ax, Axes) for ax in flat_axes):
        raise TypeError("Panel labels require Matplotlib axes.")
    if label is None:
        labels = []
        for index in range(1, len(flat_axes) + 1):
            letters = ""
            while index:
                index, remainder = divmod(index - 1, 26)
                letters = string.ascii_uppercase[remainder] + letters
            labels.append(letters)
    else:
        labels = [label] if isinstance(label, str) else list(label)
        if len(labels) != len(flat_axes):
            raise ValueError("Provide one panel label per axis.")
    text_kwargs = {
        "fontsize": mpl.rcParams["axes.titlesize"],
        "fontweight": "bold",
        "va": "bottom",
        "ha": "right",
        "clip_on": False,
        "gid": "panel-label",
        **(kwargs or {}),
    }
    for ax, lb in zip(flat_axes, labels):
        ax.annotate(
            lb,
            xy=(x, y),
            xycoords="axes fraction",
            xytext=offset_points,
            textcoords="offset points",
            **text_kwargs,
        )


def new_figure(
    width: WIDTHS = "single",
    height_in: float | None = None,
    width_in: float | None = None,
    nrows: int = 1,
    ncols: int = 1,
    context: CONTEXTS = "paper",
    font_scale: float = 1.0,
    **subplots_kwargs,
) -> tuple[Figure, Any]:
    """Create a figure with one of the paper-width presets.

    Parameters
    ----------
    width:
        Named width preset (ignored when *width_in* is given).
    height_in:
        Explicit figure height in inches.  Defaults to
        ``DEFAULT_HEIGHT_IN * nrows``.
    width_in:
        Explicit figure width in inches.  Overrides the *width* preset,
        useful for large multi-panel figures that need more room than the
        standard ``"double"`` (7.2 in) column allows.
    nrows, ncols:
        Passed to ``plt.subplots``.  The default is a single panel, but this
        is often overridden for multi-panel figures.
    context:
        Seaborn context to set for the figure.  Controls the base scaling of
        fonts and lines.  The default is ``"paper"``; for talks or posters use
        ``"talk"`` or ``"poster"`` for larger text.
    font_scale:
        Additional scaling factor for fonts and lines, applied on top of the
        *context* scaling to fine-tune text size without changing the overall
        context.  Defaults to 1.0 (no additional scaling).
    **subplots_kwargs:
        Additional keyword arguments passed to ``plt.subplots`` (e.g.
        ``sharex=True`` or ``gridspec_kw={"width_ratios": [1, 2]}``).  Do not
        pass *figsize*: the size is determined by *width*/*width_in* and
        *height_in*.
    """
    if "figsize" in subplots_kwargs:
        raise ValueError(
            "Do not pass 'figsize'; use the width/width_in and height_in "
            "parameters instead."
        )

    set_theme(context=context, font_scale=font_scale)
    w = width_in if width_in is not None else FIG_WIDTHS_IN[width]
    h = height_in if height_in is not None else DEFAULT_HEIGHT_IN * nrows
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(w, h), **subplots_kwargs)
    return fig, ax
