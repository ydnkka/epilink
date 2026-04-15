"""Shared plotting constants, theme helpers, and PLOS-compliant figure export.

Provides:
- PLOS figure dimension constants (:data:`PLOS_WIDTHS_CM`, :data:`PLOS_MAX_HEIGHT_CM`).
- Canonical model, condition, scenario, and stability display mappings used
  across all evaluation figures.
- :func:`set_plos_theme` for consistent seaborn/matplotlib styling.
- :func:`save_plos_figure` for exporting PDF and TIFF at publication DPI.
- :func:`add_panel_labels` for labelling subplot panels (A, B, C, …).
"""

from __future__ import annotations

import logging
import string
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sns

LOGGER = logging.getLogger(__name__)

PLOS_WIDTHS_CM = {
    "min": 6.68,
    "single": 8.5,
    "text_column": 13.2,
    "full": 19.05,
}
PLOS_MAX_HEIGHT_CM = 22.23

# ─── Shared evaluation constants ──────────────────────────────────────────────

#: Canonical model order used across all figures.
MODELS: list[str] = ["EDD", "EDS", "ESD", "ESS", "LD", "LS"]

#: Raw-key → display-label mapping for condition columns.
CONDITION_LABELS: dict[str, str] = {
    "matched": "Matched",
    "mismatched": "Mismatched",
}
#: Display order matching CONDITION_LABELS insertion order.
CONDITION_ORDER: list[str] = list(CONDITION_LABELS.values())
#: Colour assigned to each condition display name.
CONDITION_COLORS: dict[str, str] = {
    "Matched": "#185FA5",
    "Mismatched": "#993C1D",
}

#: Raw scenario key → short display label (canonical, shared across all figures).
SCENARIO_LABELS: dict[str, str] = {
    "baseline": "Baseline",
    "incubation_mean_0.75": "Inc mean ↓25%",
    "incubation_mean_1.25": "Inc mean ↑25%",
    "incubation_cv_0.75": "Inc CV ↓25%",
    "incubation_cv_1.25": "Inc CV ↑25%",
    "testing_delay_mean_0.75": "Delay mean ↓25%",
    "testing_delay_mean_1.25": "Delay mean ↑25%",
    "testing_delay_cv_0.75": "Delay CV ↓25%",
    "testing_delay_cv_1.25": "Delay CV ↑25%",
    "substitution_rate_0.75": "Clock rate ↓25%",
    "substitution_rate_1.25": "Clock rate ↑25%",
    "relaxation_0.00": "Relax. strict clock",
    "relaxation_1.25": "Relax. ↑25%",
}
#: Non-baseline scenario keys in the canonical display order.
SCENARIO_ORDER: list[str] = [
    "incubation_mean_0.75",
    "incubation_mean_1.25",
    "incubation_cv_0.75",
    "incubation_cv_1.25",
    "testing_delay_mean_0.75",
    "testing_delay_mean_1.25",
    "testing_delay_cv_0.75",
    "testing_delay_cv_1.25",
    "substitution_rate_0.75",
    "substitution_rate_1.25",
    "relaxation_0.00",
    "relaxation_1.25",
]

#: Per-model colour palette, indexed in MODELS order.
MODEL_PALETTE: list[str] = [
    "#E63946",
    "#457B9D",
    "#2A9D8F",
    "#E9C46A",
    "#9B5DE5",
    "#F4A261",
]
#: Per-model line styles, indexed in MODELS order.
MODEL_LINESTYLES: list = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

#: Colours for temporal-stability metrics.
STABILITY_COLORS: dict[str, str] = {
    "forward": "#0072B2",
    "backward": "#D55E00",
    "jaccard": "#CC79A7",
}
#: Marker styles for temporal-stability metrics.
STABILITY_MARKERS: dict[str, str] = {
    "forward": "^",
    "backward": "*",
    "jaccard": "d",
}
#: Legend labels for temporal-stability metrics.
STABILITY_LABELS: dict[str, str] = {
    "forward": "Forward",
    "backward": "Backward",
    "jaccard": "Jaccard",
}


def set_plos_theme(
    context: Literal["paper", "talk", "poster"] = "paper",
    font: str = "Arial",
    font_scale: float = 1.0,
) -> None:
    sns.set_theme(
        style="white",
        context=context,
        font_scale=font_scale,
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": [font, "Arial", "Liberation Sans", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "patch.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.grid": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        },
    )


def cm_to_inch(cm: float) -> float:
    return cm / 2.54


def save_plos_figure(
    fig: plt.Figure,
    stem: str,
    *,
    out_dir: str | Path = ".",
    width_cm: float = PLOS_WIDTHS_CM["text_column"],
    height_cm: float | None = None,
    dpi: int = 600,
    save_pdf: bool = True,
    save_tiff: bool = True,
    save_eps: bool = False,
    close: bool = False,
) -> dict[str, Path]:
    if not (300 <= dpi <= 600):
        raise ValueError("dpi should usually be between 300 and 600 for PLOS figures.")
    if width_cm <= 0:
        raise ValueError("width_cm must be positive.")
    if height_cm is not None and height_cm <= 0:
        raise ValueError("height_cm must be positive when provided.")
    if height_cm is not None and height_cm > PLOS_MAX_HEIGHT_CM:
        raise ValueError(f"height_cm exceeds PLOS maximum of {PLOS_MAX_HEIGHT_CM} cm.")

    current_w, current_h = fig.get_size_inches()
    target_w = cm_to_inch(width_cm)
    if height_cm is None:
        aspect = current_h / current_w
        target_h = target_w * aspect
    else:
        target_h = cm_to_inch(height_cm)

    max_height = cm_to_inch(PLOS_MAX_HEIGHT_CM)
    if target_h > max_height:
        raise ValueError(
            "Figure height exceeds the PLOS maximum after resizing. "
            "Pass an explicit height_cm to keep it within bounds."
        )

    fig.set_size_inches(target_w, target_h, forward=True)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}

    if save_pdf:
        pdf_path = output_dir / f"{stem}.pdf"
        fig.savefig(pdf_path, dpi=dpi, transparent=False)
        saved_paths["pdf"] = pdf_path
        LOGGER.info("figures: saved PDF  %s", pdf_path)

    if save_eps:
        eps_path = output_dir / f"{stem}.eps"
        fig.savefig(eps_path, format="eps", dpi=dpi, transparent=False)
        saved_paths["eps"] = eps_path
        LOGGER.info("figures: saved EPS  %s", eps_path)

    if save_tiff:
        tif_path = output_dir / f"{stem}.tif"
        fig.savefig(
            tif_path,
            format="tiff",
            dpi=dpi,
            transparent=False,
            pil_kwargs={"compression": "tiff_lzw"},
        )
        saved_paths["tiff"] = tif_path
        LOGGER.info("figures: saved TIFF %s", tif_path)

    if close:
        plt.close(fig)

    return saved_paths


def add_panel_labels(
    axes: Sequence[plt.Axes],
    *,
    x: float = 0,
    y: float = 1.1,
    size: float | str = "medium",
) -> None:
    labels = list(string.ascii_uppercase)
    for ax, label in zip(axes, labels):
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=size,
            fontweight="bold",
            va="top",
        )
