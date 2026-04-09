from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import string
from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sns

PLOS_WIDTHS_CM = {
    "min": 6.68,
    "single": 8.5,
    "text_column": 13.2,
    "full": 19.05,
}
PLOS_MAX_HEIGHT_CM = 22.23


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
    outdir: str | Path = ".",
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

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}

    if save_pdf:
        pdf_path = output_dir / f"{stem}.pdf"
        fig.savefig(pdf_path, dpi=dpi, transparent=False)
        saved_paths["pdf"] = pdf_path
        print(f"Saved PDF: {pdf_path}")

    if save_eps:
        eps_path = output_dir / f"{stem}.eps"
        fig.savefig(eps_path, format="eps", dpi=dpi, transparent=False)
        saved_paths["eps"] = eps_path
        print(f"Saved EPS: {eps_path}")

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
        print(f"Saved TIFF: {tif_path}")

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
