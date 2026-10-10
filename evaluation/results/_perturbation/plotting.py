"""Common figure controls for perturbation paired percentage-point changes."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes

from .common import Study, scenario_axis

MODEL_COLORS = ("#0072B2", "#D55E00", "#555555", "#009E73")


def symmetric_bound(*arrays: np.ndarray) -> float:
    values = np.concatenate(
        [np.asarray(array, dtype=float).ravel() for array in arrays]
    )
    finite = values[np.isfinite(values)]
    return max(1.0, float(np.abs(finite).max()) * 1.05) if len(finite) else 1.0


def delta_heatmap(
    ax: Axes,
    study: Study,
    values: np.ndarray,
    counts: np.ndarray,
    names: list[str],
    *,
    bound: float,
    improvement_positive: bool,
    labels_on_left: bool = True,
    annotate: bool = True,
):
    if values.shape != counts.shape or values.shape != (
        len(study.variants),
        len(names),
    ):
        raise ValueError("Heatmap dimensions do not match scenarios and models")
    cmap = mpl.colormaps["RdBu" if improvement_positive else "RdBu_r"].copy()
    cmap.set_bad("0.88")
    image = ax.imshow(
        np.ma.masked_invalid(values),
        vmin=-bound,
        vmax=bound,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(range(len(names)), names)
    scenario_axis(ax, study, show_labels=labels_on_left)
    if annotate:
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                text = "—" if not np.isfinite(value) else f"{value:+.1f}"
                if counts[row, col] < len(study.seeds):
                    text += f"\nn={counts[row, col]}"
                ax.text(
                    col,
                    row,
                    text,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white"
                    if np.isfinite(value) and abs(value) > bound * 0.65
                    else "0.15",
                )
    return image
