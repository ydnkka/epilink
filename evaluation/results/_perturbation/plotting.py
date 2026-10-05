"""Common figure controls for perturbation paired percentage-point changes."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes

from .common import Study, scenario_axis

MODEL_COLORS = ("#0072B2", "#D55E00", "#555555", "#009E73", "#CC79A7")


def symmetric_bound(*arrays: np.ndarray) -> float:
    values = np.concatenate([np.asarray(array, dtype=float).ravel() for array in arrays])
    finite = values[np.isfinite(values)]
    return max(1.0, float(np.abs(finite).max()) * 1.05) if len(finite) else 1.0


def delta_heatmap(ax: Axes, study: Study, values: np.ndarray, counts: np.ndarray,
                  names: list[str], *, bound: float, improvement_positive: bool,
                  labels_on_left: bool = True, annotate: bool = True):
    if values.shape != counts.shape or values.shape != (len(study.variants), len(names)):
        raise ValueError("Heatmap dimensions do not match scenarios and models")
    cmap = mpl.colormaps["RdBu" if improvement_positive else "RdBu_r"].copy()
    cmap.set_bad("0.88")
    image = ax.imshow(np.ma.masked_invalid(values), vmin=-bound, vmax=bound,
                      cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(names)), names)
    scenario_axis(ax, study, show_labels=labels_on_left)
    if annotate:
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                text = "—" if not np.isfinite(value) else f"{value:+.1f}"
                if counts[row, col] < len(study.seeds):
                    text += f"\nn={counts[row, col]}"
                ax.text(col, row, text, ha="center", va="center", fontsize=6.5,
                        color="white" if np.isfinite(value) and abs(value) > bound * 0.65 else "0.15")
    return image


def grouped_range_axis(ax: Axes, study: Study, frame, identifiers: tuple[str, ...],
                       *, bound: float, labels_on_left: bool) -> None:
    """Offset seed-level min/max whiskers within each perturbation-scenario row."""
    if len(identifiers) > len(MODEL_COLORS):
        raise ValueError("Add a distinct color for each displayed pipeline")
    for model_index, identifier in enumerate(identifiers):
        group = frame.loc[frame.identifier == identifier].set_index("scenario")
        if set(group.index) != set(study.scenario_names) or group.index.has_duplicates:
            raise ValueError(f"Incomplete paired range evidence: {identifier}")
        group = group.loc[study.scenario_names]
        positions = np.arange(len(group)) + (model_index - (len(identifiers) - 1) / 2) * 0.12
        mean = group.mean_pp.to_numpy(float)
        minimum = group.min_pp.to_numpy(float)
        maximum = group.max_pp.to_numpy(float)
        counts = group["count"].to_numpy(int)
        color = MODEL_COLORS[model_index]
        for full, face in ((True, color), (False, "white")):
            valid = np.isfinite(mean) & (counts == len(study.seeds) if full else
                                         (counts > 0) & (counts < len(study.seeds)))
            if valid.any():
                ax.errorbar(
                    mean[valid], positions[valid],
                    xerr=np.vstack((np.maximum(0, mean[valid] - minimum[valid]),
                                    np.maximum(0, maximum[valid] - mean[valid]))),
                    fmt="o", markersize=4.5, markerfacecolor=face,
                    markeredgecolor=color, color=color, elinewidth=0.9, capsize=2,
                )
        for position, count in zip(positions, counts):
            if count == 0:
                ax.text(bound * 0.96, position, "n=0", color=color, fontsize=6,
                        ha="right", va="center")
    scenario_axis(ax, study, show_labels=labels_on_left)
    ax.axvline(0, color="0.4", linestyle="--", lw=0.8)
    ax.set_xlim(-bound, bound)
    ax.grid(axis="x", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
