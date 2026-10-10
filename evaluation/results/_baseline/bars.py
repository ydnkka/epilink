"""Shared held-out bar metrics and frozen-setting labels for manuscript figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from .common import (
    PROCESS_LABELS,
    SCORE_LABELS,
    SCORES_BY_PROCESS,
    setting_label,
)
from .plots import TREE_LABELS

METRICS = (
    ("M0_precision", "Precision", "#0072B2"),
    ("M0_recall", "Recall", "#E69F00"),
    ("M0_f1", "$F_1$", "#009E73"),
    ("Mge3_contamination", "Distant-pair contamination", "#CC79A7"),
)
GRAPH_APPROACHES = ("components", "leiden")
TREE_KINDS = ("raw", "dated")


def graph_variants(approach: str, process: str) -> list[tuple[str, str]]:
    if approach not in GRAPH_APPROACHES:
        raise ValueError(f"Not a graph-clustering approach: {approach}")
    scorers = SCORES_BY_PROCESS[process]
    return [
        (
            SCORE_LABELS[score],
            f"components/{score}" if approach == "components" else f"leiden/{score}",
        )
        for score in scorers
    ]


def treecluster_variant(
    kind: str, process: str, points: dict, config: dict
) -> tuple[str, str]:
    if kind not in TREE_KINDS:
        raise ValueError(f"Not a TreeCluster pipeline: {kind}")
    pipeline = f"treecluster/{process}/{kind}"
    point = points[pipeline]
    if point["status"] == "selected":
        definition = point["definition"]
        if definition["tree_kind"] != kind or definition["data_process"] != process:
            raise ValueError(f"Incorrect frozen TreeCluster definition: {pipeline}")
        detail = (
            f"{TREE_LABELS[definition['method']]}, {setting_label(definition, config)}"
        )
    else:
        detail = "Infeasible"
    return f"{PROCESS_LABELS[process]}\n{detail}", pipeline


def plot_grouped_bars(
    ax: Axes, variants: list[tuple[str, str]], summary: pd.DataFrame, points: dict
) -> None:
    """One group per frozen pipeline; whiskers describe held-out seed variation."""
    positions = np.arange(len(variants))
    width = 0.18
    for metric_index, (metric, _, color) in enumerate(METRICS):
        for x, (_, pipeline) in zip(positions, variants):
            point = points[pipeline]
            if point["status"] != "selected":
                continue
            result = summary.loc[pipeline].to_dict()
            count = int(result[f"{metric}_count"])
            mean = result[f"{metric}_mean"]
            if not count or pd.isna(mean):
                continue
            xpos = x + (metric_index - (len(METRICS) - 1) / 2) * width
            ax.bar(xpos, mean * 100, width=width * 0.95, color=color)
            sd = result[f"{metric}_std"]
            if pd.notna(sd):
                ax.errorbar(
                    xpos,
                    mean * 100,
                    yerr=sd * 100,
                    fmt="none",
                    ecolor="0.25",
                    elinewidth=0.7,
                    capsize=2,
                )
    ax.set_xticks(positions, [label for label, _ in variants])
    ax.set_xlim(-0.55, len(variants) - 0.45)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color="0.9")
    ax.set_axisbelow(True)


def metric_legend() -> list[Patch]:
    return [Patch(facecolor=color, label=label) for _, label, color in METRICS]
