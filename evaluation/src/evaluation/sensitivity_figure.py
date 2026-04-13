"""Generate sensitivity lollipop figures (AP loss and F1 loss) as PDF and TIF.

Run from any directory:
    python evaluation/src/evaluation/sensitivity_figure.py

Outputs go to evaluation/results/figures/sensitivity_ap_loss.{pdf,tif}
                  and evaluation/results/figures/sensitivity_f1_loss.{pdf,tif}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from plotting import cm_to_inch, save_plos_figure, set_plos_theme, PLOS_WIDTHS_CM  # noqa: E402

FIGURE_OUTPUT_DIR = _HERE.parent.parent / "results" / "figures"
DATA_PATH = _HERE.parent.parent / "results" / "synthetic" / "results.parquet"

MODELS = ["EDD", "EDS", "ESD", "ESS", "LD", "LS"]

SCENARIOS = [
    ("incubation_mean_0.75",    "Inc mean ↓25%"),
    ("incubation_mean_1.25",    "Inc mean ↑25%"),
    ("incubation_cv_0.75",      "Inc CV ↓25%"),
    ("incubation_cv_1.25",      "Inc CV ↑25%"),
    ("testing_delay_mean_0.75", "Delay mean ↓25%"),
    ("testing_delay_mean_1.25", "Delay mean ↑25%"),
    ("testing_delay_cv_0.75",   "Delay CV ↓25%"),
    ("testing_delay_cv_1.25",   "Delay CV ↑25%"),
    ("substitution_rate_0.75",  "Clock rate ↓25%"),
    ("substitution_rate_1.25",  "Clock rate ↑25%"),
    ("relaxation_0.00",         "Relax. strict clock"),
    ("relaxation_1.25",         "Relax. ↑25%"),
]
SCENARIO_KEYS = [sk for sk, _ in SCENARIOS]
SCENARIO_LABELS_LIST = [sl for _, sl in SCENARIOS]
CONDITION_LABELS = {
    "matched": "Matched",
    "generation_varied_inference_fixed": "Mismatched",
}


CLAMP = {"ap_loss": 1.1, "f1_loss": 0.7}
TICKS = {"ap_loss": [-1.0, -0.5, 0.0, 0.5, 1.0], "f1_loss": [-0.6, -0.3, 0.0, 0.3, 0.6]}
COL_M = "#185FA5"
COL_MM = "#993C1D"


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["condition"] = df["condition"].map(CONDITION_LABELS).fillna(df["condition"])
    return df[df["scenario"] != "baseline"].copy()


def _draw_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    model: str,
    show_ylabels: bool,
) -> None:
    clamp = CLAMP[metric]
    n = len(SCENARIOS)
    offset = 0.18

    model_df = df[df["model"] == model]

    for cond, y_offset, col in [
        ("Matched",    -offset, COL_M),
        ("Mismatched", +offset, COL_MM),
    ]:
        cond_df = model_df[model_df["condition"] == cond].set_index("scenario")
        for i, sk in enumerate(SCENARIO_KEYS):
            if sk not in cond_df.index:
                continue
            v = cond_df.loc[sk, metric]
            if pd.isna(v):
                continue
            clipped = abs(v) > clamp
            xv = float(np.clip(v, -clamp, clamp))
            cy = float(i) + y_offset

            ax.plot(
                [0, xv], [cy, cy],
                color=col, linewidth=1.2, alpha=0.7,
                linestyle="--" if clipped else "-",
                solid_capstyle="round",
            )
            ax.plot(xv, cy, "o", color=col, markersize=3.0, zorder=3, markeredgewidth=0)

            if clipped:
                sign = 1 if v > 0 else -1
                ax.annotate(
                    "",
                    xy=(sign * clamp, cy),
                    xytext=(sign * clamp * 0.88, cy),
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.8),
                )

    ax.axvline(0, color="black", linewidth=0.6, alpha=0.25, zorder=0)

    ax.set_xlim(-clamp * 1.08, clamp * 1.08)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(SCENARIO_LABELS_LIST, fontsize="medium")
    if not show_ylabels:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(TICKS[metric])
    ax.set_xticklabels(
        [("0" if t == 0.0 else f"{t:+.1f}") for t in TICKS[metric]], fontsize="small",
    )
    ax.set_title(model, fontsize="small", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", linestyle="-", linewidth=0.5, alpha=0.25, zorder=0)


def make_sensitivity_figure(df: pd.DataFrame, metric: str) -> plt.Figure:
    set_plos_theme()
    nrows, ncols = 2, 3

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(cm_to_inch(PLOS_WIDTHS_CM["text_column"]), cm_to_inch(PLOS_WIDTHS_CM["text_column"] * 1.2)),
        constrained_layout=True,
    )

    for idx, (ax, model) in enumerate(zip(axes.flatten(), MODELS)):
        _draw_panel(ax, df, metric, model, show_ylabels=(idx % ncols == 0))

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_M,  markersize=5, label="Matched"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_MM, markersize=5, label="Mismatched"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        frameon=False,
    )

    metric_label = "Average precision (AP) loss" if metric == "ap_loss" else "F1 score loss"
    fig.supxlabel(f"{metric_label} relative to baseline", x=0.6, ha="center")

    return fig


if __name__ == "__main__":
    df = load_data()
    for name, metric in zip(("S1 Fig", "Fig4"), ("ap_loss", "f1_loss")):
        fig = make_sensitivity_figure(df, metric)
        save_plos_figure(fig, f"{name}", outdir=FIGURE_OUTPUT_DIR)
        plt.close(fig)
    print("Done.")