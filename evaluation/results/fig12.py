"""Compact main-results comparison of held-out graph and tree clusters (fig12)."""

from __future__ import annotations

import argparse
import math

import pandas as pd
from matplotlib.lines import Line2D

from epilink_evaluation.utils import style

from ._baseline.common import (
    PROCESS_LABELS,
    PROCESSES,
    SCORE_COLORS,
    SCORES_BY_PROCESS,
    add_arguments,
    load_run,
    output_directory,
    selected_summary,
)

METHODS = (
    ("components", "Connected components", "o"),
    ("leiden", "Leiden", "s"),
    ("raw", "TreeCluster: undated", "P"),
    ("dated", "TreeCluster: dated", "X"),
)
SCORER_LEGEND = (
    ("EDD", "EpiLink ED (EDD / EDS)"),
    ("ESD", "EpiLink ES (ESD / ESS)"),
    ("GD_D", "Genetic distance (GDD / GDS)"),
    ("LOGIT_D", "Logistic regression (LGD / LGS)"),
)


def cluster_rows(run, config) -> pd.DataFrame:
    """Extract every frozen M=0 cluster comparison, without selecting on held-out data."""
    summary, points = selected_summary(run, config)
    rows = []
    for process in PROCESSES:
        for kind, _, marker in METHODS:
            scores = (None,) if kind in ("raw", "dated") else SCORES_BY_PROCESS[process]
            for score in scores:
                if kind in ("raw", "dated"):
                    pipeline = f"treecluster/{process}/{kind}"
                    color = "#222222"
                else:
                    pipeline = (
                        f"components/{score}"
                        if kind == "components"
                        else f"leiden/{score}"
                    )
                    color = SCORE_COLORS[score]
                point = points[pipeline]
                if point["status"] != "selected":
                    raise ValueError(f"Missing selected cluster method: {pipeline}")
                if point["definition"]["data_process"] != process:
                    raise ValueError(f"Incorrect genetic observations: {pipeline}")
                result = summary.loc[pipeline]
                rows.append(
                    {
                        "process": process,
                        "pipeline": pipeline,
                        "setting_id": point["setting_id"],
                        "method": kind,
                        "score_name": score,
                        "marker": marker,
                        "color": color,
                        "f1_percent": 100 * result.M0_f1_mean,
                        "f1_sd_percent": 100 * result.M0_f1_std,
                        "f1_count": int(result.M0_f1_count),
                        "distant_percent": 100 * result.Mge3_contamination_mean,
                        "distant_sd_percent": 100 * result.Mge3_contamination_std,
                        "distant_count": int(result.Mge3_contamination_count),
                    }
                )
    return pd.DataFrame(rows)


def create_figure(run, config, output, *, fmt="both") -> None:
    rows = cluster_rows(run, config)
    valid = rows.dropna(subset=["f1_percent", "distant_percent"])
    if valid.empty:
        raise ValueError("No defined held-out cluster metrics")
    x_max = min(105, max(10, 5 + 10 * math.ceil(valid.distant_percent.max() / 10)))
    y_max = min(105, max(10, 5 + 10 * math.ceil(valid.f1_percent.max() / 10)))
    fig, axes = style.new_figure(
        width="double",
        height_in=4.2,
        ncols=2,
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    for ax, process in zip(axes, PROCESSES):
        for row in rows.loc[rows.process == process].itertuples(index=False):
            if pd.isna(row.f1_percent) or pd.isna(row.distant_percent):
                continue
            # Open Leiden markers leave coincident connected-component points visible.
            ax.scatter(
                row.distant_percent,
                row.f1_percent,
                marker=row.marker,
                s=78 if row.method == "leiden" else 48,
                facecolor="none" if row.method == "leiden" else row.color,
                edgecolor=row.color,
                linewidth=1.2,
                zorder=3,
            )
        ax.set(
            title=f"{PROCESS_LABELS[process]} genetic observations",
            xlabel="Distant-pair contamination (%)",
            xlim=(-2, x_max),
            ylim=(0, y_max),
        )
        ax.grid(color="0.92")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Target-pair $F_1$ (%)")
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=marker,
                color="0.2",
                linestyle="none",
                markerfacecolor="none",
                label=label,
            )
            for _, label, marker in METHODS
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=3,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color=SCORE_COLORS[score], lw=2, label=label)
            for score, label in SCORER_LEGEND
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )
    style.add_panel_labels(axes)
    paths = style.save_figure(
        fig,
        output / "fig12_cluster_recovery_contamination",
        width="double",
        save_pdf=fmt in ("pdf", "both"),
        save_png=fmt in ("png", "both"),
    )
    rows.to_csv(output / "fig12_cluster_recovery_contamination.csv", index=False)
    for path in paths.values():
        print(f"Figure saved to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, figure=True)
    args = parser.parse_args()
    run, config = load_run(args.run_dir, evaluation=True)
    create_figure(run, config, output_directory(run, args.output_dir), fmt=args.format)


if __name__ == "__main__":
    main()
