"""Compact main-results sensitivity figure: AP, cluster F1, and contamination (fig23)."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from epilink_evaluation.utils import style

from ._perturbation.common import (
    FOCUS_CLUSTER,
    FOCUS_PAIR,
    PROCESS_LABELS,
    PROCESSES,
    SCORE_LABELS,
    Study,
    add_arguments,
    cluster_summary,
    pair_summary,
)
from ._perturbation.plotting import delta_heatmap, symmetric_bound
from .fig12 import cluster_labels

METRICS = (
    ("M0_AP", "Pairwise average precision"),
    ("M0_f1", "Cluster F1"),
    ("Mge3_contamination", "Distant-pair contamination"),
)


def create_figure(study: Study, output, *, fmt="both") -> None:
    rank = pair_summary(study)
    clusters = cluster_summary(study, ("M0_f1", "Mge3_contamination"))
    matrices = {}
    records = []
    for metric, _ in METRICS:
        for process in PROCESSES:
            ranking = metric == "M0_AP"
            identifiers = FOCUS_PAIR[process] if ranking else FOCUS_CLUSTER[process]
            values, counts = study.matrix(
                rank if ranking else clusters,
                key="score_name" if ranking else "pipeline",
                metric=metric,
                identifiers=identifiers,
                criterion=None if ranking else "balanced_M0",
            )
            matrices[metric, process] = values, counts
            for i, scenario in enumerate(study.scenario_names):
                for j, identifier in enumerate(identifiers):
                    records.append(
                        {
                            "scenario": scenario,
                            "process": process,
                            "metric": metric,
                            "method": identifier,
                            "mean_change_pp": values[i, j],
                            "defined_realizations": int(counts[i, j]),
                        }
                    )
    fig, axes = style.new_figure(
        width="double",
        height_in=10.2,
        nrows=3,
        ncols=2,
        layout="constrained",
        sharey="row",
    )
    for row, (metric, label) in enumerate(METRICS):
        bound = symmetric_bound(
            *(matrices[metric, process][0] for process in PROCESSES)
        )
        for col, process in enumerate(PROCESSES):
            values, counts = matrices[metric, process]
            labels = (
                [SCORE_LABELS[score] for score in FOCUS_PAIR[process]]
                if metric == "M0_AP"
                else cluster_labels(process)
            )
            image = delta_heatmap(
                axes[row, col],
                study,
                values,
                counts,
                labels,
                bound=bound,
                improvement_positive=metric != "Mge3_contamination",
                labels_on_left=col == 0,
                annotate=False,
            )
            for i, j in zip(*np.where(counts < len(study.seeds))):
                axes[row, col].text(
                    j, i, f"n={counts[i, j]}", ha="center", va="center", fontsize=6
                )
            axes[row, col].tick_params(axis="y", labelsize=7)
            axes[row, col].tick_params(axis="x", labelsize=7)
            axes[row, col].set_title(
                f"{PROCESS_LABELS[process]} genetic observations\n{label}"
                if row == 0
                else label
            )
        fig.colorbar(
            image,
            ax=axes[row, :],
            orientation="horizontal",
            shrink=0.75,
            label=f"Change in {label.lower()} (percentage points)",
        )
    style.add_panel_labels(axes)
    paths = style.save_figure(
        fig,
        output / "fig23_parameter_sensitivity_overview",
        width="double",
        save_pdf=fmt in ("pdf", "both"),
        save_png=fmt in ("png", "both"),
    )
    pd.DataFrame(records).to_csv(
        output / "fig23_parameter_sensitivity_overview.csv", index=False
    )
    for path in paths.values():
        print(f"Figure saved to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, figure=True)
    args = parser.parse_args()
    study = Study.load(args.run_dir)
    create_figure(study, study.output(args.output_dir), fmt=args.format)


if __name__ == "__main__":
    main()
