"""Manuscript figure: paired F1 and distant-contamination changes at frozen clusters (fig12)."""

from __future__ import annotations

import argparse

from epilink_evaluation.utils import style

from ._perturbation.common import (
    FOCUS_CLUSTER,
    PROCESS_LABELS,
    PROCESSES,
    Study,
    add_arguments,
    cluster_summary,
)
from ._perturbation.plotting import delta_heatmap, symmetric_bound


def cluster_labels(process: str) -> list[str]:
    epilink = "ESD" if process == "deterministic" else "ESS"
    logit = "LGD" if process == "deterministic" else "LGS"
    gd = "GDD" if process == "deterministic" else "GDS"
    return [
        f"{epilink}\nscore\nweights",
        f"{logit}\nscore\nweights",
        f"{gd}\nbinary\nedges",
        "Tree\nundated",
        "Tree\ndated",
    ]


def create_figure(study: Study, output, *, fmt: str = "both") -> None:
    metrics = ("M0_f1", "Mge3_contamination")
    frame = cluster_summary(study, metrics)
    matrices = {
        (metric, process): study.matrix(
            frame,
            key="pipeline",
            metric=metric,
            identifiers=FOCUS_CLUSTER[process],
            criterion="balanced_M0",
        )
        for metric in metrics
        for process in PROCESSES
    }
    bounds = {
        metric: symmetric_bound(
            *(matrices[metric, process][0] for process in PROCESSES)
        )
        for metric in metrics
    }
    fig, axes = style.new_figure(
        width="double",
        height_in=9.1,
        nrows=2,
        ncols=2,
        layout="constrained",
        sharey="row",
    )
    for col, process in enumerate(PROCESSES):
        axes[0, col].set_title(f"{PROCESS_LABELS[process]} genetic observations")
        for row, metric in enumerate(metrics):
            values, counts = matrices[metric, process]
            _ = delta_heatmap(
                axes[row, col],
                study,
                values,
                counts,
                cluster_labels(process),
                bound=bounds[metric],
                improvement_positive=metric == "M0_f1",
                labels_on_left=col == 0,
            )
            if col == 0:
                axes[row, col].set_ylabel("Target-pair F1" if row == 0 else "Distant-pair contamination")
        axes[1, col].set_xlabel("Clustering method at selected settings")
    # fig.suptitle("Sensitivity of frozen M=0 clustering settings", fontweight="bold")
    for row, metric in enumerate(metrics):
        fig.colorbar(
            axes[row, -1].images[0],
            ax=axes[row, :],
            orientation="horizontal",
            shrink=0.72,
            label=("Change in target-pair F1" if row == 0 else "Change in distant-pair contamination")
            + " (percentage points)",
        )
    style.add_panel_labels(axes)
    paths = style.save_figure(
        fig,
        output / "fig12_cluster_sensitivity",
        width="double",
        save_pdf=fmt in ("pdf", "both"),
        save_png=fmt in ("png", "both"),
    )
    for path in paths.values():
        print(f"Figure saved to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, figure=True)
    args = parser.parse_args()
    study = Study.load(args.run_dir)
    print(f"Using run: {study.run}")
    create_figure(study, study.output(args.output_dir), fmt=args.format)


if __name__ == "__main__":
    main()
