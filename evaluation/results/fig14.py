"""Manuscript figure: clustering F1 and contamination means with seed ranges (fig14)."""

from __future__ import annotations

import argparse

from matplotlib.lines import Line2D

from epilink_evaluation.utils import style

from ._perturbation.common import (
    FOCUS_CLUSTER, PROCESSES, PROCESS_LABELS, Study, add_arguments,
    paired_range_summary, paired_ranges,
)
from ._perturbation.plotting import MODEL_COLORS, grouped_range_axis, symmetric_bound


def create_figure(study: Study, output, *, fmt: str = "both") -> None:
    metrics = ("M0_f1", "Mge3_contamination")
    ranges = {}
    for metric in metrics:
        frame = paired_range_summary(study, metric, ranking=False)
        for process in PROCESSES:
            ranges[metric, process] = paired_ranges(
                study, frame, metric=metric, key="pipeline",
                identifiers=FOCUS_CLUSTER[process], criterion="balanced_M0",
            )
    bounds = {
        metric: symmetric_bound(*(ranges[metric, process][["min_pp", "max_pp"]].to_numpy()
                                  for process in PROCESSES))
        for metric in metrics
    }
    fig, axes = style.new_figure(
        width="double", height_in=9.3, nrows=2, ncols=2,
        layout="constrained", sharex="row", sharey="row",
    )
    for row, metric in enumerate(metrics):
        for col, process in enumerate(PROCESSES):
            ax = axes[row, col]
            grouped_range_axis(ax, study, ranges[metric, process], FOCUS_CLUSTER[process],
                               bound=bounds[metric], labels_on_left=col == 0)
            if row == 0:
                ax.set_title(f"{PROCESS_LABELS[process]} genetic observations")
            ax.set_xlabel(("Change in target-pair F1" if row == 0 else
                           "Change in distant-pair contamination") + "\n(percentage points)")
            if col == 0:
                ax.set_ylabel("Target-pair F1" if row == 0 else "Distant-pair contamination")
    fig.suptitle("Selected cluster settings: mean and across-realization range", fontweight="bold")
    names = ("ESD / ESS (score weights)", "LGD / LGS (score weights)", "GDD / GDS (binary edges)",
             "TreeCluster undated", "TreeCluster dated")
    fig.legend(
        handles=[Line2D([0], [0], color=MODEL_COLORS[index], marker="o",
                        linestyle="none", label=name)
                 for index, name in enumerate(names)],
        loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3,
    )
    style.add_panel_labels(axes)
    paths = style.save_figure(
        fig, output / "fig14_cluster_f1_contamination_ranges", width="double",
        save_pdf=fmt in ("pdf", "both"), save_png=fmt in ("png", "both"),
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
