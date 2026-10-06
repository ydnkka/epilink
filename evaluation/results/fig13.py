"""Manuscript figure: paired AP mean and observed seed range by scorer and process (fig13)."""

from __future__ import annotations

import argparse

from matplotlib.lines import Line2D

from epilink_evaluation.utils import style

from ._perturbation.common import (
    FOCUS_PAIR,
    PROCESS_LABELS,
    PROCESSES,
    Study,
    add_arguments,
    paired_range_summary,
    paired_ranges,
)
from ._perturbation.plotting import MODEL_COLORS, grouped_range_axis, symmetric_bound


def create_figure(study: Study, output, *, fmt: str = "both") -> None:
    frame = paired_range_summary(study, "M0_AP", ranking=True)
    ranges = {
        process: paired_ranges(
            study,
            frame,
            metric="M0_AP",
            key="score_name",
            identifiers=FOCUS_PAIR[process],
        )
        for process in PROCESSES
    }
    bound = symmetric_bound(
        *(data[["min_pp", "max_pp"]].to_numpy() for data in ranges.values())
    )
    fig, axes = style.new_figure(
        width="double",
        height_in=6.6,
        ncols=2,
        layout="constrained",
        sharey=True,
    )
    for col, process in enumerate(PROCESSES):
        ax = axes[col]
        grouped_range_axis(
            ax,
            study,
            ranges[process],
            FOCUS_PAIR[process],
            bound=bound,
            labels_on_left=col == 0,
        )
        ax.set_title(f"{PROCESS_LABELS[process]} genetic observations")
        ax.set_xlabel("Change in average precision (percentage points)")
    # fig.suptitle(
    #     "Pairwise sensitivity: mean and across-realization range", fontweight="bold"
    # )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS[index],
                marker="o",
                linestyle="none",
                label=name,
            )
            for index, name in enumerate(
                (
                    "EpiLink ES (ESD / ESS)",
                    "Logistic (LGD / LGS)",
                    "Genetic distance (GDD / GDS)",
                )
            )
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
    )
    style.add_panel_labels(axes)
    paths = style.save_figure(
        fig,
        output / "fig13_pairwise_ap_ranges",
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
