"""Manuscript figure: within-seed benefit of updating EpiLink inference parameters (fig15)."""

from __future__ import annotations

import argparse

import numpy as np

from epilink_evaluation.utils import style

from ._perturbation.common import (
    PROCESS_LABELS,
    PROCESSES,
    Study,
    add_arguments,
    paired_mode_contrast,
    scenario_axis,
)
from ._perturbation.plotting import symmetric_bound


def create_figure(study: Study, output, *, fmt: str = "both") -> None:
    processes = [p for p in PROCESSES if study.pipelines(p)]
    identifiers = tuple(name for p in processes for name in study.pipelines(p))
    groups = tuple(
        (
            paired_mode_contrast(
                study, metric="M0_f1", identifiers=identifiers,
                clustering_mode=clustering,
            ).set_index(["scenario", "pipeline"]),
            clustering,
        )
        for clustering in ("baseline", "updated")
    )
    bounds = [
        symmetric_bound(group[["min", "max"]].to_numpy()) for group, _ in groups
    ]
    fig, axes = style.new_figure(
        width="double",
        height_in=8.4,
        nrows=2,
        ncols=len(processes),
        squeeze=False,
        layout="constrained",
        sharey="row",
    )
    for row, (group, clustering) in enumerate(groups):
        for col, process in enumerate(processes):
            ax = axes[row, col]
            for pipeline_index, name in enumerate(study.pipelines(process)):
                for index, scenario in enumerate(study.scenario_names):
                    if (scenario, name) not in group.index:
                        raise ValueError(
                            f"Missing matched/baseline contrast: {scenario}/{name}"
                        )
                    record = group.loc[scenario, name]
                    if record["count"] > 0:
                        mean = float(record["mean"])
                        limits = np.array([
                            [mean - record["min"]], [record["max"] - mean]
                        ])
                        ax.errorbar(
                            mean,
                            index + pipeline_index * 0.12,
                            xerr=limits,
                            fmt="D",
                            color="#0072B2",
                            markersize=4.5,
                            capsize=2,
                            elinewidth=0.9,
                        )
                    if record["count"] != len(study.seeds):
                        ax.text(
                            bounds[row] * 0.96,
                            index,
                            f"n={int(record['count'])}",
                            ha="right",
                            va="center",
                            fontsize=6.5,
                        )
            scenario_axis(ax, study, show_labels=col == 0)
            ax.axvline(0, color="0.45", linestyle="--", lw=0.9)
            ax.grid(axis="x", color="0.9")
            ax.set_xlim(-bounds[row], bounds[row])
            ax.set_xlabel("Matched − baseline Δ $F_1$\n(percentage points)")
            if row == 0:
                ax.set_title(f"{PROCESS_LABELS[process]} genetic observations")
            if col == 0:
                ax.set_ylabel(f"{clustering.capitalize()} resolution")
    style.add_panel_labels(axes)
    paths = style.save_figure(
        fig,
        output / "fig15_epilink_mode_effect",
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
