"""Supplementary Boston figures: exposure summaries for every frozen clusterer (fig19)."""

from __future__ import annotations

import argparse

import matplotlib as mpl
import numpy as np

from epilink_evaluation.utils import style

from ._boston.common import (
    CRITERIA,
    EXPOSURE_LABELS,
    BostonStudy,
    add_arguments,
    method_label,
)


def all_rows(study: BostonStudy, criterion: str):
    exposures = study.config["assessment"]["focus_exposures"]
    points = study.selected_points(criterion)
    rank = {"components": 0, "leiden": 1, "treecluster": 2}
    points.sort(
        key=lambda point: (rank[point["definition"]["kind"]], point["pipeline"])
    )
    labels = []
    concentration = np.full((len(points), len(exposures)), np.nan)
    recovery = np.full_like(concentration, np.nan)
    for i, point in enumerate(points):
        label = method_label(point)
        definition = point["definition"]
        if definition["kind"] == "treecluster":
            source = definition["baseline_data_process"][0].upper()
            cutoff = definition["threshold"]
            units = "subs/site" if definition["tree_kind"] == "raw" else "d"
            tree = "undated" if definition["tree_kind"] == "raw" else "dated"
            label = f"TreeCluster {tree} [{source} source; {cutoff:g} {units}]"
        labels.append(label)
        for j, exposure in enumerate(exposures):
            named = study.representative(point, exposure)
            if named is not None:
                concentration[i, j] = 100 * named.exposure_fraction
                recovery[i, j] = 100 * named.exposure_recovery
    return labels, concentration, recovery


def create_figure(
    study: BostonStudy, criterion: str, output, *, fmt: str = "both"
) -> None:
    labels, concentration, recovery = all_rows(study, criterion)
    fig, axes = style.new_figure(
        width="double",
        width_in=8.8,
        height_in=9.4,
        ncols=2,
        layout="constrained",
        sharey=True,
    )
    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad("0.88")
    for ax, matrix, title, show_labels in zip(
        axes,
        (concentration, recovery),
        ("Concentration in one cluster", "Recovery in one cluster"),
        (True, False),
    ):
        image = ax.imshow(
            np.ma.masked_invalid(matrix),
            vmin=0,
            vmax=100,
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xticks(
            range(len(study.config["assessment"]["focus_exposures"])),
            [
                EXPOSURE_LABELS.get(name, name)
                for name in study.config["assessment"]["focus_exposures"]
            ],
        )
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.tick_params(axis="y", labelleft=show_labels)
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(
                col,
                row,
                "—" if not np.isfinite(value) else f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if np.isfinite(value) and value < 35 else "0.12",
            )
    fig.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        shrink=0.6,
        label="Percentage of cases",
    )
    # endpoint = {"balanced_M0": "M=0", "balanced_Mle1": "M≤1", "balanced_Mle2": "M≤2"}[
    #     criterion
    # ]
    # fig.suptitle(
    #     f"Boston exposure summaries: settings selected for {endpoint} in simulation",
    #     fontweight="bold",
    # )
    style.add_panel_labels(axes)
    paths = style.save_figure(
        fig,
        output / f"fig19_boston_all_exposures_{criterion}",
        width="double",
        width_in=8.8,
        save_pdf=fmt in ("pdf", "both"),
        save_png=fmt in ("png", "both"),
    )
    for path in paths.values():
        print(f"Figure saved to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, figure=True)
    parser.add_argument("--criterion", choices=("all", *CRITERIA), default="all")
    args = parser.parse_args()
    study = BostonStudy.load(args.run_dir)
    print(f"Using run: {study.run}")
    criteria = CRITERIA if args.criterion == "all" else (args.criterion,)
    for criterion in criteria:
        create_figure(study, criterion, study.output(args.output_dir), fmt=args.format)


if __name__ == "__main__":
    main()
