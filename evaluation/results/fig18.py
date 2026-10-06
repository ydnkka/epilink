"""Manuscript figure: exposure concentration versus one-cluster recovery in Boston (fig18)."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from epilink_evaluation.utils import style

from ._boston.common import EXPOSURE_LABELS, FOCUS, BostonStudy, add_arguments


def create_figure(study: BostonStudy, output, *, fmt: str = "both") -> None:
    rows = study.focus_rows()
    exposures = study.config["assessment"]["focus_exposures"]
    fig, axes = style.new_figure(
        width="double",
        height_in=4.0,
        ncols=len(exposures),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for index, (ax, exposure) in enumerate(zip(axes, exposures)):
        background = study.exposure_totals[exposure] / study.inputs["n_cases"]
        ax.axhline(background, color="0.45", lw=1, linestyle="--")
        ax.text(
            0.02,
            background + 0.025,
            f"Exposure share of all cases: {100 * background:.1f}%",
            ha="left",
            va="bottom",
            color="0.35",
            fontsize=8,
        )
        for label, pipeline, color in FOCUS:
            subset = rows.loc[(rows.pipeline == pipeline) & (rows.exposure == exposure)]
            if len(subset) != 1:
                raise ValueError(
                    f"Missing frozen exposure cluster: {pipeline}/{exposure}"
                )
            row = subset.iloc[0]
            if pd.isna(row.exposure_recovery):
                continue  # No eligible representative cluster exists for this exposure.
            size = 40 + 120 * row.n_cases / study.inputs["n_cases"]
            ax.scatter(
                row.exposure_recovery,
                row.exposure_fraction,
                s=size,
                color=color,
                marker="s" if pipeline.startswith("treecluster/") else "o",
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
        ax.set(
            title=f"{EXPOSURE_LABELS.get(exposure, exposure)}\n({study.exposure_totals[exposure]} labelled cases)",
            xlabel="Exposure-group recovery",
            xlim=(-0.03, 1.05),
            ylim=(-0.03, 1.05),
        )
        if index == 0:
            ax.set_ylabel("Exposure concentration in the cluster")
        ax.grid(axis="both", color="0.92")
        ax.set_axisbelow(True)
    # fig.suptitle(
    #     "Boston exposure groups at settings selected in simulation", fontweight="bold"
    # )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="s" if pipeline.startswith("treecluster/") else "o",
                linestyle="none",
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=label,
            )
            for label, pipeline, color in FOCUS
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=2,
    )
    style.add_panel_labels(axes)
    paths = style.save_figure(
        fig,
        output / "fig18_boston_exposure_tradeoffs",
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
    study = BostonStudy.load(args.run_dir)
    print(f"Using run: {study.run}")
    create_figure(study, study.output(args.output_dir), fmt=args.format)


if __name__ == "__main__":
    main()
