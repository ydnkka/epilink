"""Manuscript figure: M=0 pairwise discrimination on development observations (fig02)."""

from __future__ import annotations

import argparse

import pandas as pd
from ._baseline.common import (
    PROCESS_LABELS,
    PROCESSES,
    SCORE_COLORS,
    SCORE_LABELS,
    SCORES_BY_PROCESS,
    add_arguments,
    load_run,
    output_directory,
    selected_summary,
)
from matplotlib.lines import Line2D

from epilink_evaluation.utils import style


def create_figure(run, config, output, *, fmt="both") -> None:
    _, points = selected_summary(run, config)
    seeds = config["splits"]["development"]
    heldout = pd.concat(
        [
            pd.read_csv(run / "evaluation" / f"seed_{seed}" / "pairwise/rankings.csv")
            for seed in config["splits"]["evaluation"]
        ],
        ignore_index=True,
    )
    if heldout.duplicated(["score_name", "seed"]).any():
        raise ValueError("Duplicate held-out ranking evidence")
    expected_scores = {
        score for scores in SCORES_BY_PROCESS.values() for score in scores
    }
    if (
        set(heldout.seed) != set(config["splits"]["evaluation"])
        or set(heldout.score_name) != expected_scores
        or len(heldout) != len(expected_scores) * len(config["splits"]["evaluation"])
    ):
        raise ValueError("Incomplete held-out ranking evidence")
    ap = heldout.groupby("score_name").M0_AP.mean()
    summaries = pd.read_csv(
        run / "development/summary.csv",
        usecols=[
            "pipeline",
            "setting_id",
            "n_realizations",
            "M0_recall_mean",
            "M0_precision_mean",
        ],
    ).set_index(["pipeline", "setting_id"])
    fig, axes = style.new_figure(
        width="double", height_in=3.6, ncols=2, layout="constrained"
    )
    for ax, process in zip(axes, PROCESSES):
        curves = {}
        for seed in seeds:
            curves[seed] = pd.read_parquet(
                run
                / "development"
                / f"seed_{seed}"
                / "pairwise/precision_recall.parquet",
                columns=["score_name", "M0_recall", "M0_precision"],
            )
        for score in SCORES_BY_PROCESS[process]:
            point = points[f"pairwise/{score}"]
            if point["status"] != "selected":
                raise ValueError(f"No frozen M=0 pairwise setting: {score}")
            for seed in seeds:
                curve = curves[seed]
                subset = curve.loc[curve.score_name == score]
                if subset.empty:
                    raise ValueError(f"Missing development PR curve: {score}, {seed}")
                ax.step(
                    subset.M0_recall,
                    subset.M0_precision,
                    where="post",
                    color=SCORE_COLORS[score],
                    alpha=0.5,
                    lw=1,
                )
            row = summaries.loc[(f"pairwise/{score}", point["setting_id"])]
            if row.n_realizations != len(seeds):
                raise ValueError(f"Incomplete development operating point: {score}")
            ax.plot(
                row.M0_recall_mean,
                row.M0_precision_mean,
                "D",
                color=SCORE_COLORS[score],
                markersize=6,
                markeredgecolor="black",
                markeredgewidth=0.6,
                zorder=5,
            )
        handles = [
            Line2D(
                [0],
                [0],
                color=SCORE_COLORS[score],
                marker="D",
                markersize=4,
                label=f"{SCORE_LABELS[score]} ({ap[score]:.3f})",
            )
            for score in SCORES_BY_PROCESS[process]
        ]
        ax.legend(handles=handles, title="Scorer (held-out AP)", loc="upper right")
        ax.set(
            title=f"{PROCESS_LABELS[process]} observed genetics",
            xlabel="M=0 recall",
            ylabel="M=0 precision",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.grid(axis="y", color="0.9")
    style.add_panel_labels(axes)
    paths = style.save_figure(
        fig,
        output / "fig02_pairwise_discrimination",
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
    run, config = load_run(args.run_dir, evaluation=True)
    print(f"Using run: {run}")
    create_figure(run, config, output_directory(run, args.output_dir), fmt=args.format)


if __name__ == "__main__":
    main()
