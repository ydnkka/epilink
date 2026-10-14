"""Boston exposure concentration versus recovery for all frozen M=0 clusterers (fig17)."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from epilink_evaluation.utils import style

from ._baseline.common import SCORE_COLORS
from ._boston.common import (
    EXPOSURE_LABELS,
    BostonStudy,
    add_arguments,
)
from .fig12 import METHODS, add_comparison_legends


def exposure_rows(study: BostonStudy) -> pd.DataFrame:
    """Retain every frozen clustering rule and its representative exposure clusters."""
    markers = {kind: marker for kind, _, marker in METHODS}
    rows = []
    for point in study.selected_points("balanced_M0"):
        definition = point["definition"]
        partition = study.partition(point)
        tree = definition["kind"] == "treecluster"
        method = definition["tree_kind"] if tree else definition["kind"]
        score = None if tree else definition["score_name"]
        for exposure in study.config["assessment"]["focus_exposures"]:
            named = study.representative(point, exposure)
            rows.append(
                {
                    "pipeline": point["pipeline"],
                    "criterion": point["criterion"],
                    "setting_id": point["setting_id"],
                    "baseline_setting_id": point["baseline_setting_id"],
                    "baseline_data_process": definition["baseline_data_process"],
                    "method": method,
                    "score_name": score,
                    "marker": markers[method],
                    "color": "#222222" if tree else SCORE_COLORS[score],
                    "filled": definition["baseline_data_process"] == "stochastic",
                    "exposure": exposure,
                    "exposure_total": study.exposure_totals[exposure],
                    "n_cases": int(named.n_cases) if named is not None else np.nan,
                    "n_exposure": int(named.n_exposure)
                    if named is not None
                    else np.nan,
                    "exposure_fraction": named.exposure_fraction
                    if named is not None
                    else np.nan,
                    "exposure_recovery": named.exposure_recovery
                    if named is not None
                    else np.nan,
                    "n_clusters": int(partition.n_clusters),
                    "n_singleton_cases": int(partition.n_singleton_cases),
                    "largest_cluster": int(partition.largest_cluster),
                }
            )
    return pd.DataFrame(rows)


def create_figure(study: BostonStudy, output, *, fmt: str = "both") -> None:
    rows = exposure_rows(study)
    exposures = study.config["assessment"]["focus_exposures"]
    fig, axes = style.new_figure(
        width="double",
        height_in=4.2,
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
        # Draw larger hollow deterministic markers after filled stochastic ones.
        subset = rows.loc[rows.exposure == exposure].sort_values(
            "filled", ascending=False
        )
        for row in subset.itertuples(index=False):
            if pd.isna(row.exposure_recovery) or pd.isna(row.exposure_fraction):
                continue  # No eligible representative cluster exists for this exposure.
            size = 40 + 120 * row.n_cases / study.inputs["n_cases"]
            ax.scatter(
                row.exposure_recovery,
                row.exposure_fraction,
                s=size * (1 if row.filled else 1.5),
                facecolor=row.color if row.filled else "none",
                marker=row.marker,
                edgecolor=row.color,
                linewidth=1.2,
                zorder=2 if row.method in ("raw", "dated") else 3,
            )
        ax.set(
            title=f"{EXPOSURE_LABELS.get(exposure, exposure)}\n({study.exposure_totals[exposure]} labelled cases)",
            xlabel="Exposure recovery",
            xlim=(-0.03, 1.05),
            ylim=(-0.03, 1.05),
        )
        if index == 0:
            ax.set_ylabel("Exposure concentration")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.grid(axis="both", color="0.92")
        ax.set_axisbelow(True)
    # fig.suptitle(
    #     "Boston exposure groups at settings selected in simulation", fontweight="bold"
    # )
    add_comparison_legends(fig, fill_label="Synthetic source rule")
    style.add_panel_labels(axes)
    paths = style.save_figure(
        fig,
        output / "fig17_boston_exposure_tradeoffs",
        width="double",
        save_pdf=fmt in ("pdf", "both"),
        save_png=fmt in ("png", "both"),
    )
    rows.to_csv(output / "fig17_boston_exposure_tradeoffs.csv", index=False)
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
