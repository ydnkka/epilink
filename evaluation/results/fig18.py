"""Boston singleton and largest-cluster shares by clustering approach (fig18)."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from epilink_evaluation.utils import style

from ._boston.common import (
    SCORE_LABELS,
    BostonStudy,
    add_arguments,
)

APPROACHES = (
    ("leiden", "Leiden communities"),
    ("components", "Connected components"),
    ("treecluster", "TreeCluster"),
)
METRICS = (
    ("singleton_percent", "Singleton cases", "#0072B2"),
    ("largest_percent", "Cases in largest cluster", "#D55E00"),
)


def partition_rows(study: BostonStudy) -> pd.DataFrame:
    """Extract every frozen M=0 partition, keeping all-case denominators explicit."""
    rows = []
    n = study.inputs["n_cases"]
    for point in study.selected_points("balanced_M0"):
        definition = point["definition"]
        partition = study.partition(point)
        source = definition["baseline_data_process"]
        if definition["kind"] == "treecluster":
            tree = "Undated" if definition["tree_kind"] == "raw" else "Dated"
            label = f"{tree} [{source[0].upper()}]"
        else:
            label = SCORE_LABELS[definition["score_name"]]
        rows.append(
            {
                "pipeline": point["pipeline"],
                "criterion": point["criterion"],
                "setting_id": point["setting_id"],
                "baseline_setting_id": point["baseline_setting_id"],
                "baseline_data_process": source,
                "kind": definition["kind"],
                "label": label,
                "n_cases": n,
                "n_clusters": int(partition.n_clusters),
                "n_singleton_cases": int(partition.n_singleton_cases),
                "largest_cluster": int(partition.largest_cluster),
                "singleton_percent": 100 * partition.n_singleton_cases / n,
                "largest_percent": 100 * partition.largest_cluster / n,
            }
        )
    return pd.DataFrame(rows)


def create_figure(study: BostonStudy, output, *, fmt: str = "both") -> None:
    rows = partition_rows(study)
    fig, axes = style.new_figure(
        width="double",
        height_in=4.4,
        ncols=len(APPROACHES),
        layout="constrained",
        sharex=True,
    )
    for ax, (kind, title) in zip(axes, APPROACHES):
        subset = rows.loc[rows.kind == kind]
        if subset.empty:
            raise ValueError(f"No frozen Boston partitions for {kind}")
        positions = np.arange(len(subset))
        for index, (metric, _, color) in enumerate(METRICS):
            ax.barh(
                positions + (index - 0.5) * 0.34,
                subset[metric],
                height=0.32,
                color=color,
            )
        ax.set_yticks(positions, subset.label)
        ax.set_ylim(len(subset) - 0.6, -0.6)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_title(title)
        ax.grid(axis="x", color="0.9")
        ax.set_axisbelow(True)
    fig.supxlabel("Fraction of all Boston cases (%)")
    fig.legend(
        handles=[Patch(facecolor=color, label=label) for _, label, color in METRICS],
        title="D/S = deterministic/stochastic synthetic source rules",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=2,
    )
    style.add_panel_labels(axes)
    paths = style.save_figure(
        fig,
        output / "fig18_boston_partition_context",
        width="double",
        save_pdf=fmt in ("pdf", "both"),
        save_png=fmt in ("png", "both"),
    )
    rows.to_csv(output / "fig18_boston_partition_context.csv", index=False)
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
