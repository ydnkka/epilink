"""Shared data handling and panels for one-approach baseline manuscript figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from .common import (
    PROCESS_LABELS,
    PROCESSES,
    SCORE_COLORS,
    SCORE_LABELS,
    SCORES_BY_PROCESS,
    add_arguments,
    load_run,
    output_directory,
    read_json,
    selected_summary,
)
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from epilink_evaluation.utils import style

APPROACHES = {
    "components": "Connected components",
    "leiden_binary": "Leiden (binary edges)",
    "leiden_native": "Leiden (native weights)",
    "treecluster_raw": "TreeCluster (raw tree)",
    "treecluster_dated": "TreeCluster (dated tree)",
}
FIGURE_IDS = {
    "components": "fig03",
    "leiden_binary": "fig04",
    "leiden_native": "fig05",
    "treecluster_raw": "fig07",
    "treecluster_dated": "fig08",
}
TREE_METHODS = ("max_clade", "avg_clade", "single_linkage")
TREE_LABELS = {
    "max_clade": "Max clade",
    "avg_clade": "Avg clade",
    "single_linkage": "Single linkage",
}
TREE_COLORS = {
    "max_clade": "#0072B2",
    "avg_clade": "#009E73",
    "single_linkage": "#D55E00",
}
METRIC_PAIRS = (
    ("M0_recall_mean", "M0_precision_mean"),
    ("Mge3_contamination_mean", "M0_f1_mean"),
)


def approach_matches(definition: dict, approach: str) -> bool:
    if approach == "components":
        return definition["kind"] == "components"
    kind, subtype = approach.split("_", 1)
    if kind == "leiden":
        return definition["kind"] == "leiden" and definition["weight_policy"] == subtype
    return definition["kind"] == "treecluster" and definition["tree_kind"] == subtype


def tradeoff_frontier(
    frame: pd.DataFrame, *, x: str, y: str, minimize_x: bool
) -> pd.DataFrame:
    """Return nondominated coordinates, with each equal-x group represented once.

    For PR, maximize both axes. For F1 against contamination, minimize x and
    maximize y. This is a display envelope; selection still uses the full grid.
    """
    coordinates = (
        frame[[x, y]]
        .dropna()
        .drop_duplicates()
        .sort_values([x, y], ascending=[minimize_x, False])
    )
    best_y = -np.inf
    kept = []
    for row in coordinates.itertuples(index=False, name=None):
        if row[1] > best_y:
            kept.append(row)
            best_y = row[1]
    return pd.DataFrame(kept, columns=[x, y]).sort_values(x)


def development_sweep(
    run: Path, config: dict, approach: str
) -> tuple[pd.DataFrame, dict]:
    definitions = read_json(run / "settings.json")
    columns = [
        "pipeline",
        "setting_id",
        "n_realizations",
        "M0_precision_mean",
        "M0_recall_mean",
        "M0_f1_mean",
        "Mge3_contamination_mean",
    ]
    sweep = pd.read_csv(run / "development/summary.csv", usecols=columns)
    if sweep.duplicated(["pipeline", "setting_id"]).any():
        raise ValueError("Duplicate development setting summaries")
    matched = sweep.setting_id.map(
        lambda key: approach_matches(definitions[key], approach)
    )
    sweep = sweep.loc[matched].copy()
    if (
        sweep.empty
        or not sweep.n_realizations.eq(len(config["splits"]["development"])).all()
    ):
        raise ValueError(f"Missing or incomplete development sweep: {approach}")
    sweep["process"] = sweep.setting_id.map(
        lambda key: definitions[key]["data_process"]
    )
    sweep["method"] = sweep.setting_id.map(lambda key: definitions[key].get("method"))
    sweep["threshold"] = sweep.setting_id.map(lambda key: definitions[key]["threshold"])
    return sweep, definitions


def draw_variant(
    ax_top: Axes,
    ax_bottom: Axes,
    frame: pd.DataFrame,
    color: str,
    *,
    one_dimensional: bool,
    genetic: bool = False,
) -> None:
    if one_dimensional:
        # Traverse the threshold sweep, not a mixture of unrelated parameters.
        group = frame.sort_values("threshold", ascending=genetic)
    else:
        group = frame
    for ax, (x, y) in zip((ax_top, ax_bottom), METRIC_PAIRS):
        valid = group.dropna(subset=[x, y])
        if valid.empty:
            continue
        if one_dimensional:
            ax.plot(valid[x], valid[y], "-", color=color, lw=1.2, alpha=0.85)
        else:
            ax.scatter(valid[x], valid[y], s=8, color=color, alpha=0.23, linewidths=0)
            envelope = tradeoff_frontier(
                valid, x=x, y=y, minimize_x=x == "Mge3_contamination_mean"
            )
            ax.plot(envelope[x], envelope[y], color=color, lw=1.2, alpha=0.85)


def mark_selected(
    axes: tuple[Axes, Axes], frame: pd.DataFrame, point: dict, color: str
) -> None:
    if point["status"] != "selected":
        return
    selected = frame.loc[frame.setting_id == point["setting_id"]]
    if len(selected) != 1:
        raise ValueError(
            f"Frozen setting is absent from development: {point['pipeline']}"
        )
    for ax, (x, y) in zip(axes, METRIC_PAIRS):
        if pd.notna(selected[x].iloc[0]) and pd.notna(selected[y].iloc[0]):
            ax.scatter(
                selected[x].iloc[0],
                selected[y].iloc[0],
                marker="D",
                s=65,
                facecolor=color,
                edgecolor="black",
                linewidth=0.9,
                zorder=5,
            )


def create_approach_figure(
    run: Path, config: dict, output: Path, approach: str, *, fmt: str = "both"
) -> None:
    if approach not in APPROACHES:
        raise ValueError(f"Unknown clustering approach: {approach}")
    _, points = selected_summary(run, config)
    sweep, definitions = development_sweep(run, config, approach)
    tree = approach.startswith("treecluster_")
    fig, axes = style.new_figure(
        width="double",
        height_in=6.2,
        nrows=2,
        ncols=2,
        layout="constrained",
        sharey="row",
    )
    for col, process in enumerate(PROCESSES):
        subset = sweep.loc[sweep.process == process]
        if subset.empty:
            raise ValueError(f"No {approach} settings for {process}")
        if tree:
            pipeline = f"treecluster/{process}/{approach.removeprefix('treecluster_')}"
            variants = [
                (
                    method,
                    pipeline,
                    TREE_COLORS[method],
                    subset.loc[subset.method == method],
                )
                for method in TREE_METHODS
            ]
        else:
            scorers = SCORES_BY_PROCESS[process]
            if approach == "leiden_native":
                scorers = tuple(
                    score for score in scorers if not score.startswith("GD_")
                )
            variants = []
            for score in scorers:
                pipeline = (
                    f"components/{score}"
                    if approach == "components"
                    else f"leiden/{score}/{approach.removeprefix('leiden_')}"
                )
                variants.append(
                    (
                        score,
                        pipeline,
                        SCORE_COLORS[score],
                        subset.loc[subset.pipeline == pipeline],
                    )
                )
        for variant, pipeline, color, group in variants:
            if group.empty or pipeline not in points:
                raise ValueError(
                    f"Missing {approach} development evidence: {pipeline}/{variant}"
                )
            if tree and set(group.method) != {variant}:
                raise ValueError(f"Mixed TreeCluster methods in {pipeline}/{variant}")
            if not tree and any(
                definitions[key]["score_name"] != variant for key in group.setting_id
            ):
                raise ValueError(f"Mixed scorers in {pipeline}")
            draw_variant(
                axes[0, col],
                axes[1, col],
                group,
                color,
                one_dimensional=approach == "components" or tree,
                genetic=tree
                or (approach == "components" and variant.startswith("GD_")),
            )
            point = points[pipeline]
            if not tree or point["definition"].get("method") == variant:
                mark_selected((axes[0, col], axes[1, col]), group, point, color)
        axes[0, col].set_title(f"{PROCESS_LABELS[process]} observed genetics")
        for row in range(2):
            axes[row, col].set(xlim=(0, 1), ylim=(0, 1))
            axes[row, col].grid(axis="y", color="0.9")
        axes[0, col].set_xlabel("M=0 recall")
        axes[1, col].set_xlabel("M≥3 contamination")
        if tree:
            handles = [
                Line2D([0], [0], color=TREE_COLORS[method], label=TREE_LABELS[method])
                for method in TREE_METHODS
            ]
        else:
            handles = [
                Line2D([0], [0], color=SCORE_COLORS[score], label=SCORE_LABELS[score])
                for score in scorers
            ]
        handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                marker="D",
                linestyle="none",
                markerfacecolor="white",
                label="Selected",
            )
        )
        axes[0, col].legend(
            handles=handles, loc="upper right", title="Method" if tree else "Scorer"
        )
    axes[0, 0].set_ylabel("M=0 precision")
    axes[1, 0].set_ylabel("M=0 F1")
    fig.suptitle(APPROACHES[approach], fontweight="bold")
    style.add_panel_labels(axes)
    paths = style.save_figure(
        fig,
        output / f"{FIGURE_IDS[approach]}_{approach}",
        width="double",
        save_pdf=fmt in ("pdf", "both"),
        save_png=fmt in ("png", "both"),
    )
    for path in paths.values():
        print(f"Figure saved to: {path}")


def main(approach: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Plot {APPROACHES[approach]} baseline trade-offs"
    )
    add_arguments(parser, figure=True)
    args = parser.parse_args()
    run, config = load_run(args.run_dir, evaluation=True)
    print(f"Using run: {run}")
    create_approach_figure(
        run, config, output_directory(run, args.output_dir), approach, fmt=args.format
    )
