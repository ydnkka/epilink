"""Supplementary figures: every selected frozen pipeline at all three horizons (fig17)."""

from __future__ import annotations

import argparse

import matplotlib as mpl
import numpy as np

from epilink_evaluation.utils import style

from ._perturbation.common import (
    CRITERIA, PROCESSES, PROCESS_LABELS, SCORE_LABELS,
    Study, add_arguments, cluster_summary,
)
from ._perturbation.plotting import symmetric_bound
from ._baseline.common import TREE_KIND_LABELS, WEIGHT_LABELS


def pipelines(study: Study, criterion: str, process: str) -> tuple[list[str], list[str]]:
    selected = [point for (name, _), point in study.points.items()
                if name == criterion and point["status"] == "selected"
                and point["definition"]["data_process"] == process]
    ranks = {"pairwise": 0, "components": 1, "leiden": 2, "treecluster": 3}
    score_order = ("EDD", "ESD", "GD_D", "LOGIT_D") if process == "deterministic" else (
        "EDS", "ESS", "GD_S", "LOGIT_S"
    )
    selected.sort(key=lambda point: (
        ranks[point["definition"]["kind"]],
        score_order.index(point["definition"]["score_name"])
        if point["definition"].get("score_name") else 0,
        point["definition"].get("weight_policy", ""),
        {"raw": 0, "dated": 1}.get(point["definition"].get("tree_kind"), 0),
    ))
    identifiers, labels = [], []
    for point in selected:
        definition = point["definition"]
        kind = definition["kind"]
        identifiers.append(point["pipeline"])
        if kind == "treecluster":
            method = definition["method"].replace("_", " ")
            labels.append(f"TreeCluster {TREE_KIND_LABELS[definition['tree_kind']]} ({method})")
        else:
            score = SCORE_LABELS[definition["score_name"]]
            label = {"pairwise": "Pair", "components": "Components", "leiden": "Leiden"}[kind]
            if kind == "leiden":
                label += f" ({WEIGHT_LABELS[definition['weight_policy']]})"
            labels.append(f"{score} {label}")
    if not identifiers:
        raise ValueError(f"No frozen pipeline for {criterion}/{process}")
    return identifiers, labels


def short_scenario_labels(study: Study) -> list[str]:
    names = {
        "incubation.mean": "Incubation mean", "incubation.cv": "Incubation variability",
        "testing_delay.mean": "Testing delay mean", "testing_delay.cv": "Testing delay variability",
        "substitution_rate": "Substitution rate", "relaxation": "Clock relaxation",
    }
    result = []
    for scenario in study.variants:
        level = (f"{scenario['multiplier']:g}×" if scenario["multiplier"] is not None
                 else f"{scenario['value']:g}")
        result.append(f"{names.get(scenario['parameter'], scenario['parameter'])} {level}")
    return result


def create_figure(study: Study, frame, criterion: str, output, *, fmt: str = "both") -> None:
    endpoint = CRITERIA[criterion]
    metrics = (f"{endpoint}_f1", "Mge3_contamination")
    content = {}
    for process in PROCESSES:
        names, labels = pipelines(study, criterion, process)
        for metric in metrics:
            values, counts = study.matrix(
                frame, key="pipeline", identifiers=tuple(names), metric=metric,
                criterion=criterion,
            )
            content[process, metric] = (values.T, counts.T, labels)
    bounds = {metric: symmetric_bound(*(content[process, metric][0] for process in PROCESSES))
              for metric in metrics}
    fig, axes = style.new_figure(
        width="double", width_in=11, height_in=12, nrows=2, ncols=2,
        layout="constrained", sharex="col", sharey="row",
    )
    for row, process in enumerate(PROCESSES):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            values, counts, labels = content[process, metric]
            cmap = mpl.colormaps["RdBu" if col == 0 else "RdBu_r"].copy()
            cmap.set_bad("0.88")
            ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-bounds[metric],
                      vmax=bounds[metric], aspect="auto", interpolation="nearest")
            ax.set_yticks(range(len(labels)), labels)
            ax.tick_params(axis="y", labelleft=col == 0, labelsize=8)
            ax.set_ylim(len(labels) - 0.5, -0.5)
            ax.set_xticks(range(len(study.variants)), short_scenario_labels(study),
                          rotation=70, ha="right", fontsize=7)
            for position in study.group_boundaries:
                ax.axvline(position, color="0.5", lw=0.6)
            if (counts < len(study.seeds)).any():
                for i, j in zip(*np.where(counts < len(study.seeds))):
                    ax.text(j, i, "×" if counts[i, j] == 0 else str(counts[i, j]),
                            ha="center", va="center", fontsize=6)
            if row == 0:
                ax.set_title("F1 change" if col == 0 else "Distant contamination change")
            if col == 0:
                ax.set_ylabel(f"{PROCESS_LABELS[process]} genetic observations")
            if row == 1:
                ax.set_xlabel("Perturbed parameter and level")
    endpoint_label = {"M0": "M=0", "Mle1": "M≤1", "Mle2": "M≤2"}[endpoint]
    fig.suptitle(f"All frozen pipelines: {endpoint_label} paired sensitivity", fontweight="bold")
    for col, metric in enumerate(metrics):
        fig.colorbar(axes[0, col].images[0], ax=axes[:, col],
                     orientation="horizontal", shrink=0.67,
                     label=(f"Δ {endpoint_label} F1" if col == 0 else "Δ M≥3 contamination")
                     + " (percentage points)")
    style.add_panel_labels(axes)
    paths = style.save_figure(
        fig, output / f"fig17_all_pipelines_{criterion}", width="double", width_in=11,
        save_pdf=fmt in ("pdf", "both"), save_png=fmt in ("png", "both"),
    )
    for path in paths.values():
        print(f"Figure saved to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, figure=True)
    parser.add_argument("--criterion", choices=("all", *CRITERIA), default="all")
    args = parser.parse_args()
    study = Study.load(args.run_dir)
    print(f"Using run: {study.run}")
    metrics = tuple(f"{endpoint}_f1" for endpoint in CRITERIA.values()) + (
        "Mge3_contamination",
    )
    frame = cluster_summary(study, metrics)
    names = CRITERIA if args.criterion == "all" else (args.criterion,)
    for criterion in names:
        create_figure(study, frame, criterion, study.output(args.output_dir), fmt=args.format)


if __name__ == "__main__":
    main()
