"""Single-panel regret plot for one shared EpiLink Leiden CPM resolution.

Regret compares the best development M=0 F1 at each common resolution (with a
scorer/policy-specific graph threshold) to that pipeline's best setting over
its entire saved development grid. Neither held-out results nor per-seed
threshold optima are used to choose the default.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from manuscript_common import add_arguments, load_run, output_directory, read_json
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from epilink_evaluation.selection.operating import select_operating_points
from epilink_evaluation.utils import style

SCORERS = ("EDD", "EDS", "ESD", "ESS")
POLICIES = ("binary", "native")
PIPELINES = tuple(
    f"leiden/{score}/{policy}" for score in SCORERS for policy in POLICIES
)
SECONDARY_METRICS = (
    "M0_precision",
    "M0_recall",
    "Mge3_contamination",
    "selected_pairs",
    "singleton_fraction",
    "largest_cluster_fraction",
    "Mle1_f1",
    "Mle2_f1",
)
METRIC_COLUMNS = [
    "split",
    "seed",
    "pipeline",
    "setting_id",
    "M0_f1",
    *SECONDARY_METRICS,
]


def load_evidence(run, config):
    definitions = read_json(run / "settings.json")
    identifiers = {
        identifier
        for identifier, definition in definitions.items()
        if definition.get("kind") == "leiden" and definition["pipeline"] in PIPELINES
    }
    frame = pd.read_csv(run / "development/metrics.csv", usecols=METRIC_COLUMNS)
    frame = frame.loc[frame.setting_id.isin(identifiers)].copy()
    seeds = set(config["splits"]["development"])
    if (
        not identifiers
        or frame.empty
        or not frame.split.eq("development").all()
        or frame.duplicated(["pipeline", "setting_id", "seed"]).any()
        or len(frame) != len(identifiers) * len(seeds)
    ):
        raise ValueError("Incomplete or duplicate EpiLink Leiden development sweep")
    for (pipeline, identifier), group in frame.groupby(["pipeline", "setting_id"]):
        if pipeline != definitions[identifier]["pipeline"] or set(group.seed) != seeds:
            raise ValueError(
                f"Incomplete development realization coverage: {pipeline}/{identifier}"
            )
    if set(frame.pipeline) != set(PIPELINES):
        raise ValueError(
            "Expected all four EpiLink variants under both Leiden policies"
        )
    criteria = [
        row for row in config["selection"]["criteria"] if row["name"] == "balanced_M0"
    ]
    if len(criteria) != 1 or criteria[0]["objective"] != "M0_f1":
        raise ValueError(
            "The shared resolution is defined for the frozen balanced_M0 objective"
        )
    return frame, definitions, criteria[0]


def common_resolutions(definitions: dict, pipelines: tuple[str, ...]) -> list[float]:
    grids = [
        {
            float(definition["resolution"])
            for definition in definitions.values()
            if definition.get("kind") == "leiden" and definition["pipeline"] == pipeline
        }
        for pipeline in pipelines
    ]
    if not grids or not all(grids):
        raise ValueError("Missing EpiLink Leiden resolution grid")
    shared = sorted(set.intersection(*grids))
    if not shared:
        raise ValueError("Binary and native EpiLink grids have no common resolution")
    return shared


def chosen_settings(frame, definitions, criterion, seeds):
    return {
        point["pipeline"]: point
        for point in select_operating_points(frame, definitions, [criterion], seeds)
    }


def validate_reference(run, reference: dict, pipelines: tuple[str, ...]) -> None:
    saved = read_json(run / "selection/operating_points.json")
    points = {
        point["pipeline"]: point
        for point in saved["operating_points"]
        if point["criterion"] == "balanced_M0" and point["pipeline"] in pipelines
    }
    if set(points) != set(pipelines):
        raise ValueError(
            "Frozen selection does not contain all EpiLink Leiden pipelines"
        )
    for pipeline in pipelines:
        calculated = reference[pipeline]
        frozen = points[pipeline]
        if (
            calculated["status"] != frozen["status"]
            or calculated["setting_id"] != frozen["setting_id"]
            or not np.isclose(
                calculated["development_objective_mean"],
                frozen["development_objective_mean"],
                atol=1e-12,
            )
        ):
            raise ValueError(
                f"Development optimum differs from frozen selection: {pipeline}"
            )


def regret_tables(frame, definitions, criterion, seeds, pipelines):
    """Select one threshold per pipeline/resolution, then compare with its grid optimum."""
    reference = chosen_settings(frame, definitions, criterion, seeds)
    if set(reference) != set(pipelines) or any(
        point["status"] != "selected" for point in reference.values()
    ):
        raise ValueError(
            "Each EpiLink pipeline requires a feasible development reference"
        )
    resolutions = common_resolutions(definitions, pipelines)
    records = []
    for resolution in resolutions:
        selected_ids = {
            identifier
            for identifier in frame.setting_id.unique()
            if float(definitions[identifier]["resolution"]) == resolution
        }
        candidates = frame.loc[frame.setting_id.isin(selected_ids)]
        selected = chosen_settings(candidates, definitions, criterion, seeds)
        if set(selected) != set(pipelines):
            raise ValueError(f"Missing EpiLink pipelines at resolution {resolution:g}")
        for pipeline in pipelines:
            optimum = reference[pipeline]
            point = selected[pipeline]
            source = optimum["definition"]
            row = {
                "resolution": resolution,
                "pipeline": pipeline,
                "score_name": source["score_name"],
                "data_process": source["data_process"],
                "weight_policy": source["weight_policy"],
                "status": point["status"],
                "reference_setting_id": optimum["setting_id"],
                "reference_resolution": source["resolution"],
                "reference_threshold": source["threshold"],
                "reference_M0_f1_mean": optimum["development_objective_mean"],
                "selected_setting_id": point.get("setting_id"),
            }
            if point["status"] == "selected":
                evidence = frame.loc[
                    (frame.pipeline == pipeline)
                    & (frame.setting_id == point["setting_id"])
                ]
                if set(evidence.seed) != set(seeds) or len(evidence) != len(seeds):
                    raise ValueError(
                        f"Incomplete candidate evidence: {pipeline}/{resolution:g}"
                    )
                regret = 100 * (
                    optimum["development_objective_mean"]
                    - point["development_objective_mean"]
                )
                if regret < -1e-10:
                    raise ValueError(f"Negative regret for {pipeline}/{resolution:g}")
                row.update(
                    selected_threshold=point["definition"]["threshold"],
                    selected_M0_f1_mean=point["development_objective_mean"],
                    selected_M0_f1_sd=point["development_objective_sd"],
                    regret_pp=max(0.0, regret),
                )
                for metric in SECONDARY_METRICS:
                    row[f"{metric}_mean"] = evidence[metric].mean()
                    row[f"{metric}_count"] = evidence[metric].count()
            records.append(row)
    details = pd.DataFrame(records)
    aggregate = []
    for resolution, group in details.groupby("resolution", sort=True):
        values = group.regret_pp.dropna()
        complete = len(values) == len(pipelines)
        aggregate.append(
            {
                "resolution": resolution,
                "n_feasible": len(values),
                "mean_regret_pp": values.mean() if complete else np.nan,
                "q25_regret_pp": values.quantile(0.25) if complete else np.nan,
                "q75_regret_pp": values.quantile(0.75) if complete else np.nan,
                "max_regret_pp": values.max() if complete else np.nan,
            }
        )
    summary = pd.DataFrame(aggregate)
    feasible = summary.dropna(subset=["max_regret_pp", "mean_regret_pp"])
    if feasible.empty:
        raise ValueError("No common resolution is feasible for every EpiLink pipeline")
    default = feasible.sort_values(
        ["max_regret_pp", "mean_regret_pp", "resolution"]
    ).iloc[0]
    summary["selected_default"] = summary.resolution.eq(default.resolution)
    return details, summary, reference


def create_figure(summary: pd.DataFrame, output, *, fmt: str = "both") -> None:
    data = summary.dropna(subset=["mean_regret_pp", "q25_regret_pp", "q75_regret_pp"])
    selected = summary.loc[summary.selected_default].iloc[0]
    fig, ax = style.new_figure(width="double", height_in=3.8, layout="constrained")
    ax.fill_between(
        data.resolution,
        data.q25_regret_pp,
        data.q75_regret_pp,
        color="#8B9BAE",
        alpha=0.32,
    )
    ax.plot(
        data.resolution, data.mean_regret_pp, "o-", color="#182838", lw=2, markersize=4
    )
    ax.axvline(selected.resolution, linestyle="--", color="#B45511", lw=1.6)
    ax.scatter(
        [selected.resolution],
        [selected.mean_regret_pp],
        s=48,
        color="#B45511",
        zorder=4,
    )
    ax.set(
        xlabel="Leiden resolution",
        ylabel="Difference from best M=0 F1 (percentage points)",
        ylim=(0, None),
    )
    ax.set_xticks(summary.resolution, [f"{gamma:g}" for gamma in summary.resolution])
    ax.grid(axis="y", color="0.9")
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Patch(
                facecolor="#8B9BAE",
                alpha=0.32,
                label="Middle half of EpiLink configurations",
            ),
            Line2D([0], [0], color="#182838", marker="o", label="Mean difference"),
            Line2D(
                [0],
                [0],
                color="#B45511",
                linestyle="--",
                marker="o",
                label=f"Chosen default ({selected.resolution:g})",
            ),
        ],
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
    )
    paths = style.save_figure(
        fig,
        output / "epilink_resolution_regret",
        width="double",
        save_pdf=fmt in ("pdf", "both"),
        save_svg=fmt in ("svg", "both"),
    )
    for path in paths.values():
        print(f"Figure saved to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, figure=True)
    args = parser.parse_args()
    run, config = load_run(args.run_dir)
    frame, definitions, criterion = load_evidence(run, config)
    details, summary, reference = regret_tables(
        frame, definitions, criterion, config["splits"]["development"], PIPELINES
    )
    validate_reference(run, reference, PIPELINES)
    output = output_directory(run, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    details_path = output / "epilink_resolution_regret_by_pipeline.csv"
    summary_path = output / "epilink_resolution_regret_summary.csv"
    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)
    create_figure(summary, output, fmt=args.format)
    default = summary.loc[summary.selected_default].iloc[0]
    print(f"Saved calculations: {details_path} and {summary_path}")
    print(
        f"Development-only minimax resolution: {default.resolution:g} "
        f"(worst regret {default.max_regret_pp:.2f} pp; "
        f"mean regret {default.mean_regret_pp:.2f} pp)"
    )


if __name__ == "__main__":
    main()
