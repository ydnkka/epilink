"""Validated saved evidence for perturbation manuscript displays."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .._paths import output_directory as results_output_directory

ROOT = (
    Path(__file__).resolve().parents[2]
    / "02_synthetic_perturbation"
    / "outputs"
    / "perturbation"
)
PROCESSES = ("deterministic", "stochastic")
PROCESS_LABELS = {"deterministic": "Deterministic", "stochastic": "Stochastic"}
SCORE_LABELS = {
    "EDD": "EDD",
    "ESD": "ESD",
    "GD_D": "GDD",
    "LOGIT_D": "LGD",
    "EDS": "EDS",
    "ESS": "ESS",
    "GD_S": "GDS",
    "LOGIT_S": "LGS",
}
FOCUS_PAIR = {
    "deterministic": ("ESD", "LOGIT_D", "GD_D"),
    "stochastic": ("ESS", "LOGIT_S", "GD_S"),
}
FOCUS_CLUSTER = {
    "deterministic": (
        "leiden/ESD/native",
        "leiden/LOGIT_D/native",
        "leiden/GD_D/binary",
        "treecluster/deterministic/raw",
        "treecluster/deterministic/dated",
    ),
    "stochastic": (
        "leiden/ESS/native",
        "leiden/LOGIT_S/native",
        "leiden/GD_S/binary",
        "treecluster/stochastic/raw",
        "treecluster/stochastic/dated",
    ),
}
PARAMETER_LABELS = {
    "incubation.mean": "Incubation mean",
    "incubation.cv": "Incubation variability",
    "testing_delay.mean": "Testing delay mean",
    "testing_delay.cv": "Testing delay variability",
    "substitution_rate": "Substitution rate",
    "relaxation": "Clock relaxation",
}
CRITERIA = {
    "balanced_M0": "M0",
    "balanced_Mle1": "Mle1",
    "balanced_Mle2": "Mle2",
}


def add_arguments(parser: argparse.ArgumentParser, *, figure: bool = False) -> None:
    parser.add_argument(
        "--run-dir", type=Path, help="Pinned run; default: perturbation/current.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Override the run-specific results directory"
    )
    if figure:
        parser.add_argument("--format", choices=("pdf", "png", "both"), default="both")


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


@dataclass
class Study:
    run: Path
    config: dict
    scenarios: list[dict]
    points: dict[tuple[str, str], dict]

    @classmethod
    def load(cls, path: Path | None = None) -> Study:
        if path is None:
            path = Path(read_json(ROOT / "current.json")["run_directory"])
        run = path.expanduser().resolve()
        manifest = read_json(run / "manifest.json")
        config = manifest["config"]
        if manifest["status"] != "complete" or config["smoke_mode"]:
            raise ValueError(
                "Manuscript displays require a complete full perturbation run"
            )
        scenarios = read_json(run / "scenarios.json")
        names = [row["name"] for row in scenarios]
        if names[0] != "baseline" or len(set(names)) != len(names):
            raise ValueError("Expected one unperturbed control and unique scenarios")
        coverage = pd.read_csv(run / "coverage.csv")
        expected = {(name, mode) for name in names for mode in config["modes"]}
        if (
            set(zip(coverage.scenario, coverage["mode"])) != expected
            or len(coverage) != len(expected)
            or not coverage.status.eq("complete").all()
            or not coverage.completed.eq(coverage.expected).all()
        ):
            raise ValueError("Incomplete perturbation scenario/mode coverage")
        if set(config["modes"]) != {"matched", "baseline_fixed"}:
            raise ValueError("Both EpiLink inference modes are required")
        selection = read_json(run / "selection.json")
        reference = read_json(run / "reference.json")
        if selection["run_fingerprint"] != reference["run_fingerprint"]:
            raise ValueError("Frozen selection and reference baseline differ")
        points = {}
        for point in selection["operating_points"]:
            key = (point["criterion"], point["pipeline"])
            if key in points:
                raise ValueError(f"Duplicate frozen point: {key}")
            points[key] = point
        return cls(run, config, scenarios, points)

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(self.config["seeds"])

    @property
    def variants(self) -> list[dict]:
        return self.scenarios[1:]

    @property
    def scenario_names(self) -> list[str]:
        return [row["name"] for row in self.variants]

    @property
    def scenario_labels(self) -> list[str]:
        labels = []
        for scenario in self.variants:
            level = (
                f"{scenario['multiplier']:g}×"
                if scenario["multiplier"] is not None
                else f"{scenario['value']:g}"
            )
            labels.append(
                f"{PARAMETER_LABELS.get(scenario['parameter'], scenario['parameter'])} · {level}"
            )
        return labels

    @property
    def group_boundaries(self) -> list[float]:
        return [
            i - 0.5
            for i in range(1, len(self.variants))
            if self.variants[i]["parameter"] != self.variants[i - 1]["parameter"]
        ]

    def output(self, path: Path | None) -> Path:
        return results_output_directory(self.run, "02_synthetic_perturbation", path)

    def point(self, pipeline: str, criterion: str = "balanced_M0") -> dict:
        point = self.points[criterion, pipeline]
        if point["status"] != "selected":
            raise ValueError(f"No feasible frozen setting: {criterion}/{pipeline}")
        return point

    def table(self, name: str, columns: list[str]) -> pd.DataFrame:
        return pd.read_csv(self.run / name, usecols=columns)

    def matrix(
        self,
        frame: pd.DataFrame,
        *,
        identifiers: tuple[str, ...],
        key: str,
        metric: str,
        mode: str = "matched",
        criterion: str | None = None,
        delta: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scenario × scorer/pipeline values and defined counts; fail on partial joins."""
        names = self.scenario_names if delta else ["baseline"]
        subset = frame.loc[frame["mode"] == mode]
        if criterion is not None:
            subset = subset.loc[subset.criterion == criterion]
        expected_keys = ["scenario", key]
        if subset.duplicated(expected_keys).any():
            raise ValueError(f"Duplicate result rows for {mode}/{criterion}")
        indexed = subset.set_index(expected_keys)
        values = np.full((len(names), len(identifiers)), np.nan)
        counts = np.zeros(values.shape, dtype=int)
        prefix = f"delta_{metric}" if delta else metric
        for i, scenario in enumerate(names):
            for j, name in enumerate(identifiers):
                if (scenario, name) not in indexed.index:
                    raise ValueError(f"Missing {mode} evidence: {scenario}/{name}")
                row = indexed.loc[scenario, name]
                if row.n_realizations != len(self.seeds) or (
                    delta and row.n_controls != len(self.seeds)
                ):
                    raise ValueError(
                        f"Incomplete seed/control coverage: {scenario}/{name}"
                    )
                if (
                    criterion is not None
                    and row.setting_id != self.point(name, criterion)["setting_id"]
                ):
                    raise ValueError(f"Frozen setting mismatch: {criterion}/{name}")
                count = int(row[f"{prefix}_count"])
                if count < 0 or count > len(self.seeds):
                    raise ValueError(f"Invalid defined-value count: {scenario}/{name}")
                counts[i, j] = count
                if count:
                    value = row[f"{prefix}_mean"]
                    if not np.isfinite(value):
                        raise ValueError(
                            f"Undefined mean with defined values: {scenario}/{name}"
                        )
                    values[i, j] = float(value) * 100
        return values, counts


def pair_summary(
    study: Study, metric: str = "M0_AP", *, delta: bool = True
) -> pd.DataFrame:
    prefix = f"delta_{metric}" if delta else metric
    columns = [
        "scenario",
        "mode",
        "score_name",
        "n_realizations",
        f"{prefix}_mean",
        f"{prefix}_std",
        f"{prefix}_count",
    ]
    if delta:
        columns.append("n_controls")
    return study.table(
        "rankings_delta_summary.csv" if delta else "rankings_summary.csv", columns
    )


def cluster_summary(
    study: Study, metrics: tuple[str, ...], *, delta: bool = True
) -> pd.DataFrame:
    columns = [
        "scenario",
        "mode",
        "criterion",
        "pipeline",
        "setting_id",
        "n_realizations",
    ]
    if delta:
        columns.append("n_controls")
    for metric in metrics:
        prefix = f"delta_{metric}" if delta else metric
        columns.extend((f"{prefix}_mean", f"{prefix}_std", f"{prefix}_count"))
    return study.table(
        "results_delta_summary.csv" if delta else "results_summary.csv", columns
    )


def paired_range_summary(study: Study, metric: str, *, ranking: bool) -> pd.DataFrame:
    """Load the mean and *seed-level* min/max of each control-paired delta."""
    prefix = f"delta_{metric}"
    columns = [
        "scenario",
        "mode",
        "n_realizations",
        "n_controls",
        *(f"{prefix}_{stat}" for stat in ("mean", "min", "max", "count")),
    ]
    if ranking:
        columns.append("score_name")
        name = "rankings_delta_summary.csv"
    else:
        columns.extend(("criterion", "pipeline", "setting_id"))
        name = "results_delta_summary.csv"
    return study.table(name, columns)


def paired_ranges(
    study: Study,
    frame: pd.DataFrame,
    *,
    metric: str,
    key: str,
    identifiers: tuple[str, ...],
    criterion: str | None = None,
    mode: str = "matched",
) -> pd.DataFrame:
    """One mean/min/max across seeds per scenario and frozen scorer/pipeline."""
    subset = frame.loc[(frame["mode"] == mode) & frame[key].isin(identifiers)]
    if criterion is not None:
        subset = subset.loc[subset.criterion == criterion]
    if subset.duplicated(["scenario", key]).any():
        raise ValueError("Duplicate paired-delta summary rows")
    indexed = subset.set_index(["scenario", key])
    records = []
    for scenario in study.scenario_names:
        for identifier in identifiers:
            if (scenario, identifier) not in indexed.index:
                raise ValueError(
                    f"Missing paired-delta summary: {scenario}/{identifier}"
                )
            row = indexed.loc[scenario, identifier]
            if row.n_realizations != len(study.seeds) or row.n_controls != len(
                study.seeds
            ):
                raise ValueError(
                    f"Incomplete seed/control coverage: {scenario}/{identifier}"
                )
            if (
                criterion is not None
                and row.setting_id != study.point(identifier, criterion)["setting_id"]
            ):
                raise ValueError(f"Frozen setting mismatch: {criterion}/{identifier}")
            count = int(row[f"delta_{metric}_count"])
            if count < 0 or count > len(study.seeds):
                raise ValueError(
                    f"Invalid defined-value count: {scenario}/{identifier}"
                )
            values = [row[f"delta_{metric}_{stat}"] for stat in ("mean", "min", "max")]
            if count:
                if (
                    not np.isfinite(values).all()
                    or not values[1] - 1e-12 <= values[0] <= values[2] + 1e-12
                ):
                    raise ValueError(
                        f"Inconsistent paired-delta mean/range: {scenario}/{identifier}"
                    )
                values = [float(value) * 100 for value in values]
            else:
                values = [np.nan] * 3
            records.append(
                {
                    "scenario": scenario,
                    "identifier": identifier,
                    "mean_pp": values[0],
                    "min_pp": values[1],
                    "max_pp": values[2],
                    "count": count,
                }
            )
    return pd.DataFrame(records)


def paired_mode_contrast(
    study: Study, *, ranking: bool, metric: str, identifiers: tuple[str, ...]
) -> pd.DataFrame:
    """Subtract EpiLink modes within scenario/seed, after their paired controls."""
    key = "score_name" if ranking else "pipeline"
    keys = ["scenario", "seed", key]
    columns = [*keys, "mode", "control_available", f"delta_{metric}"]
    if not ranking:
        columns.extend(("criterion", "setting_id"))
        keys.extend(("criterion", "setting_id"))
    source = "rankings_deltas.csv" if ranking else "results_deltas.csv"
    frame = study.table(source, columns)
    frame = frame.loc[frame[key].isin(identifiers)].copy()
    if not ranking:
        frame = frame.loc[frame.criterion == "balanced_M0"]
        for pipeline in identifiers:
            selected = study.point(pipeline)
            if set(frame.loc[frame.pipeline == pipeline, "setting_id"]) != {
                selected["setting_id"]
            }:
                raise ValueError(
                    f"Mode evidence differs from frozen setting: {pipeline}"
                )
    if frame.duplicated([*keys, "mode"]).any():
        raise ValueError("Duplicate seed-level mode evidence")
    matched = frame.loc[frame["mode"] == "matched"].set_index(keys)
    fixed = frame.loc[frame["mode"] == "baseline_fixed"].set_index(keys)
    if set(matched.index) != set(fixed.index):
        raise ValueError("Matched/fixed modes lack paired seed evidence")
    expected = {
        (scenario, seed, name)
        for scenario in study.scenario_names
        for seed in study.seeds
        for name in identifiers
    }
    actual = {(index[0], index[1], index[2]) for index in matched.index}
    if actual != expected or len(matched) != len(expected):
        raise ValueError("Missing scenario/seed/ESD-or-ESS mode comparisons")
    fixed = fixed.reindex(matched.index)
    if not matched.control_available.all() or not fixed.control_available.all():
        raise ValueError("Missing unperturbed control for mode comparison")
    difference = (matched[f"delta_{metric}"] - fixed[f"delta_{metric}"]) * 100
    records = difference.rename("difference_pp").reset_index()
    return (
        records.groupby(["scenario", key], sort=False)
        .difference_pp.agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )


def scenario_axis(ax: Axes, study: Study, *, show_labels: bool) -> None:
    ax.set_yticks(range(len(study.variants)))
    ax.set_yticklabels(study.scenario_labels)
    ax.tick_params(axis="y", labelleft=show_labels)
    ax.set_ylim(len(study.variants) - 0.5, -0.5)
    for position in study.group_boundaries:
        ax.axhline(position, color="0.5", lw=0.55, alpha=0.8)
