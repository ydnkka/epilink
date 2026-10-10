"""Validated saved evidence for the four full-graph EpiLink clustering arms."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from epilink_evaluation.scorers import SCORERS
from epilink_evaluation.workflows.perturbation_config import MODES

from .._paths import output_directory as results_output_directory

ROOT = (
    Path(__file__).resolve().parents[2]
    / "02_synthetic_perturbation/outputs/perturbation"
)
PROCESSES = ("deterministic", "stochastic")
PROCESS_LABELS = {"deterministic": "Deterministic", "stochastic": "Stochastic"}
MODE_LABELS = {
    "baseline_inference_baseline_clustering": "Baseline inference\nBaseline resolution",
    "baseline_inference_updated_clustering": "Baseline inference\nUpdated resolution",
    "matched_inference_baseline_clustering": "Matched inference\nBaseline resolution",
    "matched_inference_updated_clustering": "Matched inference\nUpdated resolution",
}
PARAMETER_LABELS = {
    "incubation.mean": "Incubation mean",
    "incubation.cv": "Incubation variability",
    "testing_delay.mean": "Testing delay mean",
    "testing_delay.cv": "Testing delay variability",
    "substitution_rate": "Substitution rate",
    "relaxation": "Clock relaxation",
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
        if (
            manifest["status"] != "complete"
            or config["smoke_mode"]
            or config["schema_version"] != 2
        ):
            raise ValueError(
                "Manuscript displays require a complete full clustering-only perturbation run"
            )
        if set(config["modes"]) != set(MODES):
            raise ValueError("All four inference/clustering modes are required")
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
            or not coverage.completed.eq(
                len(config["scorers"]) * len(config["seeds"])
            ).all()
            or not coverage.completed.eq(coverage.expected).all()
        ):
            raise ValueError("Incomplete perturbation scenario/mode coverage")
        selection = read_json(run / "selection.json")
        reference = read_json(run / "reference.json")
        if selection["run_fingerprint"] != reference["run_fingerprint"]:
            raise ValueError("Frozen selection and reference baseline differ")
        points = {
            (p["criterion"], p["pipeline"]): p for p in selection["operating_points"]
        }
        if len(points) != len(config["scorers"]):
            raise ValueError("Expected one baseline resolution per EpiLink pipeline")
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
        return [
            f"{PARAMETER_LABELS.get(s['parameter'], s['parameter'])} · "
            + (
                f"{s['multiplier']:g}×"
                if s["multiplier"] is not None
                else f"{s['value']:g}"
            )
            for s in self.variants
        ]

    @property
    def group_boundaries(self) -> list[float]:
        return [
            i - 0.5
            for i in range(1, len(self.variants))
            if self.variants[i]["parameter"] != self.variants[i - 1]["parameter"]
        ]

    def pipelines(self, process):
        return tuple(
            f"leiden/{name}"
            for name in self.config["scorers"]
            if SCORERS[name].spec.data_process == process
        )

    @property
    def panels(self):
        """One identifiable panel per scorer, grouped by observed genetics."""
        return tuple(
            (process, pipeline)
            for process in PROCESSES
            for pipeline in self.pipelines(process)
        )

    def output(self, path: Path | None) -> Path:
        return results_output_directory(self.run, "02_synthetic_perturbation", path)

    def point(self, scenario, mode, pipeline):
        selection = read_json(
            self.run / "scenarios" / scenario / mode / "selection.json"
        )
        points = [
            p
            for p in selection["operating_points"]
            if p["pipeline"] == pipeline and p["criterion"] == self.config["criterion"]
        ]
        if len(points) != 1 or points[0]["status"] != "selected":
            raise ValueError(f"No feasible resolution: {scenario}/{mode}/{pipeline}")
        point = points[0]
        if point["definition"].get("graph_mode") != "full":
            raise ValueError("Expected full score-weighted EpiLink graphs")
        if (
            MODES[mode][1] == "baseline"
            and point["setting_id"]
            != self.points[self.config["criterion"], pipeline]["setting_id"]
        ):
            raise ValueError("Baseline clustering differs from the frozen reference")
        return point

    def table(self, name: str, columns: list[str]) -> pd.DataFrame:
        return pd.read_csv(self.run / name, usecols=columns)

    def matrix(self, frame, *, identifiers, metric, mode, delta=True):
        names = self.scenario_names if delta else ["baseline"]
        subset = frame.loc[
            (frame["mode"] == mode) & (frame.criterion == self.config["criterion"])
        ]
        if subset.duplicated(["scenario", "pipeline"]).any():
            raise ValueError("Duplicate clustering result rows")
        indexed = subset.set_index(["scenario", "pipeline"])
        values = np.full((len(names), len(identifiers)), np.nan)
        counts = np.zeros(values.shape, dtype=int)
        prefix = f"delta_{metric}" if delta else metric
        for i, scenario in enumerate(names):
            for j, pipeline in enumerate(identifiers):
                if (scenario, pipeline) not in indexed.index:
                    raise ValueError(f"Missing evidence: {scenario}/{mode}/{pipeline}")
                row = indexed.loc[scenario, pipeline]
                if row.n_realizations != len(self.seeds) or (
                    delta and row.n_controls != len(self.seeds)
                ):
                    raise ValueError("Incomplete seed/control coverage")
                if row.setting_id != self.point(scenario, mode, pipeline)["setting_id"]:
                    raise ValueError("Clustering setting mismatch")
                count = int(row[f"{prefix}_count"])
                if not 0 <= count <= len(self.seeds):
                    raise ValueError("Invalid defined-value count")
                counts[i, j] = count
                if count:
                    value = row[f"{prefix}_mean"]
                    if not np.isfinite(value):
                        raise ValueError("Undefined mean with defined values")
                    values[i, j] = float(value) * 100
        return values, counts


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
        columns.extend(
            f"{prefix}_{stat}" for stat in ("mean", "std", "min", "max", "count")
        )
    return study.table(
        "results_delta_summary.csv" if delta else "results_summary.csv", columns
    )


def paired_mode_contrast(
    study: Study, *, metric: str, identifiers: tuple[str, ...], clustering_mode: str
) -> pd.DataFrame:
    """Matched minus baseline inference within seed; resolution IDs may differ."""
    keys = ["scenario", "seed", "pipeline", "criterion"]
    frame = study.table(
        "results_deltas.csv",
        [*keys, "mode", "setting_id", "control_available", f"delta_{metric}"],
    )
    modes = [
        mode for mode, (_, clustering) in MODES.items() if clustering == clustering_mode
    ]
    frame = frame.loc[
        frame.pipeline.isin(identifiers)
        & frame["mode"].isin(modes)
        & frame.criterion.eq(study.config["criterion"])
    ]
    for row in frame.itertuples():
        if (
            row.setting_id
            != study.point(row.scenario, row.mode, row.pipeline)["setting_id"]
        ):
            raise ValueError("Mode evidence differs from its selected resolution")
    if frame.duplicated([*keys, "mode"]).any():
        raise ValueError("Duplicate seed-level mode evidence")
    matched = frame.loc[
        frame["mode"] == f"matched_inference_{clustering_mode}_clustering"
    ].set_index(keys)
    fixed = frame.loc[
        frame["mode"] == f"baseline_inference_{clustering_mode}_clustering"
    ].set_index(keys)
    if set(matched.index) != set(fixed.index):
        raise ValueError("Inference modes lack paired seed evidence")
    expected = {
        (scenario, seed, name, study.config["criterion"])
        for scenario in study.scenario_names
        for seed in study.seeds
        for name in identifiers
    }
    if set(matched.index) != expected:
        raise ValueError("Missing scenario/seed mode comparisons")
    fixed = fixed.reindex(matched.index)
    if not matched.control_available.all() or not fixed.control_available.all():
        raise ValueError("Missing unperturbed control for mode comparison")
    records = (
        ((matched[f"delta_{metric}"] - fixed[f"delta_{metric}"]) * 100)
        .rename("difference_pp")
        .reset_index()
    )
    return (
        records.groupby(["scenario", "pipeline"], sort=False)
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
