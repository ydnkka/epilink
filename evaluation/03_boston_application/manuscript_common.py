"""Validated frozen-transfer evidence for Boston manuscript displays."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from epilink_evaluation.provenance import read_json

ROOT = Path(__file__).resolve().parent / "outputs" / "boston"
FOCUS = (
    ("ESD native", "leiden/ESD/native", "#0072B2"),
    ("LGD native", "leiden/LOGIT_D/native", "#D55E00"),
    ("GDD binary", "leiden/GD_D/binary", "#555555"),
    ("TreeCluster raw", "treecluster/empirical/deterministic/raw", "#009E73"),
    ("TreeCluster dated", "treecluster/empirical/deterministic/dated", "#CC79A7"),
)
CRITERIA = ("balanced_M0", "balanced_Mle1", "balanced_Mle2")
SCORE_LABELS = {
    "EDD": "EDD",
    "EDS": "EDS",
    "ESD": "ESD",
    "ESS": "ESS",
    "GD_D": "GDD",
    "GD_S": "GDS",
    "LOGIT_D": "LGD",
    "LOGIT_S": "LGS",
}


def add_arguments(parser: argparse.ArgumentParser, *, figure: bool = False) -> None:
    parser.add_argument(
        "--run-dir", type=Path, help="Pinned Boston run; default: boston/current.json"
    )
    parser.add_argument("--output-dir", type=Path, help="Default: <run-dir>/manuscript")
    if figure:
        parser.add_argument("--format", choices=("pdf", "svg", "both"), default="both")


def status_complete(path: Path) -> int:
    status = read_json(path)
    if (
        status["status"] != "complete"
        or status["configured"] <= 0
        or status["configured"] != status["completed"]
        or status.get("errors")
    ):
        raise ValueError(f"Incomplete Boston partition coverage: {path}")
    return status["completed"]


@dataclass
class BostonStudy:
    run: Path
    inputs: dict
    config: dict
    selected: dict[tuple[str, str], dict]
    settings: dict
    summary: pd.DataFrame
    named: pd.DataFrame
    agreement: pd.DataFrame
    exposure_totals: dict[str, int]

    @classmethod
    def load(cls, path: Path | None = None) -> BostonStudy:
        if path is None:
            path = Path(read_json(ROOT / "current.json")["run_directory"])
        run = path.expanduser().resolve()
        manifest = read_json(run / "manifest.json")
        n_graph = status_complete(run / "clusters/status.json")
        n_tree = status_complete(run / "trees/status.json")
        if manifest["status"] != "complete":
            report = (run / "report.md").read_text()
            if (
                manifest["status"] != "running"
                or manifest.get("requested_stage") != "explore"
                or "Status: complete (requested stage: all)." not in report
            ):
                raise ValueError("Boston run has no completed frozen-transfer report")
            warnings.warn(
                "Boston manifest still records a subsequent running 'explore' stage; "
                "using independently complete graph/tree checkpoints and the earlier "
                "completed 'all' assessment. Pin a finalized run for publication.",
                stacklevel=2,
            )
        reference = read_json(run / "reference.json")
        selection = read_json(run / "selection.json")
        if (
            selection["reference_selection_fingerprint"]
            != reference["selection_fingerprint"]
        ):
            raise ValueError(
                "Boston selection does not match its pinned baseline reference"
            )
        settings = read_json(run / "settings.json")
        inputs = read_json(run / "inputs.json")
        n_cases = inputs["n_cases"]
        if (
            n_cases < 2
            or inputs["n_all_pairs"] != n_cases * (n_cases - 1) // 2
            or not 0 < inputs["n_observed_pairs"] <= inputs["n_all_pairs"]
        ):
            raise ValueError("Invalid Boston case or censored-pair universe")
        cases = pd.read_parquet(inputs["cases_path"], columns=["case_id", "Exposure"])
        if len(cases) != n_cases or cases.case_id.duplicated().any():
            raise ValueError("Prepared Boston cases differ from the frozen input count")
        totals = cases.Exposure.value_counts().to_dict()
        selected = {}
        for point in selection["operating_points"]:
            if (
                point["status"] != "selected"
                or point["definition"]["kind"] == "pairwise"
            ):
                continue
            key = (point["criterion"], point["pipeline"])
            if key in selected or point["setting_id"] not in settings:
                raise ValueError(f"Duplicate or unknown Boston frozen setting: {key}")
            if settings[point["setting_id"]] != point["definition"]:
                raise ValueError(
                    f"Boston definition differs from frozen setting: {key}"
                )
            selected[key] = point
        summary = pd.read_csv(run / "assessment/summary.csv", dtype={"setting_id": str})
        named = pd.read_csv(
            run / "assessment/named_cluster_overlaps.csv", dtype={"setting_id": str}
        )
        agreement = pd.read_csv(
            run / "assessment/tree_agreement.csv",
            dtype={
                "setting_id": str,
                "tree_setting_id": str,
            },
        )
        ids = {point["setting_id"] for point in selected.values()}
        graph_ids = {
            identifier
            for identifier in ids
            if settings[identifier]["kind"] != "treecluster"
        }
        tree_ids = ids - graph_ids
        if (
            len(graph_ids) != n_graph
            or len(tree_ids) != n_tree
            or summary.setting_id.duplicated().any()
            or set(summary.setting_id) != ids
            or not summary.n_observed_pairs.eq(inputs["n_observed_pairs"]).all()
            or not summary.n_all_pairs.eq(inputs["n_all_pairs"]).all()
            or not np.allclose(
                summary.candidate_coverage,
                inputs["n_observed_pairs"] / inputs["n_all_pairs"],
            )
        ):
            raise ValueError("Boston assessment does not cover all frozen partitions")
        if (
            agreement.duplicated(["setting_id", "tree_setting_id"]).any()
            or len(agreement) != n_graph * n_tree
            or set(agreement.setting_id) != graph_ids
            or set(agreement.tree_setting_id) != tree_ids
            or not agreement.n_cases.eq(n_cases).all()
        ):
            raise ValueError("Incomplete Boston graph/tree agreement coverage")
        if (
            named.duplicated(["setting_id", "exposure"]).any()
            or not set(named.setting_id) <= ids
        ):
            raise ValueError(
                "Duplicate or unknown Boston representative exposure clusters"
            )
        if not set(manifest["config"]["assessment"]["focus_exposures"]) <= set(totals):
            raise ValueError("Boston focus exposure is absent from the prepared cases")
        for row in named.itertuples(index=False):
            if (
                row.exposure_total != totals.get(row.exposure, 0)
                or row.pipeline != settings[row.setting_id]["pipeline"]
            ):
                raise ValueError(
                    "Representative exposure counts disagree with Boston inputs"
                )
        return cls(
            run,
            inputs,
            manifest["config"],
            selected,
            settings,
            summary,
            named,
            agreement,
            totals,
        )

    def output(self, path: Path | None) -> Path:
        return path if path is not None else self.run / "manuscript"

    def point(self, pipeline: str, criterion: str = "balanced_M0") -> dict:
        try:
            return self.selected[criterion, pipeline]
        except KeyError as exc:
            raise ValueError(
                f"No frozen Boston setting: {criterion}/{pipeline}"
            ) from exc

    def selected_points(self, criterion: str = "balanced_M0") -> list[dict]:
        if criterion not in CRITERIA:
            raise ValueError(f"Unknown frozen criterion: {criterion}")
        return [
            point for (name, _), point in self.selected.items() if name == criterion
        ]

    def partition(self, point: dict) -> pd.Series:
        subset = self.summary.loc[self.summary.setting_id == point["setting_id"]]
        if (
            len(subset) != 1
            or subset.pipeline.iloc[0] != point["pipeline"]
            or subset.baseline_setting_id.iloc[0] != point["baseline_setting_id"]
        ):
            raise ValueError(
                f"Assessment summary differs from frozen setting: {point['pipeline']}"
            )
        return subset.iloc[0]

    def representative(self, point: dict, exposure: str) -> pd.Series | None:
        subset = self.named.loc[
            (self.named.setting_id == point["setting_id"])
            & (self.named.exposure == exposure)
        ]
        if subset.empty:
            return None  # No eligible cluster contains the exposure; never imply zero recovery.
        if (
            len(subset) != 1
            or subset.pipeline.iloc[0] != point["pipeline"]
            or subset.baseline_setting_id.iloc[0] != point["baseline_setting_id"]
        ):
            raise ValueError(
                f"Ambiguous exposure cluster: {point['pipeline']}/{exposure}"
            )
        row = subset.iloc[0]
        if (
            not 0 < row.n_exposure <= row.n_cases <= self.inputs["n_cases"]
            or row.exposure_total != self.exposure_totals[exposure]
            or not np.isclose(row.exposure_fraction, row.n_exposure / row.n_cases)
            or not np.isclose(
                row.exposure_recovery, row.n_exposure / row.exposure_total
            )
        ):
            raise ValueError(
                f"Inconsistent exposure denominators: {point['pipeline']}/{exposure}"
            )
        return row

    def focus_rows(self, criterion: str = "balanced_M0") -> pd.DataFrame:
        rows = []
        for label, pipeline, color in FOCUS:
            point = self.point(pipeline, criterion)
            partition = self.partition(point)
            for exposure in self.config["assessment"]["focus_exposures"]:
                named = self.representative(point, exposure)
                rows.append(
                    {
                        "pipeline": pipeline,
                        "label": label,
                        "color": color,
                        "setting_id": point["setting_id"],
                        "baseline_setting_id": point["baseline_setting_id"],
                        "exposure": exposure,
                        "exposure_total": self.exposure_totals[exposure],
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


def method_label(point: dict) -> str:
    definition = point["definition"]
    kind = definition["kind"]
    if kind == "treecluster":
        name = definition["method"].replace("_", " ").title()
        return f"TreeCluster {definition['tree_kind']} ({name})"
    score = SCORE_LABELS[definition["score_name"]]
    label = (
        "Components"
        if kind == "components"
        else f"Leiden {definition['weight_policy']}"
    )
    return f"{score} {label}"
