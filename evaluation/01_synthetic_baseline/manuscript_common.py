"""Read and validate saved evidence for standalone baseline manuscript scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent / "outputs" / "baseline"
PROCESSES = ("deterministic", "stochastic")
PROCESS_LABELS = {"deterministic": "Deterministic", "stochastic": "Stochastic"}
SCORES_BY_PROCESS = {
    "deterministic": ("EDD", "ESD", "GD_D", "LOGIT_D"),
    "stochastic": ("EDS", "ESS", "GD_S", "LOGIT_S"),
}
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
SCORE_COLORS = {
    "EDD": "#0072B2",
    "ESD": "#009E73",
    "GD_D": "#555555",
    "LOGIT_D": "#D55E00",
    "EDS": "#0072B2",
    "ESS": "#009E73",
    "GD_S": "#555555",
    "LOGIT_S": "#D55E00",
}


def add_arguments(parser: argparse.ArgumentParser, *, figure: bool = False) -> None:
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Pinned baseline run; default: baseline/current.json",
    )
    parser.add_argument("--output-dir", type=Path, help="Default: <run-dir>/manuscript")
    if figure:
        parser.add_argument("--format", choices=("pdf", "svg", "both"), default="both")


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def load_run(path: Path | None, *, evaluation: bool = False) -> tuple[Path, dict]:
    if path is None:
        path = Path(read_json(ROOT / "current.json")["run_directory"])
    path = path.expanduser().resolve()
    manifest = read_json(path / "manifest.json")
    if evaluation and (
        manifest["status"] != "complete"
        or not (path / "evaluation/selection_used.json").exists()
    ):
        raise ValueError(f"Run has no complete frozen evaluation: {path}")
    return path, manifest["config"]


def output_directory(run: Path, path: Path | None) -> Path:
    return path if path is not None else run / "manuscript"


def frozen_points(run: Path, criterion: str = "balanced_M0") -> dict[str, dict]:
    frozen = read_json(run / "evaluation/selection_used.json")
    points = [
        point for point in frozen["operating_points"] if point["criterion"] == criterion
    ]
    if not points:
        raise ValueError(f"No frozen criterion {criterion!r} in {run}")
    if any(point["rule"]["objective"] != "M0_f1" for point in points):
        raise ValueError("Manuscript displays require a frozen M0_f1 criterion")
    return {point["pipeline"]: point for point in points}


def selected_summary(run: Path, config: dict) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Require one held-out row and all configured seeds per selected pipeline."""
    points = frozen_points(run)
    summary = pd.read_csv(run / "evaluation/operating_summary.csv")
    summary = summary.loc[summary.criterion == "balanced_M0"].copy()
    selected = {
        name: point for name, point in points.items() if point["status"] == "selected"
    }
    if summary.pipeline.duplicated().any() or set(summary.pipeline) != set(selected):
        raise ValueError("Operating summary does not match frozen selected pipelines")
    expected_seeds = set(config["splits"]["evaluation"])
    results = pd.read_csv(
        run / "evaluation/operating_results.csv",
        usecols=["criterion", "pipeline", "seed", "setting_id"],
    )
    results = results.loc[results.criterion == "balanced_M0"]
    for row in summary.itertuples(index=False):
        point = selected[row.pipeline]
        subset = results.loc[results.pipeline == row.pipeline]
        if (
            row.setting_id != point["setting_id"]
            or row.objective != "M0_f1"
            or row.objective_endpoint != "M0"
            or row.n_realizations != len(expected_seeds)
            or set(subset.seed) != expected_seeds
            or len(subset) != len(expected_seeds)
            or set(subset.setting_id) != {row.setting_id}
        ):
            raise ValueError(
                f"Incomplete or unfrozen held-out evidence: {row.pipeline}"
            )
    return summary.set_index("pipeline"), points


def setting_label(definition: dict, config: dict) -> str:
    """Present native cutoffs with their meaning and physical units."""
    kind = definition["kind"]
    threshold = definition.get("threshold")
    if threshold is None:
        cutoff = "empty selection"
    elif kind == "treecluster":
        if definition["tree_kind"] == "raw":
            snps = threshold * config["simulation"]["sequence_length"]
            cutoff = f"{snps:g} SNP"
        else:
            cutoff = f"{threshold:g} d"
    elif definition["score_name"].startswith("GD_"):
        cutoff = f"GD <= {threshold:g} SNP"
    else:
        cutoff = f"score >= {threshold:g}"
    if kind == "leiden":
        return f"{cutoff}; CPM {definition['resolution']:g} ({definition['weight_policy']})"
    return cutoff


def method_label(definition: dict) -> str:
    kind = definition["kind"]
    if kind == "treecluster":
        method = definition["method"].replace("_", " ").title()
        return f"TreeCluster {definition['tree_kind']} ({method})"
    label = {"pairwise": "Pairwise", "components": "Components", "leiden": "Leiden"}[
        kind
    ]
    name = SCORE_LABELS[definition["score_name"]]
    if kind == "leiden":
        return f"{label} {name} ({definition['weight_policy']})"
    return f"{label} {name}"


def format_metric(row: pd.Series, metric: str, *, percent: bool = True) -> str:
    """Format an equal-realization mean (sample SD), preserving undefined values."""
    mean = row[f"{metric}_mean"]
    count = int(row[f"{metric}_count"])
    if not count or pd.isna(mean):
        return "--"
    scale = 100 if percent else 1
    if percent:
        result = f"{mean * scale:.1f}"
    else:
        result = f"{mean:,.0f}"
    sd = row[f"{metric}_std"]
    if pd.notna(sd):
        result += f" ({sd * scale:.1f})" if percent else f" ({sd:,.0f})"
    if count != int(row["n_realizations"]):
        result += f" [n={count}]"
    return result
