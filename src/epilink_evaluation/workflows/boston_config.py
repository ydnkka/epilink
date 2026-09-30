"""Configuration loading for Boston empirical studies."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..provenance import implementation_signature
from .boston_scoring import BASELINE_SCORES, validate_scorers


def load_study_config(
    argv=None, config_path=None, baseline_run=None, output=None
):
    if argv is not None:
        import argparse

        parser = argparse.ArgumentParser(description="Boston empirical clustering")
        parser.add_argument("--config", type=Path, default=config_path)
        parser.add_argument("--baseline-run", type=Path, default=baseline_run)
        parser.add_argument("--output", type=Path, default=output)
        args = parser.parse_args(argv)
        config_path = args.config
        baseline_run = args.baseline_run
        output = args.output

    config_path = Path(config_path or "boston_application/config.yaml").resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Expected Boston schema_version: 1")

    config["config_path"] = str(config_path)
    config["baseline_run"] = str(
        (config_path.parent / config["baseline_run"]).resolve()
        if baseline_run is None
        else Path(baseline_run).resolve()
    )
    config["output_directory"] = str(
        (config_path.parent / config["output_directory"]).resolve()
        if output is None
        else Path(output).resolve()
    )
    inputs = config["inputs"]
    if "data_root" in inputs:
        inputs["data_root"] = str((config_path.parent / inputs["data_root"]).resolve())
    prepared = Path(config["output_directory"]) / "boston_inputs"
    for key, filename in (("cases_path", "cases.parquet"), ("pairs_path", "observed_pairs.parquet")):
        inputs[key] = str(
            (config_path.parent / inputs[key]).resolve() if key in inputs
            else prepared / filename
        )

    config["scorers"] = config.get("scorers", [name for name in BASELINE_SCORES if name not in ("ES", "ED")])
    validate_scorers(config["scorers"])
    assessment = config.setdefault("assessment", {})
    if "treecluster_path" in assessment and assessment["treecluster_path"] is not None:
        assessment["treecluster_path"] = str(
            (config_path.parent / assessment["treecluster_path"]).resolve()
        )
    assessment.setdefault("treecluster_path", None)
    assessment.setdefault("focus_exposures", ["Conference", "SNF"])
    assessment.setdefault("min_cluster_size", 2)
    if (not isinstance(assessment["focus_exposures"], list)
            or not assessment["focus_exposures"]
            or any(not isinstance(x, str) or not x for x in assessment["focus_exposures"])
            or len(set(assessment["focus_exposures"])) != len(assessment["focus_exposures"])):
        raise ValueError("Boston focus_exposures must be unique nonempty strings")
    if not isinstance(assessment["min_cluster_size"], int) or assessment["min_cluster_size"] < 2:
        raise ValueError("Boston min_cluster_size must be an integer >= 2")
    trees = config.setdefault("trees", {})
    trees.setdefault("enabled", False)
    if trees["enabled"]:
        if not trees.get("alignment_path"):
            raise ValueError("Boston trees.alignment_path is required when trees are enabled")
        trees["alignment_path"] = str((config_path.parent / trees["alignment_path"]).resolve())
    config["implementation"] = implementation_signature()
    return config
