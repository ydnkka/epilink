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

    config["scorers"] = config.get("scorers", list(BASELINE_SCORES))
    validate_scorers(config["scorers"])
    config["implementation"] = implementation_signature()
    return config
