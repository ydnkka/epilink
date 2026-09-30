"""Configuration loading for Boston empirical studies."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..provenance import implementation_signature


def load_study_config(
    argv=None, config_path=None, baseline_run=None, output=None, seeds=None
):
    if argv is not None:
        import argparse

        parser = argparse.ArgumentParser(description="Boston empirical clustering")
        parser.add_argument("--config", type=Path, default=config_path)
        parser.add_argument("--baseline-run", type=Path, dest="baseline_run")
        parser.add_argument("--output", type=Path, default=output)
        parser.add_argument("--seeds", type=int, nargs="+", default=seeds)
        args = parser.parse_args(argv)
        config_path = args.config
        baseline_run = args.baseline_run
        output = args.output
        seeds = args.seeds

    config_path = Path(config_path or "synthetic_baseline/boston_config.yaml").resolve()
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

    if seeds is not None:
        config["seeds"] = seeds
    if (
        not config["seeds"]
        or any(type(s) is not int or s < 0 for s in config["seeds"])
        or len(set(config["seeds"])) != len(config["seeds"])
    ):
        raise ValueError("Seeds must be distinct nonnegative integers")

    config["implementation"] = implementation_signature()
    config["scorers"] = config.get(
        "scorers", ["EDD", "EDS", "ESD", "ESS", "GD_D", "GD_S", "LOGIT_D", "LOGIT_S"]
    )
    return config
