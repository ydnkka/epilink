"""Command-line entry points for reproducible baseline and input preparation."""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

from .config import load_config
from .provenance import read_json, versions

STAGES = (
    "prepare",
    "pairwise",
    "clusters",
    "develop",
    "select",
    "evaluate",
    "report",
    "all",
)


def smoke_config(config):
    config = deepcopy(config)
    config["name"] += "_smoke"
    config["output_directory"] += "_smoke"
    config["inputs"]["smoke_cases"] = 64
    config["splits"] = {"train": [71001], "development": [72001], "evaluation": [73001]}
    config["scorer"]["mc_samples"] = 1024
    config["thresholds"] = {
        "epilink": [0.1, 0.5],
        "genetic": [0, 2],
        "logistic": [0.01, 0.25],
    }
    config["clustering"]["leiden"].update(resolutions=[0.05, 0.5], restarts=2)
    config["treecluster"].update(
        genetic_thresholds=[0.0002, 0.002], threshold_days=[14, 56], rng_seed=76001
    )
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("baseline", "check", "prepare-tree", "prepare-boston")
    )
    parser.add_argument("--config", default="synthetic_baseline/config.yaml", type=Path)
    parser.add_argument("--stage", choices=STAGES, default="develop")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="64-case pipeline validation in a separate output namespace",
    )
    parser.add_argument("--output", type=Path, help="Override output root")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config(args.config)
    if args.output:
        config["output_directory"] = str(args.output.resolve())
    if args.smoke:
        config = smoke_config(config)
    from .scorers import SCORERS

    unknown = set(config["scorers"]) - set(SCORERS)
    if unknown:
        parser.error(f"Unknown scorer identifiers: {sorted(unknown)}")
    if args.command == "check":
        from .phylogeny.external import command_identity
        from .workflows.settings import settings_registry

        tools = {}
        for name, command in config["treecluster"]["executables"].items():
            try:
                tools[name] = command_identity(command)
            except FileNotFoundError as exc:
                tools[name] = {"unavailable": str(exc)}
        settings = settings_registry(config)
        print(
            json.dumps(
                {
                    "name": config["name"],
                    "versions": versions(),
                    "tools": tools,
                    "splits": config["splits"],
                    "output": config["output_directory"],
                    "operating_definitions_per_realization": len(settings),
                    "tree_input_exists": Path(config["inputs"]["tree_path"]).exists(),
                },
                indent=2,
            )
        )
        return int(
            config["treecluster"]["enabled"]
            and any("unavailable" in tool for tool in tools.values())
        )
    if args.command == "prepare-tree":
        from .inputs.scovmod import prepare_tree

        print(prepare_tree(config))
        return 0
    if args.command == "prepare-boston":
        from .inputs.boston import prepare_boston

        # The baseline config anchors the repository input directory.
        root = Path(config["inputs"]["infection_path"]).parents[2]
        print(prepare_boston(root, Path(config["output_directory"]) / "boston_inputs"))
        return 0
    if args.stage == "report":
        from .reporting.report import render_report

        directory = Path(
            read_json(Path(config["output_directory"]) / "current.json")[
                "run_directory"
            ]
        )
        render_report(directory)
        print(directory / "report.html")
        return 0
    from .workflows.baseline import Baseline

    return 0 if Baseline(config).run(args.stage) else 1
