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
        "command", choices=("baseline", "check", "prepare-tree", "prepare-boston", "perturbation", "boston")
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--baseline-run", type=Path, help="Perturbation reference: runs/<id> or current.json")
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
    if args.command == "perturbation":
        from .workflows.perturbation_config import load_study_config

        if args.stage not in (None, "all", "report"):
            parser.error("Perturbation supports --stage all or report; settings are already frozen")
        config = load_study_config(
            args.config or "synthetic_perturbation/config.yaml", smoke=args.smoke,
            output=args.output, baseline_run=args.baseline_run,
        )
        if args.stage == "report":
            from .reporting.perturbation import render_report

            directory = Path(read_json(Path(config["output_directory"]) / "current.json")["run_directory"])
            render_report(directory)
            print(directory / "report.html")
            return 0
        from .workflows.perturbation import PerturbationStudy

        return 0 if PerturbationStudy(config).run() else 1
    if args.baseline_run:
        parser.error("--baseline-run applies to perturbation only")
    args.config = args.config or Path("synthetic_baseline/config.yaml")
    args.stage = args.stage or "develop"
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

        root = Path(config["inputs"]["infection_path"]).parents[2]
        print(prepare_boston(root, Path(config["output_directory"]) / "boston_inputs"))
        return 0
    if args.command == "boston":
        from .workflows.boston import main as boston_main

        return boston_main([
            "--config", str(args.config or Path("synthetic_baseline/boston_config.yaml")),
            "--baseline-run", str(args.baseline_run) if args.baseline_run else "current.json",
            *(["--output", str(args.output)] if args.output else []),
        ])
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
