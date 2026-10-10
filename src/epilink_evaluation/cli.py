"""Command-line entry points for reproducible baseline and input preparation."""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

from .config import load_config, load_diagnostics_config
from .provenance import read_json, versions

STAGES = (
    "prepare",
    "backbone",
    "observations",
    "graphs",
    "pairwise",
    "clusters",
    "trees",
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
    config["experiment_root"] += "_smoke"
    config["inputs"]["smoke_cases"] = 64
    config["splits"] = {"train": [71001], "development": [72001], "evaluation": [73001]}
    if "diagnostics" in config:
        config["diagnostics"]["leiden"].update(resolutions=[0.1, 0.5], restarts=2)
        return config
    config["scorer"]["mc_samples"] = 1024
    config["thresholds"] = {
        "epilink": [0.1, 0.5],
        "genetic": [0, 2],
        "logistic": [0.01, 0.25],
    }
    config["clustering"]["leiden"].update(resolutions=[0.05, 0.5], restarts=2)
    config["treecluster"].update(
        genetic_threshold_snps=[1, 10], temporal_threshold_days=[14, 56]
    )
    if "phylogeny" not in config:
        config["phylogeny"] = {}
    config["phylogeny"].update(seed=76001)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "diagnostics",
            "baseline",
            "check",
            "scovmod",
            "prepare-boston",
            "perturbation",
            "boston",
            "reset-outputs",
        ),
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument(
        "--baseline-run",
        type=Path,
        help="Frozen baseline reference: runs/<id> or current.json",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="64-case pipeline validation in a separate output namespace",
    )
    parser.add_argument("--output", type=Path, help="Override output root")
    parser.add_argument(
        "--evaluations",
        nargs="+",
        choices=("diagnostics", "baseline", "perturbation", "boston", "all"),
        help="Evaluations to clear outputs for (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.command == "reset-outputs":
        return _reset_outputs(args)
    if args.command == "diagnostics":
        if args.baseline_run:
            parser.error(
                "Diagnostics use a shared synthetic experiment, not a frozen baseline"
            )
        if args.stage not in (
            None,
            "prepare",
            "backbone",
            "observations",
            "graphs",
            "report",
            "all",
        ):
            parser.error(
                "Diagnostics supports prepare, backbone, observations, graphs, report or all"
            )
        config = load_diagnostics_config(
            args.config or "evaluation/00_synthetic_diagnostics/config.yaml"
        )
        if args.output:
            config["output_directory"] = str(args.output.resolve())
        if args.smoke:
            config = smoke_config(config)
        if args.stage == "report":
            from .reporting.diagnostics import render_report

            directory = Path(
                read_json(Path(config["output_directory"]) / "current.json")[
                    "run_directory"
                ]
            )
            render_report(directory)
            print(directory / "report.html")
            return 0
        from .workflows.diagnostics import Diagnostics

        return 0 if Diagnostics(config).run(args.stage or "all") else 1
    if args.command == "perturbation":
        from .workflows.perturbation_config import load_study_config

        if args.stage not in (None, "all", "report"):
            parser.error(
                "Perturbation supports --stage all or report; all includes development selection and held-out clustering"
            )
        config = load_study_config(
            args.config or "evaluation/02_synthetic_perturbation/config.yaml",
            smoke=args.smoke,
            output=args.output,
            baseline_run=args.baseline_run,
        )
        if args.stage == "report":
            from .reporting.perturbation import render_report

            directory = Path(
                read_json(Path(config["output_directory"]) / "current.json")[
                    "run_directory"
                ]
            )
            render_report(directory)
            print(directory / "report.html")
            return 0
        from .workflows.perturbation import PerturbationStudy

        return 0 if PerturbationStudy(config).run(args.stage or "all") else 1
    if args.command in ("boston", "prepare-boston"):
        from .workflows.boston_config import load_study_config

        if args.command == "prepare-boston":
            if args.stage not in (None, "prepare"):
                parser.error("prepare-boston only supports --stage prepare")
            args.stage = "prepare"
        if args.stage not in (None, "prepare", "trees", "all", "report"):
            parser.error("Boston supports --stage prepare, trees, all or report")
        if args.smoke:
            parser.error("Boston does not support --smoke")
        config = load_study_config(
            config_path=args.config or "evaluation/03_boston_application/config.yaml",
            output=args.output,
            baseline_run=args.baseline_run,
        )
        if args.stage == "prepare":
            from .inputs.boston import prepare_boston

            inputs = config["inputs"]
            if "data_root" not in inputs:
                parser.error("Boston preparation requires inputs.data_root")
            directory = Path(inputs["cases_path"]).parent
            if (
                Path(inputs["cases_path"]) != directory / "cases.parquet"
                or Path(inputs["pairs_path"]) != directory / "observed_pairs.parquet"
            ):
                parser.error(
                    "Boston preparation requires cases.parquet and observed_pairs.parquet in one directory"
                )
            print(prepare_boston(inputs["data_root"], directory))
            return 0
        if args.stage == "report":
            from .reporting.boston import render_report

            directory = Path(
                read_json(Path(config["output_directory"]) / "current.json")[
                    "run_directory"
                ]
            )
            render_report(directory)
            print(directory / "report.html")
            return 0
        from .workflows.boston import BostonEmpirical

        return 0 if BostonEmpirical(config).run(args.stage or "all") else 1
    if args.baseline_run:
        parser.error("--baseline-run applies to perturbation or boston only")
    args.config = args.config or Path("evaluation/01_synthetic_baseline/config.yaml")
    if args.command == "scovmod":
        if args.stage not in (None, "prepare"):
            parser.error("SCoVMod supports --stage prepare only")
        args.stage = "prepare"
    args.stage = args.stage or "develop"
    if args.stage == "trees":
        parser.error("--stage trees applies to boston only")
    if args.stage in ("backbone", "observations", "graphs"):
        parser.error("--stage backbone/observations/graphs applies to diagnostics only")
    config = load_config(args.config)
    if args.output:
        config["output_directory"] = str(args.output.resolve())
    if args.smoke:
        config = smoke_config(config)
    if args.command == "scovmod":
        from .inputs.scovmod import prepare_tree

        print(prepare_tree(config).parent)
        return 0
    from .scorers import SCORERS

    unknown = set(config["scorers"]) - set(SCORERS)
    if unknown:
        parser.error(f"Unknown scorer identifiers: {sorted(unknown)}")
    if args.command == "check":
        from .phylogeny.external import command_identity, treecluster_executable
        from .workflows.settings import settings_registry

        tools = {}
        treecluster_exec = treecluster_executable(config["treecluster"])
        try:
            tools["treecluster"] = command_identity(treecluster_exec)
        except FileNotFoundError as exc:
            tools["treecluster"] = {"unavailable": str(exc)}

        phylogeny = config.get("phylogeny", {})
        iqtree_exec = phylogeny.get("executable", phylogeny.get("iqtree_executable", "iqtree"))
        try:
            tools["iqtree"] = command_identity(iqtree_exec)
        except FileNotFoundError as exc:
            tools["iqtree"] = {"unavailable": str(exc)}

        settings = settings_registry(config)
        print(
            json.dumps(
                {
                    "name": config["name"],
                    "versions": versions(),
                    "tools": tools,
                    "splits": config["splits"],
                    "output": config["output_directory"],
                    "operating_definitions_per_realization": None,
                    "configured_clustering_definitions_per_realization": sum(
                        d["kind"] != "pairwise" for d in settings.values()
                    ),
                    "pairwise_candidates_pending_development": True,
                    "tree_input_exists": Path(config["inputs"]["tree_path"]).exists(),
                },
                indent=2,
            )
        )
        return int(
            config["treecluster"].get("enabled", False)
            and any("unavailable" in tool for tool in tools.values())
        )
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
    if args.command == "reset-outputs":
        return _reset_outputs(args)
    from .workflows.baseline import Baseline

    return 0 if Baseline(config).run(args.stage) else 1


def _reset_outputs(args):
    """Clear outputs from selected evaluations."""
    evaluations = args.evaluations or ["all"]
    if "all" in evaluations:
        evaluations = ["diagnostics", "baseline", "perturbation", "boston"]

    dry_run = args.dry_run
    total_removed = 0

    for eval_name in evaluations:
        if eval_name == "diagnostics":
            output_roots = [
                Path("evaluation/00_synthetic_diagnostics/outputs/diagnostics"),
                Path("evaluation/00_synthetic_diagnostics/outputs/diagnostics_smoke"),
            ]
        elif eval_name == "baseline":
            output_roots = [
                Path("evaluation/01_synthetic_baseline/outputs/baseline"),
                Path("evaluation/01_synthetic_baseline/outputs/baseline_smoke"),
            ]
        elif eval_name == "perturbation":
            output_roots = [
                Path("evaluation/02_synthetic_perturbation/outputs/perturbation"),
                Path("evaluation/02_synthetic_perturbation/outputs/perturbation_smoke"),
            ]
        elif eval_name == "boston":
            output_roots = [
                Path("evaluation/03_boston_application/outputs/boston"),
            ]
        else:
            continue

        for output_root in output_roots:
            if not output_root.exists():
                continue

            removed_count = 0
            if dry_run:
                runs_dir = output_root / "runs"
                artifacts_dir = output_root / "artifacts"
                current_json = output_root / "current.json"

                if runs_dir.exists():
                    runs = list(runs_dir.iterdir())
                    if runs:
                        print(f"Would remove {len(runs)} run(s) from {output_root}")
                        for run in runs:
                            print(f"  - {run}")
                        removed_count += len(runs)

                if artifacts_dir.exists():
                    import shutil

                    size = sum(
                        f.stat().st_size
                        for f in artifacts_dir.rglob("*")
                        if f.is_file()
                    )
                    print(
                        f"Would remove artifacts from {output_root} ({size / 1024 / 1024:.1f} MB)"
                    )
                    removed_count += 1

                if current_json.exists():
                    print(f"Would remove {current_json}")
            else:
                import shutil

                runs_dir = output_root / "runs"
                artifacts_dir = output_root / "artifacts"
                current_json = output_root / "current.json"

                if runs_dir.exists():
                    runs = list(runs_dir.iterdir())
                    if runs:
                        shutil.rmtree(runs_dir)
                        print(f"Removed {len(runs)} run(s) from {output_root}")
                        removed_count += len(runs)

                if artifacts_dir.exists():
                    shutil.rmtree(artifacts_dir)
                    print(f"Removed artifacts from {output_root}")
                    removed_count += 1

                if current_json.exists():
                    current_json.unlink()
                    print(f"Removed {current_json}")
                    removed_count += 1

            total_removed += removed_count

    if total_removed == 0:
        print("No outputs found to remove")
    elif dry_run:
        print(f"\nTotal: would remove {total_removed} items (dry run)")
    else:
        print(f"\nTotal: removed {total_removed} items")

    return 0
