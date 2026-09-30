"""Run from the repository root: python3 -m synthetic_exploration.run."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd
import yaml
from evaluation.config import load_config, build_run_specs

from .common import log, write_json, source_hashes, versions
from .data import prepare_dataset, load_dataset, score_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--stage", choices=["prepare", "all"], default="all")
    args = parser.parse_args()
    config_path = args.config.resolve()
    settings = yaml.safe_load(config_path.read_text())
    source_path = (config_path.parent / settings["source_config"]).resolve()
    source = load_config(source_path)
    output = (config_path.parent / settings["output_directory"]).resolve()
    runs = [r for r in build_run_specs(source)
            if r.condition in settings["conditions"] and r.scenario_name in settings["scenarios"]]
    requested = {(c, s) for c in settings["conditions"] for s in settings["scenarios"]}
    if {(r.condition, r.scenario_name) for r in runs} != requested:
        raise ValueError("Requested conditions/scenarios are missing from the source configuration.")
    for run in runs:
        for seed in settings["seeds"]:
            dataset = prepare_dataset(run.tree_path, run.generation_parameters, seed, output)
            if args.stage == "prepare":
                continue
            from .observations import investigate_observations
            from .scores import investigate_scores
            from .clusters import investigate_clusters
            from .validation import investigate_validation
            from .report import make_report
            from .table import export_analysis_table
            directory = output / "runs" / run.condition / run.scenario_name / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            manifest = {
                "status": "running", "run": asdict(run), "seed": seed,
                "settings": settings, "source_config": str(source_path),
                "dataset": str(dataset), "versions": versions(), "source_hashes": source_hashes(),
            }
            write_json(directory / "manifest.json", manifest)
            pairs, cases = load_dataset(dataset)
            scores, models = score_pairs(pairs, run.inference_parameters, seed, settings["models"])
            scores.to_parquet(directory / "scores.parquet", index=False)
            export_analysis_table(pairs, scores, run, seed, output, settings["models"])
            investigate_observations(pairs, directory / "01_observations", settings)
            investigate_scores(pairs, scores, directory / "02_scores", settings)
            investigate_clusters(pairs, cases, scores, run, directory / "03_clusters", settings, seed)
            investigate_validation(pairs, cases, scores, models, run, source, output,
                                   directory / "04_validation", settings, seed, config_path.parent)
            make_report(directory, pairs, cases, settings)
            manifest["status"] = "complete"
            manifest["dataset_manifest"] = json.loads((dataset / "manifest.json").read_text())
            write_json(directory / "manifest.json", manifest)
            log(f"Complete: {directory / 'report.html'}")


if __name__ == "__main__":
    main()
