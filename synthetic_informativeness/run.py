"""Run matched-baseline informativeness analyses.

From the repository root:
    python3 -m synthetic_informativeness.run
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import yaml
from evaluation.config import build_run_specs, load_config

from synthetic_exploration.common import log, source_hashes, versions, write_json
from synthetic_exploration.data import load_dataset, prepare_dataset, score_pairs

from .clusters import investigate_clusters
from .common import resolve_relative
from .pairwise import investigate_pairwise
from .report import make_report
from .treecluster import investigate_treecluster


def normalize_paths(settings: dict, config_path: Path) -> dict:
    settings = dict(settings)
    tc = dict(settings.get("treecluster", {}))
    if "raw_trees" in tc:
        tc["raw_trees"] = {k: str(resolve_relative(config_path, v)) for k, v in tc["raw_trees"].items()}
    if "dated_trees" in tc:
        tc["dated_trees"] = {k: str(resolve_relative(config_path, v)) for k, v in tc["dated_trees"].items()}
    settings["treecluster"] = tc
    return settings


def selected_run(source: dict, settings: dict):
    runs = [r for r in build_run_specs(source)
            if r.condition == settings["condition"] and r.scenario_name == settings["scenario"]]
    if len(runs) != 1:
        raise ValueError("Expected exactly one matched baseline run from the source configuration.")
    return runs[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    config_path = args.config.resolve()
    settings = yaml.safe_load(config_path.read_text())
    settings = normalize_paths(settings, config_path)
    source_path = resolve_relative(config_path, settings["source_config"])
    source = load_config(source_path)
    output = resolve_relative(config_path, settings["output_directory"])
    data_cache = resolve_relative(config_path, settings.get("data_cache_directory", settings["output_directory"]))
    run = selected_run(source, settings)
    seed = int(settings["seed"])
    training_seed = int(settings["training_seed"])
    if training_seed == seed:
        raise ValueError("training_seed must differ from seed")
    directory = output / "runs" / run.condition / run.scenario_name / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "run": asdict(run),
        "seed": seed,
        "training_seed": training_seed,
        "settings": settings,
        "source_config": str(source_path),
        "versions": versions(),
        "source_hashes": source_hashes(),
    }
    write_json(directory / "manifest.json", manifest)
    dataset = prepare_dataset(run.tree_path, run.generation_parameters, seed, data_cache)
    training_dataset = prepare_dataset(run.tree_path, run.generation_parameters, training_seed, data_cache)
    pairs, cases = load_dataset(dataset)
    training_pairs, _ = load_dataset(training_dataset)
    epilink_scores, _ = score_pairs(pairs, run.inference_parameters, seed, settings["models"])
    epilink_scores.to_parquet(directory / "epilink_scores.parquet", index=False)
    candidates = investigate_pairwise(
        pairs, training_pairs, epilink_scores, settings["models"], directory / "01_pairwise", settings)
    investigate_clusters(pairs, cases, candidates, run, directory / "02_clusters", settings, seed)
    investigate_treecluster(pairs, cases, run, directory / "03_treecluster", settings)
    make_report(directory, pairs, cases, settings)
    manifest["status"] = "complete"
    manifest["dataset"] = str(dataset)
    manifest["training_dataset"] = str(training_dataset)
    manifest["data_cache"] = str(data_cache)
    manifest["dataset_manifest"] = json.loads((dataset / "manifest.json").read_text())
    write_json(directory / "manifest.json", manifest)
    log(f"Complete: {directory / 'report.html'}")


if __name__ == "__main__":
    main()
