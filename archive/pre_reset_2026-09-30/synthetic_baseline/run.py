"""Run comprehensive baseline assessment.

From the repository root:
    python -m synthetic_baseline.run [--stage STAGE] [--config CONFIG] [--full-sweep]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from evaluation.config import build_run_specs, load_config
from synthetic_exploration.common import log, source_hashes, versions
from synthetic_exploration.data import load_dataset, prepare_dataset, score_pairs


def prepare_and_load_dataset(tree_path, generation_parameters, seed, output_root):
    """Prepare dataset if needed and load pairs/cases."""
    dataset = prepare_dataset(tree_path, generation_parameters, seed, Path(output_root))
    pairs, cases = load_dataset(dataset)
    return pairs, cases


def normalize_paths(settings: dict, config_path: Path) -> dict:
    settings = dict(settings)
    tc = dict(settings.get("treecluster", {}))
    if "raw_trees" in tc:
        tc["raw_trees"] = {k: str((config_path.parent / v).resolve()) for k, v in tc["raw_trees"].items()}
    if "dated_trees" in tc:
        tc["dated_trees"] = {k: str((config_path.parent / v).resolve()) for k, v in tc["dated_trees"].items()}
    settings["treecluster"] = tc
    return settings


def apply_full_sweep(settings: dict) -> dict:
    """Apply full-sweep mode settings."""
    if "full_sweep" in settings:
        fs = settings["full_sweep"]
        if "clusters_resolutions" in fs:
            settings["clusters"]["leiden_resolutions"] = fs["clusters_resolutions"]
        if "pairwise_thresholds" in fs:
            settings["pairwise"]["thresholds"] = fs["pairwise_thresholds"]
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
    parser.add_argument("--stage", choices=["prepare", "01", "02", "03", "04", "all"], default="all",
                        action="append", nargs="?", const="all")
    parser.add_argument("--full-sweep", action="store_true", help="Use extended resolution/threshold sweeps")
    args = parser.parse_args()
    
    config_path = args.config.resolve()
    settings = yaml.safe_load(config_path.read_text())
    settings = normalize_paths(settings, config_path)
    
    if args.full_sweep:
        settings = apply_full_sweep(settings)
        log("Running in full-sweep mode (extended resolutions/thresholds)")
    
    # Flatten stage argument (handle both --stage X and --stage X --stage Y)
    stages = args.stage if isinstance(args.stage, list) else [args.stage]
    if "all" in stages:
        stages = ["01", "02", "03", "04"]
    
    source_path = (config_path.parent / settings["source_config"]).resolve()
    source = load_config(source_path)
    output = (config_path.parent / settings["output_directory"]).resolve()
    run = selected_run(source, settings)
    
    for seed in settings["seeds"]:
        seed = int(seed)
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
            "stages_to_run": stages,
        }
        
        # Prepare datasets
        log(f"Preparing evaluation dataset (seed={seed})")
        dataset = prepare_dataset(run.tree_path, run.generation_parameters, seed, output)
        log(f"Preparing training dataset (seed={training_seed})")
        training_dataset = prepare_dataset(run.tree_path, run.generation_parameters, training_seed, output)
        
        pairs, cases = load_dataset(dataset)
        training_pairs, _ = load_dataset(training_dataset)
        
        log(f"Scoring all observed pairs")
        epilink_scores, models = score_pairs(pairs, run.inference_parameters, seed, settings["models"])
        epilink_scores.to_parquet(directory / "epilink_scores.parquet", index=False)
        
        # Run stages
        if "01" in stages:
            from .stage_01_observations import investigate_observations
            investigate_observations(pairs, directory / "01_observations", settings)
        
        if "02" in stages:
            from .stage_02_pairwise import investigate_pairwise
            candidates = investigate_pairwise(pairs, training_pairs, epilink_scores, settings["models"],
                                              directory / "02_pairwise", settings)
        else:
            candidates = {}
        
        if "03" in stages:
            from .stage_03_clusters import investigate_clusters
            investigate_clusters(pairs, cases, candidates, run, directory / "03_clusters", settings, seed)
        
        if "04" in stages:
            from .stage_04_validation import investigate_validation
            investigate_validation(pairs, cases, epilink_scores, models, run, source, output,
                                   directory / "04_validation", settings, seed, config_path.parent)
        
        # Generate report
        from .report import make_report
        make_report(directory, pairs, cases, settings)
        
        manifest["status"] = "complete"
        manifest["dataset"] = str(dataset)
        manifest["training_dataset"] = str(training_dataset)
        manifest["dataset_manifest"] = json.loads((dataset / "manifest.json").read_text())
        write_json = lambda p, v: Path(p).write_text(json.dumps(v, indent=2, default=str) + "\n")
        write_json(directory / "manifest.json", manifest)
        
        log(f"Complete: {directory / 'report.html'}")


if __name__ == "__main__":
    main()
