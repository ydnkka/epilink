"""Explicit experiment definitions; paths are relative to the configuration file."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from .natural_history import natural_history
from .selection.search import cutoff_settings, resolution_settings


def _load(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    config = deepcopy(config)
    config["output_directory"] = str(
        (path.parent / config["output_directory"]).resolve()
    )
    if config.get("experiment_config"):
        shared_path = (path.parent / config["experiment_config"]).resolve()
        shared = _load(shared_path)
        for key in ("inputs", "generation", "simulation", "splits"):
            if key in config and config[key] != shared[key]:
                raise ValueError(
                    f"Configure {key} in the shared experiment, not the study"
                )
            config[key] = deepcopy(shared[key])
        config["experiment_config"] = str(shared_path)
        config["experiment_root"] = shared["output_directory"]
    if config.get("experiment_root"):
        config["experiment_root"] = str(
            (path.parent / config["experiment_root"]).resolve()
        )
    for key in ("tree_path", "tree_source_path", "infection_path", "transmission_path"):
        if config["inputs"].get(key):
            config["inputs"][key] = str((path.parent / config["inputs"][key]).resolve())
    config["config_path"] = str(path)
    return config


def load_config(path):
    config = _load(path)
    config.setdefault("inference", deepcopy(config["generation"]))
    validate(config)
    return config


def validate_experiment(config):
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version: 1")
    splits = config["splits"]
    all_seeds = []
    for split in ("train", "development", "evaluation"):
        seeds = splits[split]
        if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError(f"{split} requires nonnegative integer seeds")
        all_seeds.extend(seeds)
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("Training, development and evaluation seeds must be distinct")
    for params in (config["generation"],):
        for name in ("incubation", "testing_delay"):
            for field in ("mean", "cv"):
                if not np.isfinite(params[name][field]) or params[name][field] <= 0:
                    raise ValueError(f"{name}.{field} must be finite and positive")
        if params["substitution_rate"] <= 0 or params["relaxation"] < 0:
            raise ValueError("Invalid substitution_rate or relaxation")
        if int(params["genome_length"]) <= 0:
            raise ValueError("genome_length must be positive")
    if not 0 < config["simulation"]["fraction_sampled"] <= 1:
        raise ValueError("fraction_sampled must be in (0, 1]")
    if config["simulation"]["sequence_length"] <= 0:
        raise ValueError("sequence_length must be positive")
    natural_history(config["generation"])


def validate(config):
    validate_experiment(config)
    if config["inference"] != config["generation"]:
        raise ValueError(
            "The initial baseline requires matched generation/inference parameters"
        )
    if config["clustering"]["leiden"]["objective"] not in ("CPM", "modularity"):
        raise ValueError("Leiden objective must be CPM or modularity")
    if config["clustering"]["leiden"]["restarts"] < 1:
        raise ValueError("Leiden requires at least one restart")
    leiden = config["clustering"]["leiden"]
    resolution_settings(leiden["resolutions"])
    cutoff_settings(config["treecluster"]["genetic_threshold_snps"], integer=True)
    cutoff_settings(config["treecluster"]["temporal_threshold_days"])
    tree_seed = config["treecluster"].get("rng_seed")
    if tree_seed is not None and (type(tree_seed) is not int or tree_seed < 0):
        raise ValueError("rng_seed must be a nonnegative integer")
    if len(set(config["scorers"])) != len(config["scorers"]):
        raise ValueError("Duplicate scorer identifiers")
    ids = [criterion["name"] for criterion in config["selection"]["criteria"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate operating criterion names")


def load_diagnostics_config(path):
    config = _load(path)
    validate_experiment(config)
    settings = config["diagnostics"]
    unknown = set(settings) - {"backbone", "leiden"}
    if unknown:
        raise ValueError(f"Unknown diagnostic settings: {sorted(unknown)}")
    leiden = settings["leiden"]
    if leiden["objective"] not in ("CPM", "modularity"):
        raise ValueError("Leiden objective must be CPM or modularity")
    if type(leiden["restarts"]) is not int or leiden["restarts"] < 1:
        raise ValueError("Leiden requires positive integer restarts")
    if type(leiden["seed"]) is not int or leiden["seed"] < 0:
        raise ValueError("Leiden seed must be a nonnegative integer")
    resolution_settings(leiden["resolutions"])
    return config
