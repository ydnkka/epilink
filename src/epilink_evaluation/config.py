"""Explicit experiment definitions; paths are relative to the configuration file."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    config = deepcopy(config)
    for key in ("output_directory",):
        config[key] = str((path.parent / config[key]).resolve())
    for key in ("tree_path", "infection_path", "transmission_path"):
        if config["inputs"].get(key):
            config["inputs"][key] = str((path.parent / config["inputs"][key]).resolve())
    config["config_path"] = str(path)
    validate(config)
    return config


def validate(config):
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
    for params in (config["generation"], config["inference"]):
        for name in ("incubation", "testing_delay"):
            for field in ("mean", "cv"):
                if not np.isfinite(params[name][field]) or params[name][field] <= 0:
                    raise ValueError(f"{name}.{field} must be finite and positive")
        if params["substitution_rate"] <= 0 or params["relaxation"] < 0:
            raise ValueError("Invalid substitution_rate or relaxation")
        if int(params["genome_length"]) <= 0:
            raise ValueError("genome_length must be positive")
    if config["inference"] != config["generation"]:
        raise ValueError(
            "The initial baseline requires matched generation/inference parameters"
        )
    if not 0 < config["simulation"]["fraction_sampled"] <= 1:
        raise ValueError("fraction_sampled must be in (0, 1]")
    if config["simulation"]["sequence_length"] <= 0:
        raise ValueError("sequence_length must be positive")
    if config["clustering"]["leiden"]["objective"] not in ("CPM", "modularity"):
        raise ValueError("Leiden objective must be CPM or modularity")
    if config["clustering"]["leiden"]["restarts"] < 1:
        raise ValueError("Leiden requires at least one restart")
    for resolution in config["clustering"]["leiden"]["resolutions"]:
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("Leiden resolutions must be finite and positive")
    for family, thresholds in config["thresholds"].items():
        if not thresholds or any(not np.isfinite(t) for t in thresholds):
            raise ValueError(f"Invalid {family} threshold grid")
        if family == "logistic" and any(t < 0 or t > 1 for t in thresholds):
            raise ValueError("Probability thresholds must be in [0, 1]")
    for name in ("genetic_thresholds", "threshold_days"):
        if any(not np.isfinite(t) or t < 0 for t in config["treecluster"][name]):
            raise ValueError("TreeCluster thresholds must be finite and nonnegative")
    if len(set(config["scorers"])) != len(config["scorers"]):
        raise ValueError("Duplicate scorer identifiers")
    ids = [criterion["name"] for criterion in config["selection"]["criteria"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate operating criterion names")


def natural_history(params):
    from epilink import NaturalHistoryParameters

    expanded = {
        k: v for k, v in params.items() if k not in ("incubation", "testing_delay")
    }
    for name in ("incubation", "testing_delay"):
        shape = 1 / params[name]["cv"] ** 2
        expanded[f"{name}_shape"] = shape
        expanded[f"{name}_scale"] = params[name]["mean"] / shape
    return NaturalHistoryParameters(**expanded)
