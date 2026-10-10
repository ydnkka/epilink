"""Explicit experiment definitions; paths are relative to the configuration file."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from .natural_history import natural_history  # noqa: F401


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
    pairwise = config.setdefault("pairwise", {})
    pairwise.setdefault("threshold_mode", "all_development_scores")
    pairwise.setdefault("selected_fractions", [])
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
    grid = leiden.get("resolutions", [])
    if not grid or any(not np.isfinite(r) or r <= 0 for r in grid):
        raise ValueError("Leiden resolutions must be finite and positive")
    if config["pairwise"].get("threshold_mode", "configured") not in (
        "configured",
        "all_development_scores",
    ):
        raise ValueError("Unknown pairwise threshold_mode")
    tolerance = config.get("grid_audit", {}).get("objective_tolerance", 0.005)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError(
            "Grid audit objective_tolerance must be finite and nonnegative"
        )
    for family, thresholds in config["thresholds"].items():
        if not thresholds or any(not np.isfinite(t) for t in thresholds):
            raise ValueError(f"Invalid {family} threshold grid")
        if family == "logistic" and any(t < 0 or t > 1 for t in thresholds):
            raise ValueError("Probability thresholds must be in [0, 1]")
    for name in ("temporal_threshold_days",):
        if any(not np.isfinite(t) or t < 0 for t in config["treecluster"][name]):
            raise ValueError("TreeCluster thresholds must be finite and nonnegative")
    genetic_threshold_snps = config["treecluster"]["genetic_threshold_snps"]
    if not genetic_threshold_snps or any(
        not np.isfinite(t) or t < 0 or int(t) != t for t in genetic_threshold_snps
    ):
        raise ValueError("genetic_threshold_snps must be nonnegative integers")
    tree_seed = config["treecluster"].get("rng_seed")
    if tree_seed is not None and (type(tree_seed) is not int or tree_seed < 0):
        raise ValueError("rng_seed must be a nonnegative integer")
    if len(set(config["scorers"])) != len(config["scorers"]):
        raise ValueError("Duplicate scorer identifiers")
    ids = [criterion["name"] for criterion in config["selection"]["criteria"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate operating criterion names")
    reference = config.get("grid_audit", {}).get("reference")
    if reference is not None:
        if set(reference) - {"thresholds", "leiden_resolution_grid", "treecluster"}:
            raise ValueError("Unknown grid audit reference fields")
        if set(reference.get("thresholds", {})) - set(config["thresholds"]):
            raise ValueError("Unknown reference threshold family")
        if set(reference.get("treecluster", {})) - {
            "genetic_threshold_snps",
            "temporal_threshold_days",
        }:
            raise ValueError("Reference TreeCluster overrides must be threshold grids")
        coarse = reference_grid_config(config)
        coarse.pop("grid_audit", None)
        validate(coarse)


def reference_grid_config(config):
    """Apply the declared coarse grids to the same scientific comparison."""
    result = deepcopy(config)
    reference = config.get("grid_audit", {}).get("reference", {})
    result["pairwise"]["threshold_mode"] = "configured"
    result["thresholds"].update(reference.get("thresholds", {}))
    if "leiden_resolution_grid" in reference:
        result["clustering"]["leiden"]["resolutions"] = reference["leiden_resolution_grid"]
    result["treecluster"].update(reference.get("treecluster", {}))
    return result


def load_diagnostics_config(path):
    config = _load(path)
    validate_experiment(config)
    settings = config["diagnostics"]
    leiden = settings["leiden"]
    if leiden["objective"] not in ("CPM", "modularity"):
        raise ValueError("Leiden objective must be CPM or modularity")
    if type(leiden["restarts"]) is not int or leiden["restarts"] < 1:
        raise ValueError("Leiden requires positive integer restarts")
    if type(leiden["seed"]) is not int or leiden["seed"] < 0:
        raise ValueError("Leiden seed must be a nonnegative integer")
    if not leiden["resolution_grid"] or any(
        not np.isfinite(r) or r <= 0 for r in leiden["resolution_grid"]
    ):
        raise ValueError("Leiden resolutions must be finite and positive")
    trees = settings["treecluster"]
    if not trees["methods"] or set(trees["methods"]) - {
        "max_clade",
        "avg_clade",
        "single_linkage",
    }:
        raise ValueError("Unsupported transmission-tree clustering method")
    if not trees["threshold_hops"] or any(
        not np.isfinite(t) or t < 0 for t in trees["threshold_hops"]
    ):
        raise ValueError("Transmission-hop thresholds must be finite and nonnegative")
    return config
