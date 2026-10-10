"""One-at-a-time perturbations with a two-by-two EpiLink clustering design."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from ..config import natural_history

PARAMETERS = {
    "incubation.mean",
    "incubation.cv",
    "testing_delay.mean",
    "testing_delay.cv",
    "substitution_rate",
    "relaxation",
}
MODES = {
    "baseline_inference_baseline_clustering": ("baseline", "baseline"),
    "baseline_inference_updated_clustering": ("baseline", "updated"),
    "matched_inference_baseline_clustering": ("matched", "baseline"),
    "matched_inference_updated_clustering": ("matched", "updated"),
}


def load_study_config(path, *, smoke=False, output=None, baseline_run=None):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Expected clustering-only perturbation schema_version: 1")
    for key in ("baseline_run", "output_directory"):
        config[key] = str((path.parent / config[key]).resolve())
    if output is not None:
        config["output_directory"] = str(Path(output).resolve())
    if baseline_run is not None:
        config["baseline_run"] = str(Path(baseline_run).resolve())
    config["config_path"] = str(path)
    all_seeds = []
    for role in ("development_seeds", "seeds"):
        seeds = config[role]
        if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError(f"{role} must contain nonnegative integer seeds")
        all_seeds.extend(seeds)
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError(
            "Perturbation development and evaluation seeds must be distinct"
        )
    modes = config["modes"]
    if set(modes) != set(MODES) or len(modes) != len(MODES):
        raise ValueError("Configure all four unique inference/clustering modes")
    scorers = config["scorers"]
    if (
        not scorers
        or not set(scorers) <= {"EDD", "EDS", "ESD", "ESS"}
        or len(set(scorers)) != len(scorers)
    ):
        raise ValueError("Perturbation scorers must be unique EpiLink identifiers")
    if not isinstance(config["criterion"], str) or not config["criterion"]:
        raise ValueError("Specify a baseline operating criterion")
    if not config["perturbations"]:
        raise ValueError("Configure at least one parameter perturbation")
    parameters = []
    for item in config["perturbations"]:
        parameter = item["parameter"]
        if parameter not in PARAMETERS or parameter in parameters:
            raise ValueError(
                f"Unsupported or duplicate perturbation parameter: {parameter}"
            )
        parameters.append(parameter)
        kinds = set(item) - {"parameter"}
        if kinds not in ({"multipliers"}, {"values"}):
            raise ValueError("Each parameter requires either multipliers or values")
        kind = next(iter(kinds))
        levels = item[kind]
        if (
            not levels
            or any(type(x) not in (int, float) or not np.isfinite(x) for x in levels)
            or len(set(levels)) != len(levels)
        ):
            raise ValueError(f"Invalid perturbation levels for {parameter}")
        if kind == "multipliers" and any(x <= 0 for x in levels):
            raise ValueError("Perturbation multipliers must be positive")
    config["smoke_mode"] = bool(smoke)
    config["case_limit"] = None
    if smoke:
        settings = config["smoke"]
        seed, development_seed, cases = (
            settings["seed"],
            settings["development_seed"],
            settings["cases"],
        )
        if (
            any(
                type(s) is not int or s < 0 or s in all_seeds
                for s in (seed, development_seed)
            )
            or seed == development_seed
        ):
            raise ValueError(
                "Smoke development/evaluation seeds must be distinct "
                "and separate from study seeds"
            )
        if type(cases) is not int or cases < 4:
            raise ValueError("Smoke cases must be an integer of at least four")
        chosen = settings["parameters"]
        if not chosen or not set(chosen) <= set(parameters):
            raise ValueError(
                "Smoke parameters must be a nonempty subset of configured parameters"
            )
        config["seeds"] = [seed]
        config["development_seeds"] = [development_seed]
        config["case_limit"] = cases
        config["perturbations"] = [
            x for x in config["perturbations"] if x["parameter"] in chosen
        ]
        config["name"] += "_smoke"
        config["output_directory"] += "_smoke"
    return config


def scenarios(config, baseline_parameters):
    """Resolve absolute levels without mutating the reference or coupling parameters."""
    result = [
        {
            "name": "baseline",
            "parameter": None,
            "value": None,
            "baseline_value": None,
            "multiplier": None,
            "generation": deepcopy(baseline_parameters),
        }
    ]
    names = {"baseline"}
    for item in config["perturbations"]:
        parameter = item["parameter"]
        keys = parameter.split(".")
        source = baseline_parameters
        for key in keys:
            source = source[key]
        kind = "multipliers" if "multipliers" in item else "values"
        for level in item[kind]:
            value = float(source * level if kind == "multipliers" else level)
            if (
                not np.isfinite(value)
                or value < 0
                or (value == 0 and parameter != "relaxation")
            ):
                raise ValueError(
                    f"Invalid value for {parameter}: {value}; only relaxation may be zero"
                )
            if value == source:
                raise ValueError(
                    f"{parameter} level {level} duplicates the baseline control"
                )
            generation = deepcopy(baseline_parameters)
            target = generation
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            # Includes EpiLink's coupled incubation/latent-shape requirements.
            natural_history(generation)
            suffix = f"x{level:g}" if kind == "multipliers" else f"{level:g}"
            name = f"{parameter.replace('.', '_')}_{suffix}"
            if name in names:
                raise ValueError(f"Duplicate scenario name: {name}")
            names.add(name)
            result.append(
                {
                    "name": name,
                    "parameter": parameter,
                    "value": value,
                    "baseline_value": float(source),
                    "multiplier": float(level) if kind == "multipliers" else None,
                    "generation": generation,
                }
            )
    return result
