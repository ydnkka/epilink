"""Configuration loading, path resolution, logging setup, and scenario/run construction.

This module is the shared foundation for all evaluation workflow modules. It provides:

- YAML config loading with directory-relative path resolution.
- Helpers to resolve configured paths for inputs and outputs.
- ``configure_logging``: consistent log format (timestamp + level + logger name) with
  optional append-mode file handler for a single unified pipeline log.
- ``get_pipeline_log_path``: reads the unified log path from the loaded config.
- Scenario and run-spec builders used by the synthetic experiments workflow.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from specs import (
    BASELINE_SCENARIO_NAME,
    copy_scenario_parameters,
    expand_baseline_parameters,
)

MODULE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = MODULE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent


_CONFIG_DIR_KEY = "__config_dir__"
_MISSING = object()


@dataclass
class ScenarioSpec:
    """Parameters for a single perturbation scenario."""

    name: str
    perturbed_parameter: str
    perturbation_multiplier: float
    generation_parameters: dict[str, Any]


@dataclass
class RunSpec:
    """Full specification for one evaluation run (condition × scenario)."""

    condition: str
    scenario_name: str
    tree_path: str
    generation_parameters: dict[str, Any]
    inference_parameters: dict[str, Any]
    logit_training_source: str
    perturbed_parameter: str
    perturbation_multiplier: float


def project_root() -> Path:
    """Return the evaluation project root."""

    return PROJECT_ROOT


def source_root() -> Path:
    """Return the directory that contains the evaluation modules."""

    return MODULE_ROOT


def resolve_path(path_like: str | Path, *, root: Path | None = None) -> Path:
    """Resolve a path relative to a supplied root or the evaluation project root."""

    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = Path(root).expanduser().resolve() if root is not None else project_root()
    return (base / path).resolve()


def load_config(path_like: str | Path = "config.yaml") -> dict[str, Any]:
    """Load a YAML config file and retain the directory it came from."""

    config_path = Path(path_like).expanduser()
    if not config_path.is_absolute():
        cwd_candidate = (Path.cwd() / config_path).resolve()
        config_path = cwd_candidate if cwd_candidate.exists() else resolve_path(config_path)

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a mapping in {config_path}, found {type(loaded).__name__}.")

    config = deepcopy(loaded)
    config[_CONFIG_DIR_KEY] = str(config_path.parent)
    return config


def _config_base_dir(config: dict[str, Any]) -> Path:
    """Return the directory associated with a loaded config mapping."""

    config_dir = config.get(_CONFIG_DIR_KEY)
    if config_dir is None:
        return project_root()
    return Path(config_dir).resolve()


def resolve_config_path(config: dict[str, Any], path_like: str | Path) -> Path:
    """Resolve a path relative to the config file location, even if it does not exist yet."""

    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_config_base_dir(config) / path).resolve()


def get_config_value(
    config: Mapping[str, Any],
    dotted_path: str,
    *,
    default: Any = _MISSING,
) -> Any:
    """Return a nested config value addressed by dotted keys."""

    current: Any = config
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            if default is _MISSING:
                raise KeyError(f"Missing config value: {dotted_path}")
            return default
        current = current[part]
    return current


def resolve_configured_path(
    config: dict[str, Any],
    dotted_path: str,
    *,
    default: Any = _MISSING,
) -> Path:
    """Resolve a configured path relative to the config file location."""

    return resolve_config_path(config, get_config_value(config, dotted_path, default=default))


def outputs_root(config: dict[str, Any]) -> Path:
    """Return the configured root directory for generated evaluation artifacts."""

    return resolve_configured_path(config, "outputs.directory")


def resolve_output_path(config: dict[str, Any], path_like: str | Path) -> Path:
    """Resolve an output path relative to the configured outputs root."""

    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (outputs_root(config) / path).resolve()


def resolve_configured_output_path(
    config: dict[str, Any],
    dotted_path: str,
    *,
    default: Any = _MISSING,
) -> Path:
    """Resolve a configured output path relative to the configured outputs root."""

    return resolve_output_path(config, get_config_value(config, dotted_path, default=default))


_LOG_FORMAT = "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> None:
    """Configure consistent logging for all workflow modules.

    Sets up a ``StreamHandler`` on stderr and, optionally, a ``FileHandler``
    that appends to a shared pipeline log.  Safe to call multiple times in the
    same process — duplicate handlers are not added.

    Parameters
    ----------
    level : int, optional
        Logging threshold, by default ``logging.INFO``.
    log_file : str or Path, optional
        Path to the unified pipeline log file.  If given, records are appended
        to this file in addition to being written to stderr.  The parent
        directory is created automatically if it does not exist.
    """
    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    root = logging.getLogger()
    root.setLevel(level)

    existing_types = {type(h) for h in root.handlers}

    if logging.StreamHandler not in existing_types:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file).resolve()
        existing_paths = {
            Path(getattr(h, "baseFilename", "")).resolve()
            for h in root.handlers
            if isinstance(h, logging.FileHandler)
        }
        if log_path not in existing_paths:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)


def get_pipeline_log_path(config: dict[str, Any]) -> Path | None:
    """Return the configured unified pipeline log path, or *None* if not set.

    Reads ``outputs.logs.pipeline_log`` from the loaded config and resolves it
    relative to the configured outputs root directory.

    Parameters
    ----------
    config : dict
        Loaded config mapping as returned by :func:`load_config`.

    Returns
    -------
    Path or None
        Absolute path to the pipeline log file, or ``None`` when the key is
        absent or its value is falsy.
    """
    raw = get_config_value(config, "outputs.logs.pipeline_log", default=None)
    if not raw:
        return None
    return resolve_output_path(config, raw)


def _check_constraint(name: str, value: float, constraint: str) -> None:
    """Raise ``ValueError`` if *value* violates the named constraint string."""
    if constraint == "> 0" and not (value > 0):
        raise ValueError(f"{name} must be > 0, got {value}")
    if constraint == ">= 0" and not (value >= 0):
        raise ValueError(f"{name} must be >= 0, got {value}")


def _apply_dotted_update(obj: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set a nested dict value addressed by a dotted key path in-place."""
    parts = dotted_path.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


def _get_dotted(obj: dict[str, Any], dotted_path: str) -> Any:
    """Retrieve a nested dict value addressed by a dotted key path."""
    cur = obj
    for part in dotted_path.split("."):
        cur = cur[part]
    return cur


def resolve_generation_baseline_parameters(config: dict[str, Any]) -> dict[str, Any]:
    """Expand the generation baseline section with fixed parameters from config."""
    return expand_baseline_parameters(config["generation_baseline"], config["fixed_parameters"])


def resolve_inference_baseline_parameters(config: dict[str, Any]) -> dict[str, Any]:
    """Expand the inference baseline section with fixed parameters from config."""
    return expand_baseline_parameters(config["inference_baseline"], config["fixed_parameters"])


def generate_scenarios(config: dict[str, Any]) -> dict[str, ScenarioSpec]:
    """Build all scenario specs from the config perturbation grid.

    The baseline scenario is included first (if ``design.include_baseline_scenario``
    is true), followed by one spec per perturbation × multiplier combination.
    Parameter constraints are checked and a ``ValueError`` is raised on violation.

    Parameters
    ----------
    config : dict
        Loaded config mapping as returned by :func:`load_config`.

    Returns
    -------
    dict[str, ScenarioSpec]
        Ordered mapping from scenario name to its :class:`ScenarioSpec`.
    """
    constraints = config["parameter_constraints"]
    scales = config["perturbation_scales"]
    perturbations = config["perturbations"]

    scenarios: dict[str, ScenarioSpec] = {}

    if config["design"].get("include_baseline_scenario", True):
        scenarios[BASELINE_SCENARIO_NAME] = ScenarioSpec(
            name=BASELINE_SCENARIO_NAME,
            perturbed_parameter="none",
            perturbation_multiplier=1.0,
            generation_parameters=resolve_generation_baseline_parameters(config),
        )

    for perturbation_name, perturbation_def in perturbations.items():
        scale_name = perturbation_def["scale"]
        target = perturbation_def["target"]

        for multiplier in scales[scale_name]:
            scenario_generation = deepcopy(config["generation_baseline"])
            current_value = _get_dotted(scenario_generation, target)
            new_value = current_value * float(multiplier)
            _apply_dotted_update(scenario_generation, target, new_value)

            flat_checks = {
                "incubation_mean": float(scenario_generation["incubation"]["mean"]),
                "incubation_cv": float(scenario_generation["incubation"]["cv"]),
                "testing_delay_mean": float(scenario_generation["testing_delay"]["mean"]),
                "testing_delay_cv": float(scenario_generation["testing_delay"]["cv"]),
                "substitution_rate": float(scenario_generation["substitution_rate"]),
                "relaxation": float(scenario_generation["relaxation"]),
            }

            for name, value in flat_checks.items():
                _check_constraint(name, value, constraints[name])

            scenario_name = f"{perturbation_name}_{multiplier:.2f}"
            scenarios[scenario_name] = ScenarioSpec(
                name=scenario_name,
                perturbed_parameter=perturbation_name,
                perturbation_multiplier=float(multiplier),
                generation_parameters=expand_baseline_parameters(
                    scenario_generation,
                    config["fixed_parameters"],
                ),
            )

    return scenarios


def build_run_specs(config: dict[str, Any]) -> list[RunSpec]:
    """Enumerate all (condition × scenario) run specifications from config.

    For each condition listed in ``execution.evaluate_conditions`` and each
    scenario produced by :func:`generate_scenarios`, one :class:`RunSpec` is
    created with the appropriate generation and inference parameter sets.

    Parameters
    ----------
    config : dict
        Loaded config mapping as returned by :func:`load_config`.

    Returns
    -------
    list[RunSpec]
        Flat list of run specs ready for evaluation.
    """
    scenarios = generate_scenarios(config)
    tree_path = resolve_configured_output_path(config, "outputs.scovmod.tree_path")
    inference_baseline = resolve_inference_baseline_parameters(config)

    runs: list[RunSpec] = []

    for condition_name in config["execution"]["evaluate_conditions"]:
        condition = config["conditions"][condition_name]

        for scenario_name, scenario in scenarios.items():
            if condition["generation_source"] == "scenario":
                generation_params = deepcopy(scenario.generation_parameters)
            elif condition["generation_source"] == "baseline":
                baseline_scenario = scenarios[BASELINE_SCENARIO_NAME]
                generation_params = deepcopy(baseline_scenario.generation_parameters)
            else:
                raise ValueError(f"Unknown generation_source: {condition['generation_source']}")

            if condition["inference_source"] == "scenario":
                inference_params = deepcopy(inference_baseline)
                copy_scenario_parameters(inference_params, generation_params)
            elif condition["inference_source"] == "baseline":
                inference_params = deepcopy(inference_baseline)
            else:
                raise ValueError(f"Unknown inference_source: {condition['inference_source']}")

            runs.append(
                RunSpec(
                    condition=condition_name,
                    scenario_name=scenario_name,
                    tree_path=str(tree_path),
                    generation_parameters=generation_params,
                    inference_parameters=inference_params,
                    logit_training_source=condition["logit_training_source"],
                    perturbed_parameter=scenario.perturbed_parameter,
                    perturbation_multiplier=scenario.perturbation_multiplier,
                )
            )

    return runs
