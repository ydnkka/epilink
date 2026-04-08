from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


_CONFIG_DIR_KEY = "__config_dir__"


@dataclass
class ScenarioSpec:
    name: str
    perturbed_parameter: str
    perturbation_multiplier: float
    generation_parameters: Dict[str, Any]


@dataclass
class RunSpec:
    condition: str
    scenario_name: str
    tree_path: str
    generation_parameters: Dict[str, Any]
    inference_parameters: Dict[str, Any]
    logit_training_source: str
    perturbed_parameter: str
    perturbation_multiplier: float


def project_root() -> Path:
    """Return the evaluation project root."""

    return Path(__file__).resolve().parents[2]


def source_root() -> Path:
    """Return the directory that contains the evaluation modules."""

    return Path(__file__).resolve().parent


def resolve_path(path_like: str | Path, *, root: Path | None = None) -> Path:
    """Resolve a path against likely evaluation roots and prefer existing targets."""

    path = Path(path_like)
    if path.is_absolute():
        return path

    candidate_roots = []
    if root is not None:
        candidate_roots.append(Path(root).resolve())
    candidate_roots.extend((source_root(), project_root(), Path.cwd().resolve()))

    seen: set[Path] = set()
    for base in candidate_roots:
        if base in seen:
            continue
        seen.add(base)
        resolved = (base / path).resolve()
        if resolved.exists():
            return resolved

    return (source_root() / path).resolve()


def load_config(path_like: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load a YAML config file and retain the directory it came from."""

    config_path = Path(path_like)
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


def _config_base_dir(config: Dict[str, Any]) -> Path:
    """Return the directory associated with a loaded config mapping."""

    config_dir = config.get(_CONFIG_DIR_KEY)
    if config_dir is None:
        return project_root()
    return Path(config_dir).resolve()


def gamma_mean_cv_to_shape_scale(mean: float, cv: float) -> Dict[str, float]:
    if mean <= 0:
        raise ValueError(f"Gamma mean must be > 0, got {mean}")
    if cv <= 0:
        raise ValueError(f"Gamma CV must be > 0, got {cv}")
    shape = 1.0 / (cv ** 2)
    scale = mean / shape
    return {"shape": shape, "scale": scale}


def _check_constraint(name: str, value: float, constraint: str) -> None:
    if constraint == "> 0" and not (value > 0):
        raise ValueError(f"{name} must be > 0, got {value}")
    if constraint == ">= 0" and not (value >= 0):
        raise ValueError(f"{name} must be >= 0, got {value}")


def _apply_dotted_update(obj: Dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


def _get_dotted(obj: Dict[str, Any], dotted_path: str) -> Any:
    cur = obj
    for part in dotted_path.split("."):
        cur = cur[part]
    return cur


def _resolve_generation_baseline(config: Dict[str, Any]) -> Dict[str, Any]:
    fixed = deepcopy(config["fixed_parameters"])
    gen = deepcopy(config["generation_baseline"])

    inc = gamma_mean_cv_to_shape_scale(gen["incubation"]["mean"], gen["incubation"]["cv"])
    td = gamma_mean_cv_to_shape_scale(gen["testing_delay"]["mean"], gen["testing_delay"]["cv"])

    return {
        **fixed,
        "incubation_shape": inc["shape"],
        "incubation_scale": inc["scale"],
        "testing_delay_shape": td["shape"],
        "testing_delay_scale": td["scale"],
        "substitution_rate": float(gen["substitution_rate"]),
        "relaxation": float(gen["relaxation"]),
    }


def _resolve_inference_baseline(config: Dict[str, Any]) -> Dict[str, Any]:
    fixed = deepcopy(config["fixed_parameters"])
    inf = deepcopy(config["inference_baseline"])

    inc = gamma_mean_cv_to_shape_scale(inf["incubation"]["mean"], inf["incubation"]["cv"])
    td = gamma_mean_cv_to_shape_scale(inf["testing_delay"]["mean"], inf["testing_delay"]["cv"])

    target = tuple(t for t in fixed["target"])
    fixed["target"] = target

    return {
        **fixed,
        "incubation_shape": inc["shape"],
        "incubation_scale": inc["scale"],
        "testing_delay_shape": td["shape"],
        "testing_delay_scale": td["scale"],
        "substitution_rate": float(inf["substitution_rate"]),
        "relaxation": float(inf["relaxation"]),
    }


def generate_scenarios(config: Dict[str, Any]) -> Dict[str, ScenarioSpec]:
    constraints = config["parameter_constraints"]
    scales = config["perturbation_scales"]
    perturbations = config["perturbations"]

    scenarios: Dict[str, ScenarioSpec] = {}

    if config["design"].get("include_baseline_scenario", True):
        scenarios["baseline"] = ScenarioSpec(
            name="baseline",
            perturbed_parameter="none",
            perturbation_multiplier=1.0,
            generation_parameters=_resolve_generation_baseline(config),
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

            tmp_cfg = deepcopy(config)
            tmp_cfg["generation_baseline"] = scenario_generation

            scenarios[f"{perturbation_name}_{multiplier:.2f}"] = ScenarioSpec(
                name=f"{perturbation_name}_{multiplier:.2f}",
                perturbed_parameter=perturbation_name,
                perturbation_multiplier=float(multiplier),
                generation_parameters=_resolve_generation_baseline(tmp_cfg),
            )

    return scenarios


def build_run_specs(config: Dict[str, Any]) -> List[RunSpec]:
    scenarios = generate_scenarios(config)
    tree_path = str(resolve_path(config["paths"]["tree_path"], root=_config_base_dir(config)))
    inference_baseline = _resolve_inference_baseline(config)

    runs: List[RunSpec] = []

    for condition_name in config["execution"]["evaluate_conditions"]:
        condition = config["conditions"][condition_name]

        for scenario_name, scenario in scenarios.items():
            if condition["generation_source"] == "scenario":
                generation_params = deepcopy(scenario.generation_parameters)
            elif condition["generation_source"] == "baseline":
                baseline_scenario = scenarios["baseline"]
                generation_params = deepcopy(baseline_scenario.generation_parameters)
            else:
                raise ValueError(f"Unknown generation_source: {condition['generation_source']}")

            if condition["inference_source"] == "scenario":
                inference_params = deepcopy(inference_baseline)

                # matched means copy the perturbed generation-side parameters into inference
                for key in (
                    "incubation_shape",
                    "incubation_scale",
                    "testing_delay_shape",
                    "testing_delay_scale",
                    "substitution_rate",
                    "relaxation",
                ):
                    inference_params[key] = generation_params[key]

            elif condition["inference_source"] == "baseline":
                inference_params = deepcopy(inference_baseline)
            else:
                raise ValueError(f"Unknown inference_source: {condition['inference_source']}")

            runs.append(
                RunSpec(
                    condition=condition_name,
                    scenario_name=scenario_name,
                    tree_path=tree_path,
                    generation_parameters=generation_params,
                    inference_parameters=inference_params,
                    logit_training_source=condition["logit_training_source"],
                    perturbed_parameter=scenario.perturbed_parameter,
                    perturbation_multiplier=scenario.perturbation_multiplier,
                )
            )

    return runs
