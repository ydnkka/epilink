from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, MutableMapping

BASELINE_SCENARIO_NAME = "baseline"

DEFAULT_TARGET = ("ad(0)", "ca(0,0)")
DEFAULT_SEED = 12345
DEFAULT_MAXIMUM_DEPTH = 0
DEFAULT_NUM_SIMULATIONS = 10_000

PAIRWISE_BOTH_SAMPLED_COLUMN = "BothSampled"
PAIRWISE_RELATED_COLUMN = "IsRelated"
PAIRWISE_CASE_A_COLUMN = "CaseA"
PAIRWISE_CASE_B_COLUMN = "CaseB"
PAIRWISE_TEMPORAL_DISTANCE_COLUMN = "SamplingDateDistanceDays"
PAIRWISE_DETERMINISTIC_DISTANCE_COLUMN = "DeterministicDistance"
PAIRWISE_STOCHASTIC_DISTANCE_COLUMN = "StochasticDistance"

PAIRWISE_DISTANCE_COLUMNS = (
    PAIRWISE_DETERMINISTIC_DISTANCE_COLUMN,
    PAIRWISE_STOCHASTIC_DISTANCE_COLUMN,
)

SCENARIO_PARAMETER_FIELDS = (
    "incubation_shape",
    "incubation_scale",
    "testing_delay_shape",
    "testing_delay_scale",
    "substitution_rate",
    "relaxation",
)

MODEL_KEYS = ("EDD", "EDS", "ESD", "ESS", "LD", "LS")
"""Canonical model abbreviations.

EDD — EpiLink, deterministic inference, deterministic data
EDS — EpiLink, deterministic inference, stochastic data
ESD — EpiLink, stochastic inference, deterministic data
ESS — EpiLink, stochastic inference, stochastic data
LD  — Logistic regression on deterministic distances
LS  — Logistic regression on stochastic distances
"""

EPILINK_SPECS: tuple[dict[str, str], ...] = (
    {
        "key": "EDD",
        "mutation_process": "deterministic",
        "distance_col": PAIRWISE_DETERMINISTIC_DISTANCE_COLUMN,
    },
    {
        "key": "EDS",
        "mutation_process": "deterministic",
        "distance_col": PAIRWISE_STOCHASTIC_DISTANCE_COLUMN,
    },
    {
        "key": "ESD",
        "mutation_process": "stochastic",
        "distance_col": PAIRWISE_DETERMINISTIC_DISTANCE_COLUMN,
    },
    {
        "key": "ESS",
        "mutation_process": "stochastic",
        "distance_col": PAIRWISE_STOCHASTIC_DISTANCE_COLUMN,
    },
)

LOGIT_SPECS: tuple[dict[str, str], ...] = (
    {"key": "LD", "distance_col": PAIRWISE_DETERMINISTIC_DISTANCE_COLUMN},
    {"key": "LS", "distance_col": PAIRWISE_STOCHASTIC_DISTANCE_COLUMN},
)

SCORE_METADATA: dict[str, dict[str, str]] = {
    "EDD": {
        "score_family": "epilink_score",
        "inference_process": "deterministic",
        "data_process": "deterministic",
    },
    "EDS": {
        "score_family": "epilink_score",
        "inference_process": "deterministic",
        "data_process": "stochastic",
    },
    "ESD": {
        "score_family": "epilink_score",
        "inference_process": "stochastic",
        "data_process": "deterministic",
    },
    "ESS": {
        "score_family": "epilink_score",
        "inference_process": "stochastic",
        "data_process": "stochastic",
    },
    "LD": {
        "score_family": "logit_probability",
        "inference_process": "not_applicable",
        "data_process": "deterministic",
    },
    "LS": {
        "score_family": "logit_probability",
        "inference_process": "not_applicable",
        "data_process": "stochastic",
    },
}

UNKNOWN_SCORE_METADATA = {
    "score_family": "other",
    "inference_process": "unknown",
    "data_process": "unknown",
}


def gamma_mean_cv_to_shape_scale(mean: float, cv: float) -> dict[str, float]:
    if mean <= 0:
        raise ValueError(f"Gamma mean must be > 0, got {mean}")
    if cv <= 0:
        raise ValueError(f"Gamma CV must be > 0, got {cv}")
    shape = 1.0 / (cv**2)
    scale = mean / shape
    return {"shape": shape, "scale": scale}


def resolve_target_labels(raw: Any = DEFAULT_TARGET) -> tuple[str, ...]:
    labels = DEFAULT_TARGET if raw is None else raw
    labels = (labels,) if isinstance(labels, str) else tuple(str(value) for value in labels)
    if not labels or any(not label.strip() for label in labels):
        raise ValueError("target must contain at least one non-empty label.")
    return labels


def expand_baseline_parameters(
    baseline: Mapping[str, Any],
    fixed_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    fixed = deepcopy(dict(fixed_parameters))
    incubation = gamma_mean_cv_to_shape_scale(
        float(baseline["incubation"]["mean"]),
        float(baseline["incubation"]["cv"]),
    )
    testing_delay = gamma_mean_cv_to_shape_scale(
        float(baseline["testing_delay"]["mean"]),
        float(baseline["testing_delay"]["cv"]),
    )

    parameters = {
        **fixed,
        "incubation_shape": incubation["shape"],
        "incubation_scale": incubation["scale"],
        "testing_delay_shape": testing_delay["shape"],
        "testing_delay_scale": testing_delay["scale"],
        "substitution_rate": float(baseline["substitution_rate"]),
        "relaxation": float(baseline["relaxation"]),
    }
    if "target" in parameters:
        parameters["target"] = resolve_target_labels(parameters["target"])
    return parameters


def copy_scenario_parameters(
    target: MutableMapping[str, Any],
    source: Mapping[str, Any],
    *,
    parameter_fields: tuple[str, ...] = SCENARIO_PARAMETER_FIELDS,
) -> None:
    for field in parameter_fields:
        target[field] = source[field]


def parameter_columns(parameters: Mapping[str, Any], *, prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{field}": parameters[field] for field in SCENARIO_PARAMETER_FIELDS}


def score_metadata(
    model_key: str,
    *,
    logistic_training_fraction: float,
) -> dict[str, str | float]:
    metadata = SCORE_METADATA.get(model_key, UNKNOWN_SCORE_METADATA)
    training_fraction = (
        logistic_training_fraction
        if metadata["score_family"] == "logit_probability"
        else float("nan")
    )
    return {**metadata, "training_fraction": training_fraction}
