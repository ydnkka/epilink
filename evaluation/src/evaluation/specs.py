"""Shared constants, column names, model specs, and parameter-expansion helpers.

This module is imported by every other evaluation module and intentionally has
no dependencies within the package so it can always be imported first.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from typing import Any

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

MODEL_LABELS = {
    "EDD": "EpiLink deterministic, deterministic data",
    "EDS": "EpiLink deterministic, stochastic data",
    "ESD": "EpiLink stochastic, deterministic data",
    "ESS": "EpiLink stochastic, stochastic data",
    "LD": "Logistic inference, deterministic data",
    "LS": "Logistic inference, stochastic data",
}

MODEL_KEYS = tuple(MODEL_LABELS.keys())

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
    """Convert Gamma mean and coefficient of variation to shape and scale parameters.

    Parameters
    ----------
    mean : float
        Gamma distribution mean (must be > 0).
    cv : float
        Coefficient of variation, i.e. SD / mean (must be > 0).

    Returns
    -------
    dict[str, float]
        Mapping with keys ``"shape"`` and ``"scale"``.
    """
    if mean <= 0:
        raise ValueError(f"Gamma mean must be > 0, got {mean}")
    if cv <= 0:
        raise ValueError(f"Gamma CV must be > 0, got {cv}")
    shape = 1.0 / (cv**2)
    scale = mean / shape
    return {"shape": shape, "scale": scale}


def resolve_target_labels(raw: Any = DEFAULT_TARGET) -> tuple[str, ...]:
    """Normalise a raw target spec into a non-empty tuple of label strings.

    Parameters
    ----------
    raw : str, iterable, or None
        Raw target value from a parameter mapping.  ``None`` falls back to
        :data:`DEFAULT_TARGET`.

    Returns
    -------
    tuple[str, ...]
        Non-empty tuple of stripped, non-empty label strings.
    """
    labels = DEFAULT_TARGET if raw is None else raw
    labels = (labels,) if isinstance(labels, str) else tuple(str(value) for value in labels)
    if not labels or any(not label.strip() for label in labels):
        raise ValueError("target must contain at least one non-empty label.")
    return labels


def expand_baseline_parameters(
    baseline: Mapping[str, Any],
    fixed_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand a baseline parameter block into a flat inference-ready parameter dict.

    Converts ``incubation`` and ``testing_delay`` mean/CV pairs to Gamma
    shape/scale, merges fixed parameters, and resolves the ``target`` labels.

    Parameters
    ----------
    baseline : Mapping
        Baseline block with nested ``incubation``, ``testing_delay``,
        ``substitution_rate``, and ``relaxation`` entries.
    fixed_parameters : Mapping
        Additional parameters merged verbatim into the output dict.

    Returns
    -------
    dict[str, Any]
        Flat parameter mapping ready for :func:`build_natural_history_parameters`.
    """
    fixed = deepcopy(dict(fixed_parameters))
    incubation = gamma_mean_cv_to_shape_scale(
        float(baseline["incubation"]["mean"]),
        float(baseline["incubation"]["cv"]),
    )
    testing_delay = gamma_mean_cv_to_shape_scale(
        float(baseline["testing_delay"]["mean"]),
        float(baseline["testing_delay"]["cv"]),
    )

    parameters: dict[str, Any] = {
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
    """Copy scenario parameter fields from *source* into *target* in-place.

    Parameters
    ----------
    target : MutableMapping
        Destination mapping to update (typically an inference parameter dict).
    source : Mapping
        Source mapping to copy from (typically a generation parameter dict).
    parameter_fields : tuple[str, ...], optional
        Fields to copy.  Defaults to :data:`SCENARIO_PARAMETER_FIELDS`.
    """
    for field in parameter_fields:
        target[field] = source[field]


def parameter_columns(parameters: Mapping[str, Any], *, prefix: str) -> dict[str, Any]:
    """Build a flat column dict from scenario parameter fields with a given prefix.

    Parameters
    ----------
    parameters : Mapping
        Parameter mapping containing :data:`SCENARIO_PARAMETER_FIELDS` keys.
    prefix : str
        String prepended to each field name (e.g. ``"generation"`` or
        ``"inference"``).

    Returns
    -------
    dict[str, Any]
        Dict keyed ``"<prefix>_<field>"`` for each field in
        :data:`SCENARIO_PARAMETER_FIELDS`.
    """
    return {f"{prefix}_{field}": parameters[field] for field in SCENARIO_PARAMETER_FIELDS}


def score_metadata(
    model_key: str,
    *,
    logistic_training_fraction: float,
) -> dict[str, str | float]:
    """Return metadata annotations for a model key, including training fraction.

    Parameters
    ----------
    model_key : str
        One of the canonical model keys (``"EDD"``, ``"EDS"``, etc.).
    logistic_training_fraction : float
        Training fraction used for logistic models; stored as ``NaN`` for
        EpiLink models where this concept does not apply.

    Returns
    -------
    dict[str, str | float]
        Metadata dict with keys ``score_family``, ``inference_process``,
        ``data_process``, and ``training_fraction``.
    """
    metadata = SCORE_METADATA.get(model_key, UNKNOWN_SCORE_METADATA)
    training_fraction = (
        logistic_training_fraction
        if metadata["score_family"] == "logit_probability"
        else float("nan")
    )
    return {**metadata, "training_fraction": training_fraction}
