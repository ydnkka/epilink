"""EpiLink model construction and logistic-regression scoring helpers.

Exposes :func:`build_linkage_model` / :func:`build_linkage_models` for
constructing deterministic and stochastic EpiLink scorers from a parameter
mapping, :func:`build_natural_history_parameters` for translating a flat
parameter dict into a ``NaturalHistoryParameters`` object, and
:func:`predict_logistic_scores` for training a logistic classifier on a
fraction of the data and returning probability scores.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from specs import (
    DEFAULT_MAXIMUM_DEPTH,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_TARGET,
    expand_baseline_parameters,
    resolve_target_labels,
)

from epilink import EpiLink, InfectiousnessToTransmission, NaturalHistoryParameters


def build_natural_history_parameters(parameters: Mapping[str, Any]) -> NaturalHistoryParameters:
    """Construct a ``NaturalHistoryParameters`` object from a flat parameter mapping.

    Missing keys fall back to hard-coded defaults matching the baseline config.
    """
    return NaturalHistoryParameters(
        incubation_shape=float(parameters.get("incubation_shape", 5.807)),
        incubation_scale=float(parameters.get("incubation_scale", 0.948)),
        latent_shape=float(parameters.get("latent_shape", 3.38)),
        symptomatic_rate=float(parameters.get("symptomatic_rate", 0.37)),
        symptomatic_shape=float(parameters.get("symptomatic_shape", 1.0)),
        transmission_rate_ratio=float(parameters.get("transmission_rate_ratio", 2.29)),
        testing_delay_shape=float(parameters.get("testing_delay_shape", 1.0)),
        testing_delay_scale=float(parameters.get("testing_delay_scale", 1.0)),
        substitution_rate=float(parameters.get("substitution_rate", 1e-3)),
        relaxation=float(parameters.get("relaxation", 0.33)),
        genome_length=int(parameters.get("genome_length", 29903)),
    )


def nhp_from_baseline(
    baseline: Mapping[str, Any],
    fixed_parameters: Mapping[str, Any],
) -> NaturalHistoryParameters:
    """Expand baseline + fixed parameters and return a ``NaturalHistoryParameters`` object."""
    return build_natural_history_parameters(expand_baseline_parameters(baseline, fixed_parameters))


def build_linkage_model(
    parameters: Mapping[str, Any],
    *,
    mutation_process: str,
    rng_seed: int,
) -> EpiLink:
    """Build a single EpiLink scorer for the given *mutation_process* and parameters.

    Parameters
    ----------
    parameters : Mapping
        Flat inference-parameter mapping (shape/scale values for incubation,
        testing delay, substitution rate, etc.).
    mutation_process : str
        ``"deterministic"`` or ``"stochastic"``.
    rng_seed : int
        RNG seed forwarded to ``InfectiousnessToTransmission``.

    Returns
    -------
    EpiLink
        Configured scorer ready to call ``.score_target()``.
    """
    transmission_profile = InfectiousnessToTransmission(
        parameters=build_natural_history_parameters(parameters),
        rng_seed=rng_seed,
    )
    return EpiLink(
        mutation_process=mutation_process,
        transmission_profile=transmission_profile,
        maximum_depth=int(parameters.get("maximum_depth", DEFAULT_MAXIMUM_DEPTH)),
        mc_samples=int(parameters.get("num_simulations", DEFAULT_NUM_SIMULATIONS)),
        target=resolve_target_labels(parameters.get("target", DEFAULT_TARGET)),
    )


def build_linkage_models(
    parameters: Mapping[str, Any],
    *,
    rng_seed: int,
) -> dict[str, EpiLink]:
    """Build both deterministic and stochastic EpiLink scorers from a parameter mapping.

    Returns
    -------
    dict[str, EpiLink]
        Keys are ``"deterministic"`` and ``"stochastic"``.
    """
    return {
        mutation_process: build_linkage_model(
            parameters,
            mutation_process=mutation_process,
            rng_seed=rng_seed,
        )
        for mutation_process in ("deterministic", "stochastic")
    }


def build_training_indices(y: np.ndarray, train_size: int, rng_seed: int) -> np.ndarray:
    """Sample stratified training indices of size *train_size*, keeping ≥ 1 example per class."""

    classes = np.unique(y)
    if len(classes) < 2:
        return np.array([], dtype=int)

    train_size = int(min(max(train_size, len(classes)), len(y)))
    rng = np.random.default_rng(rng_seed)
    selected: list[int] = []
    remaining: list[int] = []

    for class_label in classes:
        class_indices = np.flatnonzero(y == class_label)
        chosen = int(rng.choice(class_indices))
        selected.append(chosen)
        remaining.extend(int(index) for index in class_indices if index != chosen)

    extra_needed = train_size - len(selected)
    if extra_needed > 0:
        extra_pool = np.asarray(remaining, dtype=int)
        if extra_needed >= len(extra_pool):
            extra = extra_pool
        else:
            extra = rng.choice(extra_pool, size=extra_needed, replace=False)
        selected.extend(int(index) for index in np.asarray(extra, dtype=int))

    return np.asarray(sorted(selected), dtype=int)


def predict_logistic_scores(
    feature_matrix: np.ndarray,
    y: np.ndarray,
    *,
    training_fraction: float,
    rng_seed: int,
    predict_feature_matrix: np.ndarray | None = None,
    return_classifier: bool = False,
) -> tuple[np.ndarray, LogisticRegression | None]:
    """Fit logistic regression when feasible, otherwise return a constant prior score."""

    prediction_features = (
        np.asarray(feature_matrix)
        if predict_feature_matrix is None
        else np.asarray(predict_feature_matrix)
    )
    if len(y) == 0:
        return np.full(len(prediction_features), np.nan, dtype=float), None

    prevalence = float(y.mean())
    if len(np.unique(y)) < 2:
        return np.full(len(prediction_features), prevalence, dtype=float), None

    classifier = LogisticRegression(solver="lbfgs", max_iter=200)
    if float(training_fraction) >= 1.0:
        classifier.fit(feature_matrix, y)
        return classifier.predict_proba(prediction_features)[:, 1], (
            classifier if return_classifier else None
        )

    requested_train_size = int(np.ceil(len(y) * float(training_fraction)))
    train_indices = build_training_indices(y, requested_train_size, rng_seed)
    if len(train_indices) == 0 or len(np.unique(y[train_indices])) < 2:
        return np.full(len(prediction_features), prevalence, dtype=float), None

    classifier.fit(feature_matrix[train_indices], y[train_indices])
    return classifier.predict_proba(prediction_features)[:, 1], (
        classifier if return_classifier else None
    )
