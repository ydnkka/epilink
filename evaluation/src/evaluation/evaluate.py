"""Core scenario evaluation logic for the synthetic experiment pipeline.

The public entry-point is :func:`evaluate_scenario`, which simulates an epidemic
on the pre-built transmission tree, scores all sampled pairs with EpiLink and
logistic models, then computes average-precision, best-F1, and resolution
stability metrics for each model.

Result types :class:`ModelResult` and :class:`ScenarioResult` are frozen
dataclasses so they can safely cross process boundaries.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from leiden import build_weighted_graph, run_leiden_partition
from metrics import bcubed_scores
from models import (
    build_linkage_models,
    build_natural_history_parameters,
    predict_logistic_scores,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from specs import (
    EPILINK_SPECS,
    LOGIT_SPECS,
    MODEL_KEYS,
    PAIRWISE_BOTH_SAMPLED_COLUMN,
    PAIRWISE_CASE_A_COLUMN,
    PAIRWISE_CASE_B_COLUMN,
    PAIRWISE_DETERMINISTIC_DISTANCE_COLUMN,
    PAIRWISE_RELATED_COLUMN,
    PAIRWISE_STOCHASTIC_DISTANCE_COLUMN,
    PAIRWISE_TEMPORAL_DISTANCE_COLUMN,
    SCORE_METADATA,
)

from epilink import (
    InfectiousnessToTransmission,
    build_pairwise_case_table,
    simulate_epidemic_dates,
    simulate_genomic_sequences,
)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelResult:
    """Per-model evaluation metrics for one scenario run."""

    ap: float
    ap_loss: float | None
    best_f1: float
    f1_loss: float | None
    mean_stability: float
    std_stability: float


@dataclass
class ScenarioResult:
    """Aggregated evaluation results across all models for one scenario run."""

    scenario_name: str
    n_pairs: int
    prevalence: float
    models: dict[str, ModelResult] = field(default_factory=dict)


_EPILINK_SPECS = EPILINK_SPECS
_LOGIT_SPECS = LOGIT_SPECS
_SCORE_METADATA = SCORE_METADATA


# ---------------------------------------------------------------------------
# Private helpers — simulation
# ---------------------------------------------------------------------------


@cache
def _load_tree_template(tree_path: str) -> nx.DiGraph:
    """Load and cache a GML transmission tree (keyed by path string)."""
    return nx.read_gml(tree_path)


@cache
def _reference_memberships(tree_path: str) -> dict[int, set[int]]:
    """Build and cache ground-truth cluster memberships from a GML transmission tree."""
    tree = _load_tree_template(tree_path)
    memberships: dict[int, set[int]] = defaultdict(set)

    for cluster_id, node_label in enumerate(tree.nodes()):
        cluster_members = set(node_label).union(tree.successors(node_label))
        for member in cluster_members:
            memberships[int(member)].add(cluster_id)

    return dict(memberships)


# ---------------------------------------------------------------------------
# Private helpers — BCubed F1 via Leiden
# ---------------------------------------------------------------------------


def _partition_to_memberships(
    graph: ig.Graph, partition: ig.VertexClustering
) -> dict[int, set[int]]:
    """Map case IDs to their singleton cluster-ID sets from an igraph partition."""
    return {
        int(case_id): {partition.membership[index]}
        for index, case_id in enumerate(graph.vs["case_id"])
    }


def _clustering(
    case_a: np.ndarray,
    case_b: np.ndarray,
    vertex_ids: list[Any],
    scores: np.ndarray,
    sparsification_threshold: float,
    resolution_grid: np.ndarray,
    reference: dict[Any, set[int]],
    n_restarts: int,
    rng_seed: int,
) -> tuple[float, float, float]:
    graph = build_weighted_graph(
        source_ids=case_a,
        target_ids=case_b,
        weights=scores,
        minimum_weight=sparsification_threshold,
        vertex_ids=vertex_ids,
    )

    if graph.vcount() == 0:
        return float("nan"), float("nan"), float("nan")

    best_f1 = 0.0
    memberships = {}
    for resolution in resolution_grid:
        try:
            partition, _ = run_leiden_partition(
                graph,
                weight_column="weight",
                resolution=float(resolution),
                num_restarts=n_restarts,
                rng_seed=rng_seed,
            )
            predicted = _partition_to_memberships(graph, partition)
            _, _, f1 = bcubed_scores(predicted, reference)
            memberships[resolution] = predicted
            if f1 > best_f1:
                best_f1 = f1
        except (ValueError, RuntimeError):
            continue

    ordered_resolutions = [
        resolution for resolution in resolution_grid if resolution in memberships
    ]
    if len(ordered_resolutions) >= 2:
        stability = []
        for res1, res2 in zip(ordered_resolutions[:-1], ordered_resolutions[1:]):
            p1_mem = memberships[res1]
            p2_mem = memberships[res2]
            _, _, f1 = bcubed_scores(predicted_memberships=p1_mem, reference_memberships=p2_mem)
            stability.append(f1)
        mean_stability = np.mean(stability)
        std_stability = np.std(stability)
    else:
        mean_stability = float("nan")
        std_stability = float("nan")

    return float(best_f1), float(mean_stability), float(std_stability)


def _compute_loss(current, baseline, key):
    """Return relative loss ``(current - baseline) / baseline``, or ``None`` when unavailable."""
    if baseline is None or key not in baseline:
        return None
    return (current - float(baseline[key])) / float(baseline[key])


def _make_model_result(
    ap: float,
    best_f1: float,
    mean_stability: float,
    std_stability: float,
    baseline: dict[str, float] | None,
) -> ModelResult:
    return ModelResult(
        ap=ap,
        ap_loss=_compute_loss(ap, baseline, "ap"),
        best_f1=best_f1,
        f1_loss=_compute_loss(best_f1, baseline, "best_f1"),
        mean_stability=mean_stability,
        std_stability=std_stability,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_scenario(
    tree_path: str,
    scenario_name: str,
    generation_parameters: dict[str, Any],
    inference_parameters: dict[str, Any],
    logistic_classifier: dict[str, LogisticRegression] = None,
    baseline_performance: dict[str, dict[str, float]] = None,
    sparsification: dict[str, float] = None,
    n_restarts: int = 10,
    rng_seed: int = 12345,
    training_fraction: float = 0.1,
    return_classifier: bool = False,
    return_scores: bool = False,
) -> tuple[ScenarioResult, Any, Any]:
    """Simulate an epidemic, score pairs with all models, and return evaluation metrics.

    Parameters
    ----------
    tree_path : str
        Path to the pre-built GML transmission tree.
    scenario_name : str
        Label stored in the returned :class:`ScenarioResult`.
    generation_parameters : dict
        Parameters used to simulate epidemic dates and genomic sequences.
    inference_parameters : dict
        Parameters used to build EpiLink scorers.
    logistic_classifier : dict or None
        Pre-fitted logistic classifiers (keyed by model key).  When ``None``,
        classifiers are trained on *training_fraction* of the simulated pairs.
    baseline_performance : dict or None
        Per-model ``{"ap": ..., "best_f1": ...}`` from the loss-reference run,
        used to compute relative loss values.
    sparsification : dict or None
        Per-model minimum edge-weight thresholds for graph construction.
    n_restarts : int
        Number of Leiden restarts per resolution.
    rng_seed : int
        Global RNG seed.
    training_fraction : float
        Fraction of pairs used to train logistic classifiers.
    return_classifier : bool
        If ``True``, include fitted classifiers in the second return value.
    return_scores : bool
        If ``True``, include a raw scores DataFrame in the third return value.

    Returns
    -------
    tuple[ScenarioResult, classifiers | None, scores_df | None]
    """
    tree_path = str(Path(tree_path).expanduser().resolve())
    sparsification = sparsification or {}
    resolution_grid = np.arange(0.1, 1.1, 0.1)

    # ------------------------------------------------------------------
    # 1. Simulate epidemic and build pairwise table
    # ------------------------------------------------------------------
    tree = _load_tree_template(tree_path).copy()
    reference = _reference_memberships(tree_path)
    data_nhp = build_natural_history_parameters(generation_parameters)
    synthetic_genome_length = int(generation_parameters.get("synthetic_genome_length", 5_000))
    fraction_sampled = float(generation_parameters.get("fraction_sampled", 1.0))

    data_profile = InfectiousnessToTransmission(parameters=data_nhp, rng_seed=rng_seed)
    populated_tree = simulate_epidemic_dates(
        transmission_profile=data_profile,
        tree=tree,
        fraction_sampled=fraction_sampled,
    )
    genomic_outputs = simulate_genomic_sequences(
        transmission_profile=data_profile,
        tree=populated_tree,
        genome_length=synthetic_genome_length,
    )
    pairwise = build_pairwise_case_table(genomic_outputs["packed"], populated_tree)

    pairs = pairwise.loc[pairwise[PAIRWISE_BOTH_SAMPLED_COLUMN]].copy()
    is_related = pairs[PAIRWISE_RELATED_COLUMN].astype(int).values
    n_pairs = len(pairs)
    prevalence = float(is_related.mean()) if n_pairs > 0 else float("nan")
    sampling_date_differences = pairs[PAIRWISE_TEMPORAL_DISTANCE_COLUMN].to_numpy(copy=False)
    deterministic_distances = pairs[PAIRWISE_DETERMINISTIC_DISTANCE_COLUMN].to_numpy(copy=False)
    stochastic_distances = pairs[PAIRWISE_STOCHASTIC_DISTANCE_COLUMN].to_numpy(copy=False)
    case_a = pairs[PAIRWISE_CASE_A_COLUMN].to_numpy(copy=False)
    case_b = pairs[PAIRWISE_CASE_B_COLUMN].to_numpy(copy=False)
    vertex_ids = np.unique(np.concatenate((case_a, case_b))).tolist()
    distance_columns = {
        PAIRWISE_DETERMINISTIC_DISTANCE_COLUMN: deterministic_distances,
        PAIRWISE_STOCHASTIC_DISTANCE_COLUMN: stochastic_distances,
    }
    feature_matrices = {
        PAIRWISE_DETERMINISTIC_DISTANCE_COLUMN: np.column_stack(
            (sampling_date_differences, deterministic_distances)
        ),
        PAIRWISE_STOCHASTIC_DISTANCE_COLUMN: np.column_stack(
            (sampling_date_differences, stochastic_distances)
        ),
    }

    # ------------------------------------------------------------------
    # 2. Build EpiLink scorers
    # ------------------------------------------------------------------
    linkage_models = build_linkage_models(inference_parameters, rng_seed=rng_seed)

    # ------------------------------------------------------------------
    # 3. Score pairs
    # ------------------------------------------------------------------
    scores: dict[str, np.ndarray] = {}

    for spec in _EPILINK_SPECS:
        scores[spec["key"]] = np.asarray(
            linkage_models[spec["mutation_process"]].score_target(
                sample_time_difference=sampling_date_differences,
                genetic_distance=distance_columns[spec["distance_col"]],
            ),
            dtype=float,
        )

    classifiers = {}
    for spec in _LOGIT_SPECS:
        feature_matrix = feature_matrices[spec["distance_col"]]
        if logistic_classifier is not None:
            scores[spec["key"]] = logistic_classifier[spec["key"]].predict_proba(feature_matrix)[
                :, 1
            ]
        else:
            scores[spec["key"]], classifier = predict_logistic_scores(
                feature_matrix,
                is_related,
                training_fraction=training_fraction,
                rng_seed=rng_seed,
                return_classifier=return_classifier,
            )
            classifiers[spec["key"]] = classifier

    # ------------------------------------------------------------------
    # 4. Compute metrics and assemble result
    # ------------------------------------------------------------------
    result = ScenarioResult(scenario_name=scenario_name, n_pairs=n_pairs, prevalence=prevalence)
    has_both_classes = len(np.unique(is_related)) == 2
    for key in MODEL_KEYS:
        threshold = float(sparsification.get(key, 0.0001))
        ap = (
            float(average_precision_score(is_related, scores[key]))
            if has_both_classes
            else float("nan")
        )
        best_f1, stability_mean, stability_std = _clustering(
            case_a,
            case_b,
            vertex_ids,
            scores[key],
            sparsification_threshold=threshold,
            resolution_grid=resolution_grid,
            reference=reference,
            n_restarts=n_restarts,
            rng_seed=rng_seed,
        )
        baseline = (baseline_performance or {}).get(key)
        result.models[key] = _make_model_result(
            ap, best_f1, stability_mean, stability_std, baseline
        )

    if return_scores:
        scores_df = pd.DataFrame(scores)
        scores_df.insert(0, PAIRWISE_RELATED_COLUMN, is_related)
    else:
        scores_df = None
    return result, classifiers if return_classifier else None, scores_df
