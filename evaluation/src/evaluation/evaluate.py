from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import igraph as ig
import networkx as nx
import numpy as np
from epilink import (
    EpiLink,
    InfectiousnessToTransmission,
    NaturalHistoryParameters,
    build_pairwise_case_table,
    simulate_epidemic_dates,
    simulate_genomic_sequences,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

try:
    from .leiden import build_weighted_graph, run_leiden_partition
    from .metrics import bcubed_scores, predict_logistic_scores
except ImportError:
    from leiden import build_weighted_graph, run_leiden_partition
    from metrics import bcubed_scores, predict_logistic_scores



# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

MODEL_KEYS = ("EDD", "EDS", "ESD", "ESS", "LD", "LS")
"""Canonical model abbreviations.

EDD — EpiLink, deterministic inference, deterministic data
EDS — EpiLink, deterministic inference, stochastic data
ESD — EpiLink, stochastic inference,   deterministic data
ESS — EpiLink, stochastic inference,   stochastic data
LD  — Logistic regression on deterministic distances
LS  — Logistic regression on stochastic distances
"""

_DEFAULT_SPARSIFICATION = 0.0001


@dataclass(frozen=True)
class ModelResult:
    ap: float
    ap_loss: float | None
    best_f1: float
    f1_loss: float | None
    mean_stability: float
    std_stability: float


@dataclass
class ScenarioResult:
    scenario_name: str
    n_pairs: int
    prevalence: float
    models: dict[str, ModelResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal scoring specs
# ---------------------------------------------------------------------------

_EPILINK_SPECS: list[dict[str, str]] = [
    {"key": "EDD", "mutation_process": "deterministic", "distance_col": "DeterministicDistance"},
    {"key": "EDS", "mutation_process": "deterministic", "distance_col": "StochasticDistance"},
    {"key": "ESD", "mutation_process": "stochastic",    "distance_col": "DeterministicDistance"},
    {"key": "ESS", "mutation_process": "stochastic",    "distance_col": "StochasticDistance"},
]

_LOGIT_SPECS: list[dict[str, str]] = [
    {"key": "LD", "distance_col": "DeterministicDistance"},
    {"key": "LS", "distance_col": "StochasticDistance"},
]


# ---------------------------------------------------------------------------
# Private helpers — simulation
# ---------------------------------------------------------------------------

def _build_nhp(params: dict[str, Any]) -> NaturalHistoryParameters:
    return NaturalHistoryParameters(
        incubation_shape=float(params.get("incubation_shape", 5.807)),
        incubation_scale=float(params.get("incubation_scale", 0.948)),
        latent_shape=float(params.get("latent_shape", 3.38)),
        symptomatic_rate=float(params.get("symptomatic_rate", 0.37)),
        symptomatic_shape=float(params.get("symptomatic_shape", 1.0)),
        transmission_rate_ratio=float(params.get("transmission_rate_ratio", 2.29)),
        testing_delay_shape=float(params.get("testing_delay_shape", 1.0)),
        testing_delay_scale=float(params.get("testing_delay_scale", 1.0)),
        substitution_rate=float(params.get("substitution_rate", 1e-3)),
        relaxation=float(params.get("relaxation", 0.33)),
        genome_length=int(params.get("genome_length", 29903)),
    )


def _resolve_target(raw: Any) -> tuple[str, ...]:
    labels = (raw,) if isinstance(raw, str) else tuple(str(v) for v in raw)
    if not labels or any(not label.strip() for label in labels):
        raise ValueError("target must contain at least one non-empty label.")
    return labels


@lru_cache(maxsize=None)
def _load_tree_template(tree_path: str) -> nx.DiGraph:
    return nx.read_gml(tree_path)


@lru_cache(maxsize=None)
def _reference_memberships_for_tree(tree_path: str) -> dict[int, set[int]]:
    tree = _load_tree_template(tree_path)
    memberships: dict[int, set[int]] = defaultdict(set)

    for cluster_id, node_label in enumerate(tree.nodes()):
        cluster_members = {node_label} | set(tree.successors(node_label))
        for member in cluster_members:
            memberships[int(member)].add(cluster_id)

    return dict(memberships)


# ---------------------------------------------------------------------------
# Private helpers — BCubed F1 via Leiden
# ---------------------------------------------------------------------------


def _partition_to_memberships(graph: ig.Graph, partition: ig.VertexClustering) -> dict[int, set[int]]:
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

    ordered_resolutions = [resolution for resolution in resolution_grid if resolution in memberships]
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


def _make_model_result(
        ap: float,
        best_f1: float,
        mean_stability: float,
        std_stability: float,
        baseline: dict[str, float] | None,
) -> ModelResult:
    ap_loss = (ap - float(baseline["ap"])) if baseline is not None and "ap" in baseline else None
    f1_loss = (best_f1 - float(baseline["best_f1"])) if baseline is not None and "best_f1" in baseline else None
    return ModelResult(
        ap=ap,
        ap_loss=ap_loss,
        best_f1=best_f1,
        f1_loss=f1_loss,
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
) -> tuple[ScenarioResult, Any]:
    tree_path = str(Path(tree_path).expanduser().resolve())
    sparsification = sparsification or {}
    resolution_grid = np.arange(0.1, 1.1, 0.1)

    # ------------------------------------------------------------------
    # 1. Simulate epidemic and build pairwise table
    # ------------------------------------------------------------------
    tree = _load_tree_template(tree_path).copy()
    reference = _reference_memberships_for_tree(tree_path)
    data_nhp = _build_nhp(generation_parameters)
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

    pairs = pairwise.loc[pairwise["BothSampled"]].copy()
    is_related = pairs["IsRelated"].astype(int).values
    n_pairs = len(pairs)
    prevalence = float(is_related.mean()) if n_pairs > 0 else float("nan")
    sampling_date_differences = pairs["SamplingDateDistanceDays"].to_numpy(copy=False)
    deterministic_distances = pairs["DeterministicDistance"].to_numpy(copy=False)
    stochastic_distances = pairs["StochasticDistance"].to_numpy(copy=False)
    case_a = pairs["CaseA"].to_numpy(copy=False)
    case_b = pairs["CaseB"].to_numpy(copy=False)
    vertex_ids = np.unique(np.concatenate((case_a, case_b))).tolist()
    distance_columns = {
        "DeterministicDistance": deterministic_distances,
        "StochasticDistance": stochastic_distances,
    }
    feature_matrices = {
        "DeterministicDistance": np.column_stack((sampling_date_differences, deterministic_distances)),
        "StochasticDistance": np.column_stack((sampling_date_differences, stochastic_distances)),
    }

    # ------------------------------------------------------------------
    # 2. Build EpiLink scorers
    # ------------------------------------------------------------------
    raw_target = inference_parameters.get("target", ("ad(0)", "ca(0,0)"))
    maximum_depth = int(inference_parameters.get("maximum_depth", 0))
    target_labels = _resolve_target(raw_target)
    num_simulations = int(inference_parameters.get("num_simulations", 10_000))

    infer_nhp = _build_nhp(inference_parameters)
    infer_profile = InfectiousnessToTransmission(parameters=infer_nhp, rng_seed=rng_seed)
    linkage_models = {
        mp: EpiLink(
            mutation_process=mp,
            transmission_profile=infer_profile,
            maximum_depth=maximum_depth,
            mc_samples=num_simulations,
            target=target_labels,
        )
        for mp in ("deterministic", "stochastic")
    }

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
            scores[spec["key"]] = logistic_classifier[spec["key"]].predict_proba(feature_matrix)[:, 1]
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
        threshold = float(sparsification.get(key, _DEFAULT_SPARSIFICATION))
        ap = float(average_precision_score(is_related, scores[key])) if has_both_classes else float("nan")
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
        result.models[key] = _make_model_result(ap, best_f1, stability_mean, stability_std, baseline)

    return result, classifiers if return_classifier else None
