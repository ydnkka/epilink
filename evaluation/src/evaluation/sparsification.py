"""Quantify retention and runtime effects of edge sparsification, then determine optimal thresholds."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from epilink import (
    EpiLink,
    InfectiousnessToTransmission,
    NaturalHistoryParameters,
    build_pairwise_case_table,
    simulate_epidemic_dates,
    simulate_genomic_sequences,
)

try:
    from .config import gamma_mean_cv_to_shape_scale, load_config, resolve_path
    from .evaluate import MODEL_KEYS
    from .leiden import build_weighted_graph, run_leiden_partition, total_edge_weight
    from .metrics import predict_logistic_scores
except ImportError:
    from config import gamma_mean_cv_to_shape_scale, load_config, resolve_path
    from evaluate import MODEL_KEYS
    from leiden import build_weighted_graph, run_leiden_partition, total_edge_weight
    from metrics import predict_logistic_scores


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

# Maps MODEL_KEYS abbreviations to descriptive column metadata.
_SCORE_METADATA: dict[str, dict[str, str]] = {
    "EDD": {"score_family": "epilink_score",     "inference_process": "deterministic", "data_process": "deterministic"},
    "EDS": {"score_family": "epilink_score",     "inference_process": "deterministic", "data_process": "stochastic"},
    "ESD": {"score_family": "epilink_score",     "inference_process": "stochastic",    "data_process": "deterministic"},
    "ESS": {"score_family": "epilink_score",     "inference_process": "stochastic",    "data_process": "stochastic"},
    "LD":  {"score_family": "logit_probability", "inference_process": "not_applicable", "data_process": "deterministic"},
    "LS":  {"score_family": "logit_probability", "inference_process": "not_applicable", "data_process": "stochastic"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nhp_from_baseline(baseline: dict[str, Any], fixed: dict[str, Any]) -> NaturalHistoryParameters:
    """Build NaturalHistoryParameters from a nested config baseline section."""
    inc = gamma_mean_cv_to_shape_scale(float(baseline["incubation"]["mean"]), float(baseline["incubation"]["cv"]))
    td = gamma_mean_cv_to_shape_scale(float(baseline["testing_delay"]["mean"]), float(baseline["testing_delay"]["cv"]))
    return NaturalHistoryParameters(
        incubation_shape=float(inc["shape"]),
        incubation_scale=float(inc["scale"]),
        latent_shape=float(fixed.get("latent_shape", 3.38)),
        symptomatic_rate=float(fixed.get("symptomatic_rate", 0.37)),
        symptomatic_shape=float(fixed.get("symptomatic_shape", 1.0)),
        transmission_rate_ratio=float(fixed.get("transmission_rate_ratio", 2.29)),
        testing_delay_shape=float(td["shape"]),
        testing_delay_scale=float(td["scale"]),
        substitution_rate=float(baseline.get("substitution_rate", 1e-3)),
        relaxation=float(baseline.get("relaxation", 0.33)),
        genome_length=int(fixed.get("genome_length", 29903)),
    )


def timed(function, *args, **kwargs):
    """Measure wall-clock time for a function call."""
    start = time.perf_counter()
    output = function(*args, **kwargs)
    return output, time.perf_counter() - start


def sparsify_edges(pairwise_frame: pd.DataFrame, min_edge_weight: float, weight_column: str) -> pd.DataFrame:
    """Filter a pairwise table to the retained edges for a threshold."""
    if float(min_edge_weight) <= 0:
        return pairwise_frame
    return pairwise_frame.loc[pairwise_frame[weight_column] >= min_edge_weight]


def score_metadata(weight_column: str) -> dict[str, str | float]:
    """Return metadata for a score column — family, inference process, data process."""
    meta = _SCORE_METADATA.get(weight_column, {
        "score_family": "other",
        "inference_process": "unknown",
        "data_process": "unknown",
    })
    training_fraction = 0.1 if meta["score_family"] == "logit_probability" else float("nan")
    return {**meta, "training_fraction": training_fraction}


def timed_igraph_and_leiden(
    pairwise_frame: pd.DataFrame,
    weight_column: str,
    vertex_ids: pd.Index,
    resolution: float,
    rng_seed: int = 12345,
) -> tuple[float, float]:
    """Measure graph construction and Leiden runtime for a sparsified edge set."""
    graph, build_seconds = timed(
        build_weighted_graph,
        pairwise_frame=pairwise_frame,
        weight_column=weight_column,
        minimum_weight=0.0,
        vertex_ids=vertex_ids,
    )
    _, leiden_seconds = timed(
        run_leiden_partition,
        graph,
        weight_column=weight_column,
        resolution=float(resolution),
        num_restarts=1,
        rng_seed=rng_seed,
    )
    return float(build_seconds), float(leiden_seconds)


def determine_optimal_thresholds(
    retention_frame: pd.DataFrame,
    min_weight_retention: float,
) -> dict[str, float]:
    """Select the lowest positive threshold that preserves at least min_weight_retention fraction."""

    def _rank_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
        return candidates.sort_values(
            ["weight_retention_frac", "edge_retention_frac", "min_edge_weight"],
            ascending=[False, True, True],
            kind="mergesort",
        )

    optimal_thresholds: dict[str, float] = {}
    for weight_column in retention_frame["weight_column"].unique():
        sub = retention_frame[retention_frame["weight_column"] == weight_column].copy()
        positive = sub[sub["min_edge_weight"] > 0].copy()
        valid = positive[positive["weight_retention_frac"] >= min_weight_retention]

        if not valid.empty:
            best_row = _rank_candidates(valid).iloc[0]
        elif not positive.empty:
            best_row = _rank_candidates(positive).iloc[0]
        else:
            best_row = _rank_candidates(sub).iloc[0]

        optimal_thresholds[weight_column] = float(best_row["min_edge_weight"])

    return optimal_thresholds


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_path: str | Path = "config.yaml") -> None:
    parser = argparse.ArgumentParser(description="Edge sparsification analysis.")
    parser.add_argument("--config", default=config_path, help="Path to YAML configuration.")
    parser.add_argument(
        "--weight-columns", nargs="+", default=list(MODEL_KEYS),
        help="Score columns to analyse (default: all MODEL_KEYS).",
    )
    parser.add_argument("--gamma", type=float, default=0.5, help="Leiden resolution for timing diagnostics.")
    parser.add_argument("--min-weight-retention", type=float, default=0.995)
    args = parser.parse_args()

    config = load_config(args.config)
    fixed = config["fixed_parameters"]
    rng_seed = int(config.get("experiment", {}).get("seed", 12345))
    training_fraction = float(config["execution"]["evaluate_kwargs"].get("training_fraction", 0.1))
    min_edge_weights = [0.0, 0.0001, 0.001, 0.01, 0.1]

    # ------------------------------------------------------------------
    # 1. Simulate epidemic and build scored pairwise table
    # ------------------------------------------------------------------
    tree_path = str(resolve_path(config["paths"]["tree_path"]))
    tree = nx.read_gml(tree_path)

    data_nhp = _nhp_from_baseline(config["generation_baseline"], fixed)
    data_profile = InfectiousnessToTransmission(parameters=data_nhp, rng_seed=rng_seed)
    populated_tree = simulate_epidemic_dates(
        transmission_profile=data_profile,
        tree=tree,
        fraction_sampled=float(fixed.get("fraction_sampled", 1.0)),
    )
    genomic_outputs = simulate_genomic_sequences(
        transmission_profile=data_profile,
        tree=populated_tree,
        genome_length=int(fixed.get("synthetic_genome_length", 5_000)),
    )
    pairwise = build_pairwise_case_table(genomic_outputs["packed"], populated_tree)
    pairs = pairwise.loc[pairwise["BothSampled"]].copy()
    is_related = pairs["IsRelated"].astype(int).values
    sampling_dates = pairs["SamplingDateDistanceDays"].to_numpy(copy=False)

    # ------------------------------------------------------------------
    # 2. Build EpiLink scorers and score pairs
    # ------------------------------------------------------------------
    infer_nhp = _nhp_from_baseline(config["inference_baseline"], fixed)
    infer_profile = InfectiousnessToTransmission(parameters=infer_nhp, rng_seed=rng_seed)
    target_labels = tuple(str(v) for v in fixed.get("target", ("ad(0)", "ca(0,0)")))
    linkage_models = {
        mp: EpiLink(
            mutation_process=mp,
            transmission_profile=infer_profile,
            maximum_depth=int(fixed.get("maximum_depth", 0)),
            mc_samples=int(fixed.get("num_simulations", 10_000)),
            target=target_labels,
        )
        for mp in ("deterministic", "stochastic")
    }

    for spec in _EPILINK_SPECS:
        pairs[spec["key"]] = np.asarray(
            linkage_models[spec["mutation_process"]].score_target(
                sample_time_difference=sampling_dates,
                genetic_distance=pairs[spec["distance_col"]].to_numpy(copy=False),
            ),
            dtype=float,
        )

    for spec in _LOGIT_SPECS:
        feature_matrix = pairs[["SamplingDateDistanceDays", spec["distance_col"]]].to_numpy(copy=False)
        pairs[spec["key"]], _ = predict_logistic_scores(
            feature_matrix,
            is_related,
            training_fraction=training_fraction,
            rng_seed=rng_seed,
            return_classifier=False,
        )

    # ------------------------------------------------------------------
    # 3. Sparsification analysis
    # ------------------------------------------------------------------
    results_dir = resolve_path(config["outputs"]["directory"])
    results_dir.mkdir(parents=True, exist_ok=True)

    reference_nodes = pd.Index(pd.unique(pairs[["CaseA", "CaseB"]].values.ravel())).astype(str)

    retention_rows: list[dict[str, object]] = []
    for weight_column in args.weight_columns:
        if weight_column not in pairs.columns:
            continue

        reference_weight = total_edge_weight(pairs, weight_column=weight_column)
        reference_edge_count = len(pairs)
        metadata = score_metadata(weight_column)

        for threshold in min_edge_weights:
            filtered, sparsify_seconds = timed(sparsify_edges, pairs, threshold, weight_column)
            retained_weight = total_edge_weight(filtered, weight_column=weight_column)
            retained_edges = len(filtered)

            build_seconds, leiden_seconds = timed_igraph_and_leiden(
                filtered,
                weight_column=weight_column,
                vertex_ids=reference_nodes,
                resolution=args.gamma,
                rng_seed=rng_seed,
            )
            retention_rows.append({
                "weight_column": weight_column,
                **metadata,
                "min_edge_weight": float(threshold),
                "edge_retention_frac": (
                    float(retained_edges / reference_edge_count) if reference_edge_count > 0 else float("nan")
                ),
                "weight_retention_frac": (
                    float(retained_weight / reference_weight) if reference_weight > 0 else float("nan")
                ),
                "t_pipeline_s": float(sparsify_seconds + build_seconds + leiden_seconds),
            })

    retention_frame = (
        pd.DataFrame(retention_rows)
        .sort_values(["weight_column", "min_edge_weight"])
        .reset_index(drop=True)
    )
    retention_frame.to_parquet(results_dir / "sparsify_edge_retention.parquet", index=False)

    optimal_thresholds = determine_optimal_thresholds(retention_frame, args.min_weight_retention)
    (results_dir / "optimal_thresholds.json").write_text(json.dumps(optimal_thresholds, indent=2))


if __name__ == "__main__":
    main()
