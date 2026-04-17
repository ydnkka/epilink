"""Quantify retention and runtime effects of edge sparsification, then determine optimal thresholds."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from config import (
    configure_logging,
    get_config_value,
    get_pipeline_log_path,
    load_config,
    resolve_configured_output_path,
    resolve_generation_baseline_parameters,
    resolve_inference_baseline_parameters,
)
from leiden import build_weighted_graph, run_leiden_partition, total_edge_weight
from models import build_linkage_models, build_natural_history_parameters, predict_logistic_scores
from specs import (
    DEFAULT_SEED,
    EPILINK_SPECS,
    LOGIT_SPECS,
    MODEL_KEYS,
    PAIRWISE_BOTH_SAMPLED_COLUMN,
    PAIRWISE_CASE_A_COLUMN,
    PAIRWISE_CASE_B_COLUMN,
    PAIRWISE_RELATED_COLUMN,
    PAIRWISE_TEMPORAL_DISTANCE_COLUMN,
    score_metadata,
)

from epilink import (
    InfectiousnessToTransmission,
    build_pairwise_case_table,
    simulate_epidemic_dates,
    simulate_genomic_sequences,
)

LOGGER = logging.getLogger(__name__)


def timed(function, *args, **kwargs):
    """Call *function* with the given arguments and return ``(result, elapsed_seconds)``."""
    start = time.perf_counter()
    output = function(*args, **kwargs)
    return output, time.perf_counter() - start


def sparsify_edges(
    pairwise_frame: pd.DataFrame, min_edge_weight: float, weight_column: str
) -> pd.DataFrame:
    """Return *pairwise_frame* filtered to rows whose *weight_column* ≥ *min_edge_weight*.

    A threshold of zero or below returns the full frame unchanged.
    """
    if float(min_edge_weight) <= 0:
        return pairwise_frame
    return pairwise_frame.loc[pairwise_frame[weight_column] >= min_edge_weight]


def timed_igraph_and_leiden(
    pairwise_frame: pd.DataFrame,
    weight_column: str,
    vertex_ids: pd.Index,
    resolution: float,
    rng_seed: int = 12345,
) -> tuple[float, float]:
    """Return (graph-build seconds, Leiden seconds) for a sparsified edge set.

    Parameters
    ----------
    pairwise_frame : pandas.DataFrame
        Pre-filtered pairwise table (already sparsified).
    weight_column : str
        Edge weight column name.
    vertex_ids : pandas.Index
        Complete set of vertex identifiers (ensures isolated nodes are included).
    resolution : float
        Leiden resolution parameter.
    rng_seed : int, optional
        RNG seed for Leiden.

    Returns
    -------
    tuple[float, float]
        ``(build_seconds, leiden_seconds)`` wall-clock durations.
    """
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
    retention_frame: pd.DataFrame, min_weight_retention: float
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


def build_surface_axis(max_value: float, step: float) -> np.ndarray:
    """Build an inclusive non-negative axis for score-surface evaluation."""
    step = float(step)
    if step <= 0:
        raise ValueError(f"surface step must be > 0, got {step}")

    upper = step * np.ceil(max(0.0, float(max_value)) / step)
    return np.arange(0.0, upper + step, step, dtype=float)


def build_compatibility_surface(
    linkage_models: dict[str, Any],
    snps: np.ndarray,
    days: np.ndarray,
    mutation_processes: tuple[str, ...],
) -> pd.DataFrame:
    """Evaluate target-compatibility surfaces over a SNP-by-time grid."""
    snp_grid, day_grid = np.meshgrid(np.asarray(snps, dtype=float), np.asarray(days, dtype=float))
    surfaces: list[pd.DataFrame] = []
    for mutation_process in mutation_processes:
        compatibilities = np.asarray(
            linkage_models[mutation_process].score_target(
                sample_time_difference=day_grid.ravel(),
                genetic_distance=snp_grid.ravel(),
            ),
            dtype=float,
        ).reshape(snp_grid.shape)
        surfaces.append(
            pd.DataFrame(
                {
                    "snp": snp_grid.ravel().astype(float),
                    "days": day_grid.ravel().astype(float),
                    "mutation_process": mutation_process,
                    "compatibility": compatibilities.ravel().astype(float),
                }
            )
        )
    return pd.concat(surfaces, ignore_index=True)


def build_logit_surface(
    pairwise_frame: pd.DataFrame,
    is_related: np.ndarray,
    snps: np.ndarray,
    days: np.ndarray,
    training_fraction: float,
    rng_seed: int,
) -> pd.DataFrame:
    """Train logistic surfaces on deterministic and stochastic distance features."""
    snp_grid, day_grid = np.meshgrid(np.asarray(snps, dtype=float), np.asarray(days, dtype=float))
    predict_features = np.column_stack((day_grid.ravel(), snp_grid.ravel()))
    surfaces: list[pd.DataFrame] = []
    for spec in LOGIT_SPECS:
        feature_matrix = pairwise_frame[
            [PAIRWISE_TEMPORAL_DISTANCE_COLUMN, spec["distance_col"]]
        ].to_numpy(copy=False)
        probabilities, _ = predict_logistic_scores(
            feature_matrix,
            is_related,
            training_fraction=training_fraction,
            rng_seed=rng_seed,
            predict_feature_matrix=predict_features,
            return_classifier=False,
        )
        surfaces.append(
            pd.DataFrame(
                {
                    "snp": snp_grid.ravel().astype(float),
                    "days": day_grid.ravel().astype(float),
                    "weight_column": spec["key"],
                    "distance_col": spec["distance_col"],
                    **score_metadata(
                        spec["key"],
                        logistic_training_fraction=training_fraction,
                    ),
                    "logit_probability": np.asarray(probabilities, dtype=float),
                }
            )
        )
    return pd.concat(surfaces, ignore_index=True)


def merge_score_surfaces(
    compatibility_surface: pd.DataFrame,
    logit_surface: pd.DataFrame,
) -> pd.DataFrame:
    """Combine compatibility and logit surfaces into one wide table."""
    compatibility_frame = (
        compatibility_surface.pivot(
            index=["snp", "days"], columns="mutation_process", values="compatibility"
        )
        .rename(
            columns={
                "deterministic": "compatibility_deterministic",
                "stochastic": "compatibility_stochastic",
            }
        )
        .reset_index()
    )
    logit_frame = (
        logit_surface.pivot(
            index=["snp", "days"], columns="weight_column", values="logit_probability"
        )
        .rename(
            columns={
                "LD": "logit_deterministic",
                "LS": "logit_stochastic",
            }
        )
        .reset_index()
    )
    training_fraction = pd.Series(logit_surface["training_fraction"]).dropna().unique()
    logit_training_fraction = (
        float(training_fraction[0]) if len(training_fraction) > 0 else float("nan")
    )

    merged = compatibility_frame.merge(logit_frame, on=["snp", "days"], how="outer")
    merged["logit_training_fraction"] = logit_training_fraction

    for column in (
        "compatibility_deterministic",
        "compatibility_stochastic",
        "logit_deterministic",
        "logit_stochastic",
    ):
        if column not in merged.columns:
            merged[column] = np.nan

    column_order = [
        "snp",
        "days",
        "compatibility_deterministic",
        "compatibility_stochastic",
        "logit_deterministic",
        "logit_stochastic",
    ]
    return merged.loc[:, column_order].sort_values(["days", "snp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(config_path: str | Path = "config.yaml") -> None:
    """Run the sparsification analysis and write retention/threshold parquet outputs."""
    config = load_config(config_path)
    configure_logging(log_file=get_pipeline_log_path(config))
    LOGGER.info("sparsification: starting")
    workflow = get_config_value(config, "workflows.sparsification", default={})
    generation_parameters = resolve_generation_baseline_parameters(config)
    inference_parameters = resolve_inference_baseline_parameters(config)
    training_fraction = float(workflow.get("training_fraction"))
    rng_seed = int(get_config_value(config, "rng_seed", default=DEFAULT_SEED))
    resolution = float(workflow.get("resolution"))
    sparsification_thresholds = [float(value) for value in workflow.get("thresholds")]
    min_weight_retention = float(workflow.get("min_weight_retention"))

    # ------------------------------------------------------------------
    # 1. Simulate epidemic and build scored pairwise table
    # ------------------------------------------------------------------
    LOGGER.info("sparsification: simulating and scoring pairs")
    tree_path = resolve_configured_output_path(config, "outputs.scovmod.tree_path")
    tree = nx.read_gml(tree_path)

    data_profile = InfectiousnessToTransmission(
        parameters=build_natural_history_parameters(generation_parameters),
        rng_seed=rng_seed,
    )
    populated_tree = simulate_epidemic_dates(
        transmission_profile=data_profile,
        tree=tree,
        fraction_sampled=float(generation_parameters.get("fraction_sampled", 1.0)),
    )
    genomic_outputs = simulate_genomic_sequences(
        transmission_profile=data_profile,
        tree=populated_tree,
        genome_length=int(generation_parameters.get("synthetic_genome_length", 5_000)),
    )
    pairwise = build_pairwise_case_table(genomic_outputs["packed"], populated_tree)
    pairs = pairwise.loc[pairwise[PAIRWISE_BOTH_SAMPLED_COLUMN]].copy()
    is_related = pairs[PAIRWISE_RELATED_COLUMN].astype(int).values
    sampling_dates = pairs[PAIRWISE_TEMPORAL_DISTANCE_COLUMN].to_numpy(copy=False)

    # ------------------------------------------------------------------
    # 2. Build EpiLink scorers and score pairs
    # ------------------------------------------------------------------
    linkage_models = build_linkage_models(inference_parameters, rng_seed=rng_seed)

    for spec in EPILINK_SPECS:
        pairs[spec["key"]] = np.asarray(
            linkage_models[spec["mutation_process"]].score_target(
                sample_time_difference=sampling_dates,
                genetic_distance=pairs[spec["distance_col"]].to_numpy(copy=False),
            ),
            dtype=float,
        )

    for spec in LOGIT_SPECS:
        feature_matrix = pairs[[PAIRWISE_TEMPORAL_DISTANCE_COLUMN, spec["distance_col"]]].to_numpy(
            copy=False
        )
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
    LOGGER.info("sparsification: evaluating thresholds")
    results_dir = resolve_configured_output_path(config, "outputs.sparsification.directory")
    results_dir.mkdir(parents=True, exist_ok=True)

    snps = build_surface_axis(15, 0.1)
    days = build_surface_axis(20, 0.1)

    compatibility_surface = build_compatibility_surface(
        linkage_models,
        snps=snps,
        days=days,
        mutation_processes=tuple(linkage_models),
    )

    logit_surface = build_logit_surface(
        pairs,
        is_related=is_related,
        snps=snps,
        days=days,
        training_fraction=training_fraction,
        rng_seed=rng_seed,
    )
    merged_surface = merge_score_surfaces(compatibility_surface, logit_surface)
    merged_surface.to_parquet(results_dir / "score_surfaces.parquet", index=False)

    reference_nodes = pd.Index(
        pd.unique(pairs[[PAIRWISE_CASE_A_COLUMN, PAIRWISE_CASE_B_COLUMN]].values.ravel())
    ).astype(str)

    retention_rows: list[dict[str, object]] = []
    for weight_column in MODEL_KEYS:
        if weight_column not in pairs.columns:
            continue

        reference_weight = total_edge_weight(pairs, weight_column=weight_column)
        reference_edge_count = len(pairs)
        metadata = score_metadata(weight_column, logistic_training_fraction=training_fraction)

        for threshold in sparsification_thresholds:
            filtered, sparsify_seconds = timed(sparsify_edges, pairs, threshold, weight_column)
            retained_weight = total_edge_weight(filtered, weight_column=weight_column)
            retained_edges = len(filtered)

            build_seconds, leiden_seconds = timed_igraph_and_leiden(
                filtered,
                weight_column=weight_column,
                vertex_ids=reference_nodes,
                resolution=resolution,
                rng_seed=rng_seed,
            )
            retention_rows.append(
                {
                    "weight_column": weight_column,
                    **metadata,
                    "min_edge_weight": float(threshold),
                    "edge_retention_frac": (
                        float(retained_edges / reference_edge_count)
                        if reference_edge_count > 0
                        else float("nan")
                    ),
                    "weight_retention_frac": (
                        float(retained_weight / reference_weight)
                        if reference_weight > 0
                        else float("nan")
                    ),
                    "t_pipeline_s": float(sparsify_seconds + build_seconds + leiden_seconds),
                }
            )

    retention_frame = (
        pd.DataFrame(retention_rows)
        .sort_values(["weight_column", "min_edge_weight"])
        .reset_index(drop=True)
    )
    retention_frame.to_parquet(results_dir / "sparsify_edge_retention.parquet", index=False)

    optimal_thresholds = determine_optimal_thresholds(retention_frame, min_weight_retention)
    (results_dir / "optimal_thresholds.json").write_text(json.dumps(optimal_thresholds, indent=2))
    LOGGER.info(
        "sparsification: done (optimal thresholds: %s)",
        {k: f"{v:.4f}" for k, v in optimal_thresholds.items()},
    )


if __name__ == "__main__":
    main()
