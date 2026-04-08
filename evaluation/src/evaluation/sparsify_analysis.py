"""Quantify retention and runtime effects of edge sparsification, then determine optimal thresholds."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from epilink_evaluation.execution import StageContext
from epilink_evaluation.graphs import build_weighted_graph, total_edge_weight

logger = logging.getLogger(__name__)


def timed(function, *args, **kwargs):
    """Measure wall-clock time for a function call."""

    start = time.perf_counter()
    output = function(*args, **kwargs)
    duration_seconds = time.perf_counter() - start
    return output, duration_seconds


def sparsify_edges(pairwise_frame: pd.DataFrame, min_edge_weight: float, weight_column: str) -> pd.DataFrame:
    """Filter a pairwise table to the retained edges for a threshold."""

    threshold = float(min_edge_weight)
    if threshold <= 0:
        return pairwise_frame
    return pairwise_frame.loc[pairwise_frame[weight_column] >= threshold]


def score_metadata(weight_column: str) -> dict[str, str | float]:
    """Classify a score column by family, data process, and inference process."""

    normalized = str(weight_column)
    if normalized.startswith("EpiLink"):
        score_family = "epilink_score"
    elif normalized.startswith("Logit"):
        score_family = "logit_probability"
    else:
        score_family = "other"

    if "DeterministicInference" in normalized:
        inference_process = "deterministic"
    elif "StochasticInference" in normalized:
        inference_process = "stochastic"
    else:
        inference_process = "not_applicable"

    if "DeterministicData" in normalized:
        data_process = "deterministic"
    elif "StochasticData" in normalized:
        data_process = "stochastic"
    else:
        data_process = "unknown"

    training_fraction = 0.1 if score_family == "logit_probability" else float("nan")

    return {
        "score_family": score_family,
        "inference_process": inference_process,
        "data_process": data_process,
        "training_fraction": training_fraction,
    }


def timed_igraph_and_leiden(
    pairwise_frame: pd.DataFrame,
    *,
    weight_column: str,
    vertex_ids: pd.Index,
    resolution: float,
) -> tuple[float, float]:
    """Measure graph construction and Leiden runtime for a sparsified edge set."""

    graph, build_seconds = timed(
        build_weighted_graph,
        pairwise_frame,
        weight_column=weight_column,
        minimum_weight=0.0,
        vertex_ids=vertex_ids,
    )

    def _run_leiden():
        return graph.community_leiden(
            weights=weight_column,
            resolution=float(resolution),
            n_iterations=-1,
        )

    _, leiden_seconds = timed(_run_leiden)
    return float(build_seconds), float(leiden_seconds)


def determine_optimal_thresholds(
    retention_frame: pd.DataFrame,
    min_weight_retention: float,
) -> dict[str, float]:
    """Select a positive threshold that prioritizes retention before edge reduction."""

    def _rank_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
        """Rank thresholds by retained weight first, then by retained edges, then by threshold."""

        return candidates.sort_values(
            ["weight_retention_frac", "edge_retention_frac", "min_edge_weight"],
            ascending=[False, True, True],
            kind="mergesort",
        )

    optimal_thresholds: dict[str, float] = {}
    for weight_column in retention_frame["weight_column"].unique():
        sub_df = retention_frame[retention_frame["weight_column"] == weight_column].copy()
        positive_df = sub_df[sub_df["min_edge_weight"] > 0].copy()
        valid_positive_df = positive_df[positive_df["weight_retention_frac"] >= min_weight_retention]

        if not valid_positive_df.empty:
            best_row = _rank_candidates(valid_positive_df).iloc[0]
        elif not positive_df.empty:
            best_row = _rank_candidates(positive_df).iloc[0]
            logger.warning(
                "No positive threshold met the retention criteria for %s — using highest-retention positive threshold.",
                weight_column,
            )
        else:
            best_row = _rank_candidates(sub_df).iloc[0]
            logger.warning(
                "No positive threshold was available for %s — falling back to the full candidate set.",
                weight_column,
            )

        optimal_threshold = float(best_row["min_edge_weight"])
        optimal_thresholds[weight_column] = optimal_threshold

        logger.info(
            "%s → threshold=%.4f  weight_retained=%.4f  edges_retained=%.4f",
            weight_column,
            optimal_threshold,
            best_row["weight_retention_frac"],
            best_row["edge_retention_frac"],
        )

    return optimal_thresholds


def main(config_path: str | Path = "configs/config.yaml") -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=config_path, help="Path to the YAML configuration.")
    parser.add_argument("--scenario", default="baseline", help="Scenario subdir name, e.g. baseline")
    parser.add_argument("--gamma", type=float, default=0.5, help="Leiden resolution for timing diagnostics")
    parser.add_argument(
        "--min-weight-retention",
        type=float,
        default=0.995,
        help="Minimum weight retention fraction for threshold selection (default: 0.995)",
    )
    args = parser.parse_args()

    with StageContext("sparsify_analysis", args.config) as ctx:
        processed_dir = ctx.data_dir("synthetic")
        results_dir = ctx.results_dir("sparsify")

        min_edge_weights = list(
            ctx.config_value(["network", "min_edge_weights"], [0.0, 0.0001, 0.001, 0.01, 0.1])
        )
        weight_columns = [
            "EpiLinkDeterministicInferenceDeterministicDataScore",
            "EpiLinkDeterministicInferenceStochasticDataScore",
            "EpiLinkStochasticInferenceDeterministicDataScore",
            "EpiLinkStochasticInferenceStochasticDataScore",
            "LogitDeterministicDataProbability",
            "LogitStochasticDataProbability"
        ]
        scenario_dir = processed_dir / f"scenario={args.scenario}"
        scored_pairs = pd.read_parquet(scenario_dir / "pairwise_scored.parquet")

        retention_rows: list[dict[str, object]] = []
        for weight_column in weight_columns:
            reference_threshold = float(min(min_edge_weights))
            reference_frame = sparsify_edges(scored_pairs, reference_threshold, weight_column)
            reference_nodes = pd.Index(pd.unique(reference_frame[["CaseA", "CaseB"]].values.ravel())).astype(str)
            reference_weight = total_edge_weight(reference_frame, weight_column=weight_column)
            reference_edge_count = int(len(reference_frame)) if len(reference_frame) else 0
            metadata = score_metadata(weight_column)

            for threshold in min_edge_weights:
                filtered_pairs, sparsify_seconds = timed(
                    sparsify_edges,
                    scored_pairs,
                    threshold,
                    weight_column,
                )
                retained_weight = total_edge_weight(filtered_pairs, weight_column=weight_column)
                retained_edges = int(len(filtered_pairs))

                build_seconds, leiden_seconds = timed_igraph_and_leiden(
                    filtered_pairs,
                    weight_column=weight_column,
                    vertex_ids=reference_nodes,
                    resolution=args.gamma,
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
        retention_path = results_dir / "sparsify_edge_retention.parquet"
        retention_frame.to_parquet(retention_path, index=False)

        # Determine optimal thresholds from retention results
        optimal_thresholds = determine_optimal_thresholds(retention_frame, args.min_weight_retention)
        thresholds_path = results_dir / "optimal_thresholds.json"
        thresholds_path.write_text(json.dumps(optimal_thresholds, indent=2))
        logger.info("Saved optimal thresholds to: %s", thresholds_path)

        ctx.finish(
            inputs={
                "config": ctx.config_path,
                "scenario_dir": scenario_dir,
            },
            outputs={
                "results_dir": results_dir,
                "retention_path": retention_path,
                "thresholds_path": thresholds_path,
            },
            summary={
                "scenario": args.scenario,
                "weight_columns": weight_columns,
                "num_thresholds": len(min_edge_weights),
                "gamma": args.gamma,
                "min_weight_retention": args.min_weight_retention,
                "optimal_thresholds": optimal_thresholds,
            },
        )


if __name__ == "__main__":
    main()
