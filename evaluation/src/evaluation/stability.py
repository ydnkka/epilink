"""Evaluate partition stability as cases accrue over time."""
from __future__ import annotations

import argparse
import json
import logging
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
    from .leiden import build_weighted_graph, partition_to_frame, run_leiden_partition, subset_pairs_for_nodes
    from .metrics import bcubed_scores, get_reference_memberships, overlap_metrics_between, predict_logistic_scores
except ImportError:
    from config import gamma_mean_cv_to_shape_scale, load_config, resolve_path
    from evaluate import MODEL_KEYS
    from leiden import build_weighted_graph, partition_to_frame, run_leiden_partition, subset_pairs_for_nodes
    from metrics import bcubed_scores, get_reference_memberships, overlap_metrics_between, predict_logistic_scores

logger = logging.getLogger(__name__)

_DEFAULT_SPARSIFICATION = 0.0001

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
# Temporal helpers
# ---------------------------------------------------------------------------

def sampling_times(tree: nx.Graph, step_days: int) -> pd.DataFrame:
    """Assign cases to cumulative availability bins based on rounded sample dates."""
    sampling = {node_id: int(round(sample_date)) for node_id, sample_date in tree.nodes(data="sample_date")}
    case_meta = pd.DataFrame({"node": list(sampling.keys()), "sampling_time": list(sampling.values())})

    time_min = case_meta["sampling_time"].min()
    time_max = case_meta["sampling_time"].max()
    cuts = np.arange(time_min, time_max + step_days, step_days)
    bin_index = (cuts.searchsorted(case_meta["sampling_time"].values, side="right") - 1).clip(0, len(cuts) - 1)

    case_meta["available_bin_start"] = cuts[bin_index]
    case_meta = case_meta.sort_values(["available_bin_start", "sampling_time"]).reset_index(drop=True)
    codes, _ = pd.factorize(case_meta["available_bin_start"], sort=True)
    case_meta["available_time"] = codes
    return case_meta.sort_values("sampling_time").reset_index(drop=True)


def run_partition_for_nodes(
    pairwise_frame: pd.DataFrame,
    *,
    nodes_present: set,
    weight_column: str,
    minimum_weight: float,
    resolution: float,
    num_restarts: int,
    rng_seed: int,
) -> dict[Any, int]:
    """Infer a single Leiden partition for the available cases at one time point."""
    subgraph_pairs = subset_pairs_for_nodes(pairwise_frame, nodes_present)
    graph = build_weighted_graph(
        pairwise_frame=subgraph_pairs,
        weight_column=weight_column,
        minimum_weight=minimum_weight,
        vertex_ids=sorted(nodes_present),
    )
    partition, _ = run_leiden_partition(
        graph,
        weight_column=weight_column,
        resolution=resolution,
        num_restarts=num_restarts,
        rng_seed=rng_seed,
    )
    return dict(zip(graph.vs["case_id"], partition.membership))


def cumulative_stability(
    pairwise_frame: pd.DataFrame,
    case_meta: pd.DataFrame,
    *,
    weight_column: str,
    resolution: float,
    minimum_weight: float,
    num_restarts: int,
    rng_seed: int,
) -> pd.DataFrame:
    """Run cumulative clustering and compare consecutive time-step partitions."""
    transitions = []
    previous_time = None
    previous_labels: dict | None = None

    for current_time in sorted(case_meta["available_time"].unique()):
        nodes_present = set(case_meta.loc[case_meta["available_time"] <= current_time, "node"])
        labels = run_partition_for_nodes(
            pairwise_frame,
            nodes_present=nodes_present,
            weight_column=weight_column,
            minimum_weight=minimum_weight,
            resolution=resolution,
            num_restarts=num_restarts,
            rng_seed=rng_seed,
        )
        if previous_labels is not None:
            overlap_frame = overlap_metrics_between(previous_labels, labels)
            overlap_frame.insert(0, "t", previous_time)
            overlap_frame.insert(1, "t1", current_time)
            transitions.append(overlap_frame)

        previous_time = current_time
        previous_labels = labels

    return pd.concat(transitions, ignore_index=True) if transitions else pd.DataFrame()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_path: str | Path = "config.yaml") -> None:
    parser = argparse.ArgumentParser(description="Temporal partition stability analysis.")
    parser.add_argument("--config", default=config_path, help="Path to YAML configuration.")
    parser.add_argument("--step-days", type=int, default=7, help="Time-bin width in days.")
    parser.add_argument(
        "--train-max-time-index", type=int, default=2,
        help="Latest available_time index used to train logistic classifiers.",
    )
    parser.add_argument(
        "--thresholds", default=None,
        help="Path to optimal_thresholds.json from sparsification analysis (optional).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    fixed = config["fixed_parameters"]
    rng_seed = int(config.get("experiment", {}).get("seed", 12345))
    n_restarts = int(config["execution"]["evaluate_kwargs"].get("n_restarts", 10))
    training_fraction = float(config["execution"]["evaluate_kwargs"].get("training_fraction", 0.1))
    resolution_grid = np.arange(0.1, 1.1, 0.1)

    optimal_thresholds: dict[str, float] = {}
    if args.thresholds is not None:
        thresholds_path = resolve_path(args.thresholds)
        if thresholds_path.exists():
            optimal_thresholds = json.loads(thresholds_path.read_text())

    # ------------------------------------------------------------------
    # 1. Simulate epidemic and build pairwise table
    # ------------------------------------------------------------------
    tree_path = str(resolve_path(config["paths"]["tree_path"]))
    tree = nx.read_gml(tree_path)

    def _nhp(baseline: dict[str, Any]) -> NaturalHistoryParameters:
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

    data_profile = InfectiousnessToTransmission(parameters=_nhp(config["generation_baseline"]), rng_seed=rng_seed)
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
    case_meta = sampling_times(populated_tree, step_days=args.step_days)

    # ------------------------------------------------------------------
    # 2. Score pairs
    # ------------------------------------------------------------------
    infer_profile = InfectiousnessToTransmission(parameters=_nhp(config["inference_baseline"]), rng_seed=rng_seed)
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
    sampling_dates = pairs["SamplingDateDistanceDays"].to_numpy(copy=False)
    for spec in _EPILINK_SPECS:
        pairs[spec["key"]] = np.asarray(
            linkage_models[spec["mutation_process"]].score_target(
                sample_time_difference=sampling_dates,
                genetic_distance=pairs[spec["distance_col"]].to_numpy(copy=False),
            ),
            dtype=float,
        )

    # Train logistic classifiers on early cases only, predict over all pairs.
    initial_nodes = set(case_meta.loc[case_meta["available_time"] <= args.train_max_time_index, "node"])
    initial_pairs = subset_pairs_for_nodes(pairs, initial_nodes)
    y_train = initial_pairs["IsRelated"].astype(int).values
    for spec in _LOGIT_SPECS:
        train_features = initial_pairs[["SamplingDateDistanceDays", spec["distance_col"]]].to_numpy(copy=False)
        predict_features = pairs[["SamplingDateDistanceDays", spec["distance_col"]]].to_numpy(copy=False)
        pairs[spec["key"]], _ = predict_logistic_scores(
            train_features,
            y_train,
            training_fraction=training_fraction,
            rng_seed=rng_seed,
            predict_feature_matrix=predict_features,
        )

    # ------------------------------------------------------------------
    # 3. Select best resolution per model via BCubed F1 on initial cases
    # ------------------------------------------------------------------
    reference = get_reference_memberships(tree_path)
    initial_pairs = subset_pairs_for_nodes(pairs, initial_nodes)

    metric_rows = []
    for key in MODEL_KEYS:
        minimum_weight = optimal_thresholds.get(key, _DEFAULT_SPARSIFICATION)
        graph = build_weighted_graph(
            pairwise_frame=initial_pairs,
            weight_column=key,
            minimum_weight=minimum_weight,
            vertex_ids=sorted(initial_nodes),
        )
        for resolution in resolution_grid:
            partition, _ = run_leiden_partition(
                graph,
                weight_column=key,
                resolution=float(resolution),
                num_restarts=n_restarts,
                rng_seed=rng_seed,
            )
            predicted = {
                int(case_id): {int(cluster_id)}
                for case_id, cluster_id in zip(graph.vs["case_id"], partition.membership)
            }
            try:
                _, _, f1 = bcubed_scores(predicted, reference)
            except ValueError:
                f1 = float("nan")
            metric_rows.append({"weight": key, "resolution": float(resolution), "f1_score": f1})

    evaluation_metrics = pd.DataFrame(metric_rows)
    best_index = evaluation_metrics.groupby("weight")["f1_score"].idxmax()
    model_resolution_map = evaluation_metrics.loc[best_index].set_index("weight")["resolution"].to_dict()

    # ------------------------------------------------------------------
    # 4. Cumulative stability
    # ------------------------------------------------------------------
    results_dir = resolve_path(config["outputs"]["directory"])
    results_dir.mkdir(parents=True, exist_ok=True)

    case_counts = (
        case_meta.groupby("available_time", as_index=False)
        .size()
        .rename(columns={"size": "n_cases"})
    )
    case_counts.to_parquet(results_dir / "case_counts_over_time.parquet", index=False)

    mid_resolution = float(resolution_grid[len(resolution_grid) // 2])
    for key in MODEL_KEYS:
        minimum_weight = optimal_thresholds.get(key, _DEFAULT_SPARSIFICATION)
        resolution = float(model_resolution_map.get(key, mid_resolution))
        logger.info("Computing temporal stability: %s (resolution=%.2f)", key, resolution)

        stability_frame = cumulative_stability(
            pairs,
            case_meta,
            weight_column=key,
            minimum_weight=minimum_weight,
            resolution=resolution,
            num_restarts=n_restarts,
            rng_seed=rng_seed,
        )
        if stability_frame.empty:
            continue

        stability_frame.groupby("t1")[["forward", "backward", "jaccard"]].mean().to_parquet(
            results_dir / f"temporal_stability_{key}.parquet", index=True
        )

    evaluation_metrics.to_parquet(results_dir / "stability_resolution_selection.parquet", index=False)


if __name__ == "__main__":
    main()
