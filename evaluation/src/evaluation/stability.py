"""Evaluate partition stability as cases accrue over time."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from epilink import (
    InfectiousnessToTransmission,
    build_pairwise_case_table,
    simulate_epidemic_dates,
    simulate_genomic_sequences,
)

try:
    from .config import (
        load_config,
        resolve_generation_baseline_parameters,
        resolve_inference_baseline_parameters,
        project_root,
        resolve_path,
    )
    from .leiden import build_weighted_graph, partition_to_frame, run_leiden_partition, subset_pairs_for_nodes
    from .metrics import bcubed_scores, get_reference_memberships, overlap_metrics_between, predict_logistic_scores
    from .models import build_linkage_models, build_natural_history_parameters
    from .specs import (
        DEFAULT_SPARSIFICATION,
        EPILINK_SPECS,
        LOGIT_SPECS,
        MODEL_KEYS,
        PAIRWISE_BOTH_SAMPLED_COLUMN,
        PAIRWISE_RELATED_COLUMN,
        PAIRWISE_TEMPORAL_DISTANCE_COLUMN,
    )
except ImportError:
    from config import (
        load_config,
        resolve_generation_baseline_parameters,
        resolve_inference_baseline_parameters,
        project_root,
        resolve_path,
    )
    from leiden import build_weighted_graph, partition_to_frame, run_leiden_partition, subset_pairs_for_nodes
    from metrics import bcubed_scores, get_reference_memberships, overlap_metrics_between, predict_logistic_scores
    from models import build_linkage_models, build_natural_history_parameters
    from specs import (
        DEFAULT_SPARSIFICATION,
        EPILINK_SPECS,
        LOGIT_SPECS,
        MODEL_KEYS,
        PAIRWISE_BOTH_SAMPLED_COLUMN,
        PAIRWISE_RELATED_COLUMN,
        PAIRWISE_TEMPORAL_DISTANCE_COLUMN,
    )

_DEFAULT_SPARSIFICATION = DEFAULT_SPARSIFICATION
_EPILINK_SPECS = EPILINK_SPECS
_LOGIT_SPECS = LOGIT_SPECS

TREE_PATH = "data/processed/scovmod/scovmod_tree.gml"
RESULTS_DIR = "results/stability"
THRESHOLDS_PATH = "results/sparsification/optimal_thresholds.json"
STEP_DAYS = 7
TRAIN_MAX_TIME_INDEX = 2
N_RESTARTS = 10
TRAINING_FRACTION = 0.1
RNG_SEED = 12345
RESOLUTION_GRID = np.arange(0.1, 1.1, 0.1)


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
    config = load_config(config_path)
    generation_parameters = resolve_generation_baseline_parameters(config)
    inference_parameters = resolve_inference_baseline_parameters(config)

    optimal_thresholds: dict[str, float] = {}
    thresholds_path = resolve_path(THRESHOLDS_PATH)
    if thresholds_path.exists():
        optimal_thresholds = json.loads(thresholds_path.read_text())

    # ------------------------------------------------------------------
    # 1. Simulate epidemic and build pairwise table
    # ------------------------------------------------------------------
    tree_path = str(resolve_path(TREE_PATH))
    tree = nx.read_gml(tree_path)

    data_profile = InfectiousnessToTransmission(
        parameters=build_natural_history_parameters(generation_parameters),
        rng_seed=RNG_SEED,
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
    case_meta = sampling_times(populated_tree, step_days=STEP_DAYS)

    # ------------------------------------------------------------------
    # 2. Score pairs
    # ------------------------------------------------------------------
    linkage_models = build_linkage_models(inference_parameters, rng_seed=RNG_SEED)
    sampling_dates = pairs[PAIRWISE_TEMPORAL_DISTANCE_COLUMN].to_numpy(copy=False)
    for spec in _EPILINK_SPECS:
        pairs[spec["key"]] = np.asarray(
            linkage_models[spec["mutation_process"]].score_target(
                sample_time_difference=sampling_dates,
                genetic_distance=pairs[spec["distance_col"]].to_numpy(copy=False),
            ),
            dtype=float,
        )

    # Train logistic classifiers on early cases only, predict over all pairs.
    initial_nodes = set(case_meta.loc[case_meta["available_time"] <= TRAIN_MAX_TIME_INDEX, "node"])
    initial_pairs = subset_pairs_for_nodes(pairs, initial_nodes)
    y_train = initial_pairs[PAIRWISE_RELATED_COLUMN].astype(int).values
    for spec in _LOGIT_SPECS:
        train_features = initial_pairs[
            [PAIRWISE_TEMPORAL_DISTANCE_COLUMN, spec["distance_col"]]
        ].to_numpy(copy=False)
        predict_features = pairs[
            [PAIRWISE_TEMPORAL_DISTANCE_COLUMN, spec["distance_col"]]
        ].to_numpy(copy=False)
        pairs[spec["key"]], _ = predict_logistic_scores(
            train_features,
            y_train,
            training_fraction=TRAINING_FRACTION,
            rng_seed=RNG_SEED,
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
        for resolution in RESOLUTION_GRID:
            partition, _ = run_leiden_partition(
                graph,
                weight_column=key,
                resolution=float(resolution),
                num_restarts=N_RESTARTS,
                rng_seed=RNG_SEED,
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
    results_dir = resolve_path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    case_counts = (
        case_meta.groupby("available_time", as_index=False)
        .size()
        .rename(columns={"size": "n_cases"})
    )
    case_counts.to_parquet(results_dir / "case_counts_over_time.parquet", index=False)

    mid_resolution = float(RESOLUTION_GRID[len(RESOLUTION_GRID) // 2])
    for key in MODEL_KEYS:
        minimum_weight = optimal_thresholds.get(key, _DEFAULT_SPARSIFICATION)
        resolution = float(model_resolution_map.get(key, mid_resolution))

        stability_frame = cumulative_stability(
            pairs,
            case_meta,
            weight_column=key,
            minimum_weight=minimum_weight,
            resolution=resolution,
            num_restarts=N_RESTARTS,
            rng_seed=RNG_SEED,
        )
        if stability_frame.empty:
            continue

        stability_frame.groupby("t1")[["forward", "backward", "jaccard"]].mean().to_parquet(
            results_dir / f"temporal_stability_{key}.parquet", index=True
        )

    evaluation_metrics.to_parquet(results_dir / "stability_resolution_selection.parquet", index=False)


if __name__ == "__main__":
    main()
