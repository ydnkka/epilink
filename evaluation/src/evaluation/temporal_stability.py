"""Evaluate partition stability as cases accrue over time."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from epilink import (
    EpiLink,
    InfectiousnessToTransmission,
    build_pairwise_case_table,
    simulate_epidemic_dates,
    simulate_genomic_sequences,
)

from epilink_evaluation.config import config_value, ensure_directories
from epilink_evaluation.epilink_config import (
    build_inference_kwargs,
    build_natural_history_parameters,
    build_surveillance_kwargs,
)
from epilink_evaluation.execution import StageContext, load_optimal_thresholds

logger = logging.getLogger(__name__)
from epilink_evaluation.graphs import (
    build_weighted_graph,
    partition_to_frame,
    run_leiden_partition,
    subset_pairs_for_nodes,
)
from epilink_evaluation.logistic import predict_logistic_scores
from epilink_evaluation.metrics import bcubed_scores, build_star_memberships, overlap_metrics_between


def build_transmission_profile(config: dict) -> InfectiousnessToTransmission:
    """Build the transmission profile used for temporal stability simulation."""

    rng_seed = int(config_value(config, ["project", "rng_seed"], 12345))
    return InfectiousnessToTransmission(
        parameters=build_natural_history_parameters(config),
        rng_seed=rng_seed,
    )


def build_linkage_models(config: dict) -> dict[str, EpiLink]:
    """Build deterministic and stochastic EpiLink scorers directly from config."""

    inference_kwargs = build_inference_kwargs(config)
    rng_seed = int(config_value(config, ["project", "rng_seed"], 12345))
    parameters = build_natural_history_parameters(config)
    target_labels = tuple(str(value) for value in inference_kwargs["target_labels"])
    common_kwargs = {
        "transmission_profile": InfectiousnessToTransmission(parameters=parameters, rng_seed=rng_seed),
        "maximum_depth": int(inference_kwargs["maximum_depth"]),
        "mc_samples": int(inference_kwargs["num_simulations"]),
        "target": target_labels,
    }
    return {
        mutation_process: EpiLink(mutation_process=mutation_process, **common_kwargs)
        for mutation_process in ("deterministic", "stochastic")
    }


def sampling_times(tree: nx.Graph, step_days: int) -> pd.DataFrame:
    """Assign cases to cumulative availability bins based on rounded sample dates."""

    sampling = {node_id: int(round(sample_date)) for node_id, sample_date in tree.nodes(data="sample_date")}
    case_meta = pd.DataFrame({"node": list(sampling.keys()), "sampling_time": list(sampling.values())})

    time_min = case_meta["sampling_time"].min()
    time_max = case_meta["sampling_time"].max()
    cuts = np.arange(time_min, time_max + step_days, step_days)
    bin_index = cuts.searchsorted(case_meta["sampling_time"].values, side="right") - 1
    bin_index = bin_index.clip(0, len(cuts) - 1)

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
) -> dict:
    """Infer a single Leiden partition for the available cases at one time point."""

    subgraph_pairs = subset_pairs_for_nodes(pairwise_frame, nodes_present)
    graph = build_weighted_graph(
        subgraph_pairs,
        weight_column=weight_column,
        minimum_weight=minimum_weight,
        vertex_ids=sorted(nodes_present),
    )
    partition, _ = run_leiden_partition(
        graph,
        weight_column=weight_column,
        resolution=resolution,
        num_restarts=num_restarts,
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
) -> pd.DataFrame:
    """Run cumulative clustering and compare consecutive weekly partitions."""

    transitions = []
    previous_time = None
    previous_labels = None

    for current_time in sorted(case_meta["available_time"].unique()):
        nodes_present = set(case_meta.loc[case_meta["available_time"] <= current_time, "node"])
        labels = run_partition_for_nodes(
            pairwise_frame,
            nodes_present=nodes_present,
            weight_column=weight_column,
            minimum_weight=minimum_weight,
            resolution=resolution,
            num_restarts=num_restarts,
        )
        if previous_labels is not None:
            overlap_frame = overlap_metrics_between(previous_labels, labels)
            overlap_frame.insert(0, "t", previous_time)
            overlap_frame.insert(1, "t1", current_time)
            transitions.append(overlap_frame)

        previous_time = current_time
        previous_labels = labels

    return pd.concat(transitions, ignore_index=True) if transitions else pd.DataFrame()


def epilink_specs() -> list[dict[str, str]]:
    """Return EpiLink specifications for temporal stability."""

    return [
        {
            "weight_column": "EpiLinkDeterministicInferenceDeterministicDataScore",
            "distance_column": "DeterministicDistance",
            "mutation_process": "deterministic",
            "output_stem": "epilink_deterministic_inference_deterministic_data",
        },
        {
            "weight_column": "EpiLinkDeterministicInferenceStochasticDataScore",
            "distance_column": "StochasticDistance",
            "mutation_process": "deterministic",
            "output_stem": "epilink_deterministic_inference_stochastic_data",
        },
        {
            "weight_column": "EpiLinkStochasticInferenceDeterministicDataScore",
            "distance_column": "DeterministicDistance",
            "mutation_process": "stochastic",
            "output_stem": "epilink_stochastic_inference_deterministic_data",
        },
        {
            "weight_column": "EpiLinkStochasticInferenceStochasticDataScore",
            "distance_column": "StochasticDistance",
            "mutation_process": "stochastic",
            "output_stem": "epilink_stochastic_inference_stochastic_data",
        },
    ]


def logit_specs() -> list[dict[str, str]]:
    """Return logistic-baseline specifications for temporal stability."""

    return [
        {
            "weight_column": "LogitDeterministicDataProbability",
            "distance_column": "DeterministicDistance",
            "output_stem": "logit_deterministic_data",
        },
        {
            "weight_column": "LogitStochasticDataProbability",
            "distance_column": "StochasticDistance",
            "output_stem": "logit_stochastic_data",
        },
    ]


def main(config_path: str | Path = "configs/config.yaml") -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=config_path, help="Path to the YAML configuration.")
    args = parser.parse_args()

    with StageContext("temporal_stability", args.config) as ctx:
        results_dir = ctx.results_dir()
        ensure_directories(results_dir)

        step_days = int(ctx.config_value(["temporal_stability", "case_sampling", "step_days"], 7))
        train_max_time_index = int(ctx.config_value(["temporal_stability", "case_sampling", "train_max_time_index"], 2))
        num_restarts = int(ctx.config_value(["clustering", "n_restarts"], 10))
        gmin = float(ctx.config_value(["clustering", "resolution", "min"], 0.1))
        gmax = float(ctx.config_value(["clustering", "resolution", "max"], 1.0))
        gstep = float(ctx.config_value(["clustering", "resolution", "step"], 0.05))
        resolutions = np.round(np.arange(gmin, gmax + 1e-9, gstep), 10)
        rng_seed = int(ctx.config_value(["project", "rng_seed"], 12345))
        synthetic_genome_length = int(
            ctx.config_value(["data_generation", "model_parameters", "synthetic_genome_length"], 5000)
        )
        logistic_training_fraction = float(
            ctx.config_value(["temporal_stability", "logistic_training_fraction"], 0.1)
        )

        tree_path = ctx.config_value(["backbone", "tree_gml"])
        transmission_tree = nx.read_gml(ctx.resolve_path(tree_path))
        transmission_profile = build_transmission_profile(ctx.config)
        linkage_models = build_linkage_models(ctx.config)

        populated_tree = simulate_epidemic_dates(
            transmission_profile=transmission_profile,
            tree=transmission_tree,
            **build_surveillance_kwargs(ctx.config),
        )

        genomic_outputs = simulate_genomic_sequences(
            transmission_profile=transmission_profile,
            tree=populated_tree,
            genome_length=synthetic_genome_length,
        )
        pairwise_frame = build_pairwise_case_table(genomic_outputs["packed"], populated_tree)

        case_meta = sampling_times(populated_tree, step_days=step_days)
        case_counts = (
            case_meta.groupby("available_time", as_index=False)
            .size()
            .rename(columns={"size": "n_cases"})
        )
        ensure_directories(results_dir)
        case_counts.to_parquet(results_dir / "case_counts_over_time.parquet", index=False)

        epilink_method_specs = epilink_specs()
        logit_method_specs = logit_specs()

        for spec in epilink_method_specs:
            pairwise_frame[spec["weight_column"]] = np.asarray(
                linkage_models[spec["mutation_process"]].score_target(
                    sample_time_difference=pairwise_frame["SamplingDateDistanceDays"].values,
                    genetic_distance=pairwise_frame[spec["distance_column"]].values,
                ),
                dtype=float,
            )

        initial_nodes = set(case_meta.loc[case_meta["available_time"] <= train_max_time_index, "node"])
        initial_pairs = subset_pairs_for_nodes(pairwise_frame, initial_nodes)
        y = initial_pairs["IsRelated"].astype(int).values
        for spec in logit_method_specs:
            train_features = initial_pairs[["SamplingDateDistanceDays", spec["distance_column"]]].values
            predict_features = pairwise_frame[["SamplingDateDistanceDays", spec["distance_column"]]].values
            pairwise_frame[spec["weight_column"]] = predict_logistic_scores(
                train_features,
                y,
                training_fraction=logistic_training_fraction,
                rng_seed=rng_seed,
                predict_feature_matrix=predict_features,
            )
        initial_pairs = subset_pairs_for_nodes(pairwise_frame, initial_nodes)

        truth = build_star_memberships(ctx.resolve_path(tree_path))
        method_specs = epilink_specs() + logit_specs()
        weight_columns = [spec["weight_column"] for spec in method_specs]

        minimum_weight_by_column = load_optimal_thresholds(ctx, weight_columns)

        metric_rows = []
        partition_rows = []
        for weight_column in weight_columns:
            graph = build_weighted_graph(
                initial_pairs,
                weight_column=weight_column,
                minimum_weight=minimum_weight_by_column[weight_column],
                vertex_ids=sorted(initial_nodes),
            )
            for resolution in resolutions:
                partition, _ = run_leiden_partition(
                    graph,
                    weight_column=weight_column,
                    resolution=float(resolution),
                    num_restarts=num_restarts,
                )
                partition_rows.append(
                    partition_to_frame(
                        graph,
                        partition,
                        weight_column=weight_column,
                        resolution=float(resolution),
                    )
                )

        partitions = pd.concat(partition_rows, ignore_index=True) if partition_rows else pd.DataFrame()
        for (weight_column, resolution), subframe in partitions.groupby(["weight_col", "resolution"], observed=True):
            predicted = {
                int(case_id): {int(cluster_id)}
                for case_id, cluster_id in zip(subframe["case_id"].tolist(), subframe["cluster_id"].tolist())
            }
            precision, recall, f1_score = bcubed_scores(predicted, truth)
            metric_rows.append(
                {
                    "Resolution": resolution,
                    "Weight": weight_column,
                    "BCubed_Precision": precision,
                    "BCubed_Recall": recall,
                    "BCubed_F1_Score": f1_score,
                    "N_cases": len(predicted),
                }
            )

        evaluation_metrics = pd.DataFrame(metric_rows)
        best_index = evaluation_metrics.groupby("Weight")["BCubed_F1_Score"].idxmax()
        best_models = evaluation_metrics.loc[best_index, ["Weight", "Resolution"]]
        model_resolution_map = best_models.set_index("Weight")["Resolution"].to_dict()

        methods = {spec["output_stem"]: {"weight_column": spec["weight_column"]} for spec in method_specs}
        for method_name, method_config in methods.items():
            logger.info("Analysing: %s", method_name)
            stability_frame = cumulative_stability(
                pairwise_frame,
                case_meta,
                weight_column=method_config["weight_column"],
                minimum_weight=minimum_weight_by_column[method_config["weight_column"]],
                resolution=float(model_resolution_map[method_config["weight_column"]]),
                num_restarts=num_restarts,
            )
            stability_summary = stability_frame.groupby("t1")[["forward", "backward", "jaccard"]].mean()
            ensure_directories(results_dir)
            stability_summary.to_parquet(results_dir / f"temporal_stability_{method_name}.parquet", index=True)

        ctx.finish(
            inputs={
                "config": ctx.config_path,
                "tree_path": tree_path,
            },
            outputs={"results_dir": results_dir},
            summary={
                "step_days": step_days,
                "train_max_time_index": train_max_time_index,
                "logistic_training_fraction": logistic_training_fraction,
                "num_cases": len(case_meta),
                "num_initial_cases": len(initial_nodes),
                "num_methods": len(methods),
                "num_resolutions": len(resolutions),
            },
            extra_metadata={"minimum_weight_by_column": minimum_weight_by_column},
        )
