"""Stage 3: Cluster structure analysis with oracle and null baselines."""
from __future__ import annotations

import random
import time

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd

from evaluation.leiden import run_leiden_partition
from evaluation.metrics import bcubed_scores, get_reference_memberships
from synthetic_exploration.clusters import PairLookup

from .common import (
    ENDPOINTS,
    RELATIONSHIP_CATEGORIES,
    log,
    m_category_counts,
    relationship_category_counts,
    save_table,
    score_threshold_for_fraction,
    write_json,
)


def _safe_fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def partition_statistics(labels: np.ndarray, pairs: pd.DataFrame, lookup: PairLookup,
                         weights: np.ndarray | None = None,
                         threshold: float | None = None) -> tuple[dict, pd.DataFrame, list[dict]]:
    """Summarize all within-cluster pairs under M-horizon endpoints and 8 relationship categories."""
    rows, cluster_ids = lookup.within(labels)
    within_mask = np.zeros(len(pairs), dtype=bool)
    within_mask[rows] = True
    
    m = pairs.M.to_numpy(dtype=float, na_value=np.nan)
    within_m = m[rows]
    finite = np.isfinite(within_m)
    
    sizes = np.unique(labels, return_counts=True)[1]
    
    # Direct edge retention
    direct_all = pairs.AD.to_numpy(float, na_value=0) == 1
    direct_within = direct_all[rows] if len(rows) else np.array([], dtype=bool)
    ca00_all = (pairs.AD.to_numpy(float, na_value=0) == 0) & (pairs.CA.to_numpy(float, na_value=0) == 1) & \
               (pairs.m1.to_numpy(float, na_value=0) == 0) & (pairs.m2.to_numpy(float, na_value=0) == 0)
    target_all = direct_all | ca00_all
    target_within = target_all[rows] if len(rows) else np.array([], dtype=bool)
    
    summary = {
        "n_clusters": int(len(sizes)),
        "n_singletons": int(np.sum(sizes == 1)),
        "singleton_case_fraction": float(np.sum(sizes == 1) / len(labels)),
        "largest_cluster": int(sizes.max()) if len(sizes) else 0,
        "within_pairs": int(len(rows)),
        "target_edge_retention": _safe_fraction(int(np.sum(target_within)), int(np.sum(target_all))),
        "median_M_connected": float(np.median(within_m[finite])) if np.any(finite) else np.nan,
        "p90_M_connected": float(np.quantile(within_m[finite], 0.9)) if np.any(finite) else np.nan,
        "Mge3_contamination_fraction": float(np.mean(finite & (within_m >= 3))) if len(rows) else np.nan,
        "undefined_M_fraction": float(np.mean(~finite)) if len(rows) else np.nan,
    }
    
    # M-horizon endpoints
    all_m = m
    for endpoint in ENDPOINTS:
        all_target = np.isfinite(all_m) & (all_m <= endpoint.horizon)
        within_target = finite & (within_m <= endpoint.horizon)
        summary[f"{endpoint.key}_pair_precision"] = float(np.mean(within_target)) if len(rows) else np.nan
        summary[f"{endpoint.key}_pair_recall"] = _safe_fraction(int(np.sum(within_target)), int(np.sum(all_target)))
    
    # Relationship category composition (8 categories)
    cats = relationship_category_counts(pairs, within_mask)
    for cat_key, count in cats.items():
        if cat_key != "unknown":
            summary[f"n_{cat_key}"] = int(count)
            summary[f"fraction_{cat_key}"] = count / len(rows) if len(rows) else np.nan
    
    if weights is not None and threshold is not None:
        retained = weights[rows] >= threshold if len(rows) else np.array([], dtype=bool)
        summary["retained_graph_edges_within_clusters"] = int(np.sum(retained))
        summary["fraction_within_pairs_with_retained_edge"] = float(np.mean(retained)) if len(rows) else np.nan
    
    # Per-cluster rows
    cluster_rows = []
    for cluster, size in zip(*np.unique(labels, return_counts=True)):
        selected = cluster_ids == cluster
        local_m = within_m[selected]
        local_finite = np.isfinite(local_m)
        row = {
            "cluster_id": int(cluster),
            "n_cases": int(size),
            "n_pairs": int(len(local_m)),
            "Mge3_contamination_fraction": float(np.mean(local_finite & (local_m >= 3))) if len(local_m) else np.nan,
            "median_M_connected": float(np.median(local_m[local_finite])) if np.any(local_finite) else np.nan,
            "p90_M_connected": float(np.quantile(local_m[local_finite], 0.9)) if np.any(local_finite) else np.nan,
        }
        for endpoint in ENDPOINTS:
            row[f"{endpoint.key}_pair_precision"] = float(np.mean(local_finite & (local_m <= endpoint.horizon))) if len(local_m) else np.nan
        cluster_rows.append(row)
    
    # Composition rows (8 categories)
    composition_rows = []
    for cat_key, count in cats.items():
        if cat_key != "unknown":
            composition_rows.append({
                "category": cat_key,
                "n_pairs": int(count),
                "proportion": count / len(rows) if len(rows) else np.nan,
            })
    
    return summary, pd.DataFrame(cluster_rows), composition_rows


def component_labels(graph: ig.Graph) -> np.ndarray:
    if graph.vcount() == 0:
        return np.array([], dtype=np.int32)
    return np.asarray(graph.components().membership, dtype=np.int32)


def graph_from_scores(lookup: PairLookup, values: np.ndarray, threshold: float) -> ig.Graph:
    keep = values >= threshold
    graph = ig.Graph(n=lookup.n, edges=np.column_stack((lookup.a[keep], lookup.b[keep])).tolist())
    edge_weights = values[keep].astype(float)
    if len(edge_weights) and edge_weights.min() <= 0:
        edge_weights = edge_weights - edge_weights.min() + 1e-12
    graph.es["weight"] = edge_weights.tolist()
    return graph


def permute_memberships(labels: np.ndarray, dates: np.ndarray, rng: np.random.Generator,
                        temporal_days: int | None = None) -> np.ndarray:
    """Preserve cluster sizes, additionally preserving cluster counts per time bin if requested."""
    if temporal_days is None:
        return rng.permutation(labels)
    result = np.asarray(labels).copy()
    blocks = np.floor(np.asarray(dates) / temporal_days).astype(np.int64)
    for block in np.unique(blocks):
        positions = np.flatnonzero(blocks == block)
        result[positions] = rng.permutation(result[positions])
    return result


def summarise_partition(labels: np.ndarray, base: dict, pairs: pd.DataFrame, cases: pd.DataFrame,
                        lookup: PairLookup, reference: dict, weights: np.ndarray | None = None,
                        threshold: float | None = None) -> tuple[dict, pd.DataFrame, list[dict], pd.DataFrame]:
    summary, clusters, composition = partition_statistics(labels, pairs, lookup, weights, threshold)
    predicted = {int(case): {int(label)} for case, label in zip(lookup.cases.case_id, labels)}
    precision, recall, f1 = bcubed_scores(predicted, reference)
    summary = {**base, **summary, "bcubed_precision": precision,
               "bcubed_recall": recall, "bcubed_f1": f1}
    clusters = clusters.assign(**base)
    composition = [{**base, **row} for row in composition]
    memberships = pd.DataFrame({"case_id": lookup.cases.case_id, "cluster_id": labels, **base})
    return summary, clusters, composition, memberships


def run_null_baselines(labels: np.ndarray, pairs: pd.DataFrame, cases: pd.DataFrame,
                       lookup: PairLookup, reference: dict, settings: dict,
                       seed: int, base: dict) -> list[dict]:
    """Run size-preserving and size+time-preserving null randomizations."""
    null_rows = []
    rng = np.random.default_rng(seed + 1000)
    n_replicates = settings["clusters"]["null_replicates"]
    temporal_days = settings["clusters"]["temporal_block_days"]
    
    for null_kind, window in [("size_preserving", None),
                               ("size_and_time_preserving", temporal_days)]:
        for replicate in range(n_replicates):
            random_labels = permute_memberships(labels, lookup.cases.sample_date.to_numpy(), rng, window)
            summary, _, composition = partition_statistics(random_labels, pairs, lookup)
            predicted = {int(case): {int(label)} for case, label in zip(lookup.cases.case_id, random_labels)}
            precision, recall, f1 = bcubed_scores(predicted, reference)
            
            row = {**base, "null_kind": null_kind, "replicate": replicate,
                   **summary, "bcubed_precision": precision, "bcubed_recall": recall, "bcubed_f1": f1}
            for comp in composition:
                row[f"null_fraction_{comp['category']}"] = comp["proportion"]
            null_rows.append(row)
    
    return null_rows


def investigate_clusters(pairs: pd.DataFrame, cases: pd.DataFrame, candidate_scores: dict,
                         run, directory, settings, seed: int) -> None:
    """Stage 3: Graph clustering, oracle benchmark, null baselines, and TreeCluster."""
    log("3/4: cluster structure, oracle ceiling, and null baselines")
    directory.mkdir(parents=True, exist_ok=True)
    
    lookup = PairLookup(pairs, cases)
    reference = get_reference_memberships(nx.read_gml(run.tree_path))
    summaries, cluster_tables, compositions, memberships, null_rows = [], [], [], [], []
    
    # Oracle target-edge graph (perfect M=0 edges)
    log("Building oracle target-edge graph (M=0 pairs only)")
    target_mask = np.isfinite(pairs.M.to_numpy(dtype=float, na_value=np.nan)) & (pairs.M == 0)
    target_graph = ig.Graph(n=lookup.n, edges=np.column_stack((lookup.a[target_mask], lookup.b[target_mask])).tolist())
    
    ig.set_random_number_generator(random.Random(seed))
    for resolution in settings["clusters"]["leiden_resolutions"]:
        if target_graph.ecount():
            target_partition, target_quality = run_leiden_partition(
                target_graph, None, float(resolution), int(settings["clusters"]["n_restarts"]), seed)
            target_labels = np.asarray(target_partition.membership, dtype=np.int32)
        else:
            target_labels, target_quality = np.arange(lookup.n, dtype=np.int32), np.nan
        
        oracle_base = {
            "score_name": "oracle_target_edges",
            "score_family": "oracle",
            "clustering_method": "leiden",
            "source": "true_M0_edges",
            "data_process": "not_applicable",
            "inference_process": "not_applicable",
            "trained_endpoint": "not_applicable",
            "selection": "oracle_m_equals_zero",
            "requested": np.nan,
            "threshold_inclusive": np.nan,
            "retained_graph_edges": int(target_graph.ecount()),
            "algorithm": "leiden",
            "resolution": float(resolution),
            "restart_selection_modularity": target_quality,
        }
        result = summarise_partition(target_labels, oracle_base, pairs, cases, lookup, reference, None, None)
        summaries.append(result[0])
        cluster_tables.append(result[1])
        compositions.extend(result[2])
        memberships.append(result[3])
    ig.set_random_number_generator(None)
    
    save_table(directory, "oracle_target_partition", [s for s in summaries if s["score_name"] == "oracle_target_edges"])
    
    # Graph clusters from candidate scores
    for candidate in candidate_scores.values():
        values = np.asarray(candidate["values"], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite cluster score: {candidate['name']}")
        
        for fraction in settings["clusters"]["selected_fractions"]:
            threshold = score_threshold_for_fraction(values, float(fraction))
            graph = graph_from_scores(lookup, values, threshold)
            
            base = {
                "score_name": candidate["name"],
                "score_family": candidate["family"],
                "clustering_method": "graph_based",
                "source": candidate["data_process"],
                "data_process": candidate["data_process"],
                "inference_process": candidate["inference_process"],
                "trained_endpoint": candidate["trained_endpoint"],
                "selection": "top_fraction",
                "requested": float(fraction),
                "threshold_inclusive": threshold,
                "retained_graph_edges": int(graph.ecount()),
            }
            
            # Connected components (only at primary fraction for null baselines)
            if "components" in settings["clusters"]["algorithms"]:
                labels = component_labels(graph)
                result = summarise_partition(
                    labels, {**base, "algorithm": "components", "resolution": np.nan,
                             "restart_selection_modularity": np.nan},
                    pairs, cases, lookup, reference, values, threshold)
                summaries.append(result[0])
                cluster_tables.append(result[1])
                compositions.extend(result[2])
                memberships.append(result[3])
                
                # Null baselines only at primary fraction
                if np.isclose(fraction, settings["clusters"]["selected_fractions"][0]):
                    null_rows.extend(run_null_baselines(labels, pairs, cases, lookup, reference, settings, seed,
                                                        {**base, "algorithm": "components", "resolution": np.nan}))
            
            # Leiden (only at primary fraction for null baselines)
            if "leiden" in settings["clusters"]["algorithms"]:
                for resolution in settings["clusters"]["leiden_resolutions"]:
                    log(f"Cluster {candidate['name']}, top_fraction={fraction:g}, Leiden resolution={resolution}")
                    ig.set_random_number_generator(random.Random(seed))
                    if graph.ecount():
                        partition, quality = run_leiden_partition(
                            graph, "weight", float(resolution), int(settings["clusters"]["n_restarts"]), seed)
                        labels = np.asarray(partition.membership, dtype=np.int32)
                    else:
                        labels, quality = np.arange(lookup.n, dtype=np.int32), np.nan
                    
                    result = summarise_partition(
                        labels, {**base, "algorithm": "leiden", "resolution": float(resolution),
                                 "restart_selection_modularity": quality},
                        pairs, cases, lookup, reference, values, threshold)
                    summaries.append(result[0])
                    cluster_tables.append(result[1])
                    compositions.extend(result[2])
                    memberships.append(result[3])
                    
                    # Null baselines only at primary fraction and primary resolution
                    if np.isclose(resolution, settings["clusters"]["leiden_resolutions"][0]) and \
                       np.isclose(fraction, settings["clusters"]["selected_fractions"][0]):
                        null_rows.extend(run_null_baselines(labels, pairs, cases, lookup, reference, settings, seed,
                                                            {**base, "algorithm": "leiden", "resolution": float(resolution)}))
                ig.set_random_number_generator(None)
            
            # Save incrementally
            save_table(directory, "partition_summary", summaries)
            save_table(directory, "within_cluster_M_composition", compositions)
            save_table(directory, "clusters", pd.concat(cluster_tables, ignore_index=True) if cluster_tables else pd.DataFrame())
            save_table(directory, "memberships", pd.concat(memberships, ignore_index=True) if memberships else pd.DataFrame())
            save_table(directory, "null_replicates", null_rows)
    
    # Unified frontier table
    frontier = pd.DataFrame(summaries)
    if not frontier.empty:
        save_table(directory, "cluster_frontier", frontier[[
            "score_name", "score_family", "clustering_method", "source", "algorithm", "requested",
            "resolution", "Mle2_pair_recall", "Mle2_pair_precision",
            "Mge3_contamination_fraction", "n_clusters", "n_singletons", "bcubed_f1",
        ]])
