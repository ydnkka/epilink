"""Graph clustering comparisons for pairwise informativeness scores."""
from __future__ import annotations

import random

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd

from evaluation.leiden import run_leiden_partition
from evaluation.metrics import bcubed_scores, get_reference_memberships
from synthetic_exploration.clusters import PairLookup
from synthetic_exploration.common import save_table, log

from .common import ENDPOINTS, m_category_rows, score_threshold_for_fraction


def _safe_fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def partition_statistics(labels: np.ndarray, pairs: pd.DataFrame, lookup: PairLookup,
                         horizons=ENDPOINTS, weights: np.ndarray | None = None,
                         threshold: float | None = None) -> tuple[dict, pd.DataFrame, list[dict]]:
    """Summarise all within-cluster pairs under the M-horizon endpoints."""
    rows, cluster_ids = lookup.within(labels)
    within_mask = np.zeros(len(pairs), dtype=bool)
    within_mask[rows] = True
    m = pairs.M.to_numpy(dtype=float, na_value=np.nan)
    within_m = m[rows]
    finite = np.isfinite(within_m)
    sizes = np.unique(labels, return_counts=True)[1]
    direct_all = pairs.relationship.astype(str).to_numpy() == "direct"
    direct_within = direct_all[rows] if len(rows) else np.array([], dtype=bool)
    summary = {
        "n_clusters": int(len(sizes)),
        "n_singletons": int(np.sum(sizes == 1)),
        "singleton_case_fraction": float(np.sum(sizes == 1) / len(labels)),
        "largest_cluster": int(sizes.max()) if len(sizes) else 0,
        "within_pairs": int(len(rows)),
        "direct_edge_retention": _safe_fraction(int(np.sum(direct_within)), int(np.sum(direct_all))),
        "median_M_connected": float(np.median(within_m[finite])) if np.any(finite) else np.nan,
        "p90_M_connected": float(np.quantile(within_m[finite], 0.9)) if np.any(finite) else np.nan,
        "Mge3_contamination_fraction": float(np.mean(finite & (within_m >= 3))) if len(rows) else np.nan,
        "undefined_M_fraction": float(np.mean(~finite)) if len(rows) else np.nan,
    }
    all_m = m
    for endpoint in horizons:
        all_target = np.isfinite(all_m) & (all_m <= endpoint.horizon)
        within_target = finite & (within_m <= endpoint.horizon)
        summary[f"{endpoint.key}_pair_precision"] = float(np.mean(within_target)) if len(rows) else np.nan
        summary[f"{endpoint.key}_pair_recall"] = _safe_fraction(int(np.sum(within_target)), int(np.sum(all_target)))
    if weights is not None and threshold is not None:
        retained = weights[rows] >= threshold if len(rows) else np.array([], dtype=bool)
        summary["retained_graph_edges_within_clusters"] = int(np.sum(retained))
        summary["fraction_within_pairs_with_retained_edge"] = float(np.mean(retained)) if len(rows) else np.nan

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
        for endpoint in horizons:
            row[f"{endpoint.key}_pair_precision"] = float(np.mean(local_finite & (local_m <= endpoint.horizon))) if len(local_m) else np.nan
        cluster_rows.append(row)
    composition_rows = m_category_rows({}, pairs, within_mask)
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
        # Genetic-only scores are -GD for ranking, and some top genetic-only
        # graphs can be all zero-distance ties. Leiden needs positive weights,
        # so shift selected edges without changing score order.
        edge_weights = edge_weights - edge_weights.min() + 1e-12
    graph.es["weight"] = edge_weights.tolist()
    return graph


def summarise_partition(labels: np.ndarray, base: dict, pairs: pd.DataFrame, cases: pd.DataFrame,
                        lookup: PairLookup, reference: dict, weights: np.ndarray,
                        threshold: float) -> tuple[dict, pd.DataFrame, list[dict], pd.DataFrame]:
    summary, clusters, composition = partition_statistics(labels, pairs, lookup, weights=weights, threshold=threshold)
    predicted = {int(case): {int(label)} for case, label in zip(lookup.cases.case_id, labels)}
    precision, recall, f1 = bcubed_scores(predicted, reference)
    summary = {**base, **summary, "bcubed_precision": precision,
               "bcubed_recall": recall, "bcubed_f1": f1}
    clusters = clusters.assign(**base)
    composition = [{**base, **row} for row in composition]
    memberships = pd.DataFrame({"case_id": lookup.cases.case_id, "cluster_id": labels, **base})
    return summary, clusters, composition, memberships


def investigate_clusters(pairs: pd.DataFrame, cases: pd.DataFrame, candidate_scores: dict,
                         run, directory, settings, seed: int) -> None:
    log("2/3: connected components and Leiden cluster summaries")
    directory.mkdir(parents=True, exist_ok=True)
    lookup = PairLookup(pairs, cases)
    reference = get_reference_memberships(nx.read_gml(run.tree_path))
    summaries, cluster_tables, compositions, memberships = [], [], [], []
    
    # True-target-edge graph diagnostic: Leiden on oracle M=0 edges only
    log("Building true-target-edge graph (M=0 pairs only)")
    target_mask = np.isfinite(pairs.M.to_numpy(dtype=float, na_value=np.nan)) & (pairs.M == 0)
    target_graph = ig.Graph(n=lookup.n, edges=np.column_stack((lookup.a[target_mask], lookup.b[target_mask])).tolist())
    ig.set_random_number_generator(random.Random(seed))
    for oracle_resolution in settings["clusters"]["leiden_resolutions"]:
        if target_graph.ecount():
            target_partition, target_quality = run_leiden_partition(
                target_graph, None, float(oracle_resolution), int(settings["clusters"]["n_restarts"]), seed)
            target_labels = np.asarray(target_partition.membership, dtype=np.int32)
        else:
            target_labels, target_quality = np.arange(lookup.n, dtype=np.int32), np.nan
        oracle_base = {
            "score_name": "oracle_target_edges",
            "score_family": "oracle",
            "data_process": "not_applicable",
            "inference_process": "not_applicable",
            "trained_endpoint": "not_applicable",
            "selection": "oracle_m_equals_zero",
            "requested": np.nan,
            "threshold_inclusive": np.nan,
            "retained_graph_edges": int(target_graph.ecount()),
            "algorithm": "leiden",
            "resolution": float(oracle_resolution),
            "restart_selection_modularity": target_quality,
        }
        result = summarise_partition(
            target_labels, oracle_base, pairs, cases, lookup, reference,
            None, None)
        summaries.append(result[0]); cluster_tables.append(result[1]); compositions.extend(result[2]); memberships.append(result[3])
    save_table(directory, "oracle_target_partition", [s for s in summaries if s["score_name"] == "oracle_target_edges"])
    ig.set_random_number_generator(None)
    for candidate in candidate_scores.values():
        values = np.asarray(candidate.values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite cluster score: {candidate.name}")
        for fraction in settings["clusters"]["selected_fractions"]:
            threshold = score_threshold_for_fraction(values, float(fraction))
            graph = graph_from_scores(lookup, values, threshold)
            base = {
                "score_name": candidate.name,
                "score_family": candidate.family,
                "data_process": candidate.data_process,
                "inference_process": candidate.inference_process,
                "trained_endpoint": candidate.trained_endpoint,
                "selection": "top_fraction",
                "requested": float(fraction),
                "threshold_inclusive": threshold,
                "retained_graph_edges": int(graph.ecount()),
            }
            if "components" in settings["clusters"]["algorithms"]:
                labels = component_labels(graph)
                result = summarise_partition(
                    labels, {**base, "algorithm": "components", "resolution": np.nan,
                             "restart_selection_modularity": np.nan},
                    pairs, cases, lookup, reference, values, threshold)
                summaries.append(result[0]); cluster_tables.append(result[1]); compositions.extend(result[2]); memberships.append(result[3])
            if "leiden" in settings["clusters"]["algorithms"]:
                for resolution in settings["clusters"]["leiden_resolutions"]:
                    log(f"Cluster {candidate.name}, top_fraction={fraction:g}, Leiden resolution={resolution}")
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
                    summaries.append(result[0]); cluster_tables.append(result[1]); compositions.extend(result[2]); memberships.append(result[3])
            save_table(directory, "partition_summary", summaries)
            save_table(directory, "within_cluster_M_composition", compositions)
            save_table(directory, "clusters", pd.concat(cluster_tables, ignore_index=True) if cluster_tables else pd.DataFrame())
            save_table(directory, "memberships", pd.concat(memberships, ignore_index=True) if memberships else pd.DataFrame())
    ig.set_random_number_generator(None)
    frontier = pd.DataFrame(summaries)
    if not frontier.empty:
        save_table(directory, "cluster_frontier", frontier[[
            "score_name", "score_family", "data_process", "algorithm", "requested",
            "resolution", "Mle2_pair_recall", "Mle2_pair_precision",
            "Mge3_contamination_fraction", "n_clusters", "n_singletons", "bcubed_f1",
        ]])
