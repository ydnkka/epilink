"""Investigation 3: every co-clustered pair, fragmentation, and conditional nulls."""
from __future__ import annotations

import random

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from evaluation.leiden import run_leiden_partition
from evaluation.metrics import bcubed_scores, get_reference_memberships

from .common import log, save_table
from .truth import RELATIONSHIPS, ENCODING_COLUMNS, canonical_encoding


class PairLookup:
    """Map sampled-case ranks to rows without an n-by-n matrix or dataframe joins."""

    def __init__(self, pairs, cases):
        self.cases = cases.loc[cases.sampled].sort_values("node_index").reset_index(drop=True)
        self.n = len(self.cases)
        if len(pairs) != self.n * (self.n - 1) // 2:
            raise ValueError("Cluster exploration requires every sampled unordered pair.")
        rank = np.full(len(cases), -1, dtype=np.int32)
        rank[self.cases.node_index] = np.arange(self.n)
        self.a, self.b = rank[pairs.index_a], rank[pairs.index_b]
        np.testing.assert_array_equal(self.rows(self.a, self.b), np.arange(len(pairs)))

    def rows(self, a, b):
        a, b = np.minimum(a, b).astype(np.int64), np.maximum(a, b).astype(np.int64)
        return a * (2 * self.n - a - 1) // 2 + b - a - 1

    def within(self, labels):
        rows, clusters = [], []
        for cluster in np.unique(labels):
            members = np.flatnonzero(labels == cluster)
            a, b = np.triu_indices(len(members), k=1)
            rows.append(self.rows(members[a], members[b]))
            clusters.append(np.full(len(a), cluster, dtype=np.int32))
        return np.concatenate(rows), np.concatenate(clusters)


def permute_memberships(labels, dates, rng, temporal_days=None):
    """Preserve sizes, additionally preserving cluster counts per time bin if requested."""
    if temporal_days is None:
        return rng.permutation(labels)
    result = np.asarray(labels).copy()
    blocks = np.floor(np.asarray(dates) / temporal_days).astype(np.int64)
    for block in np.unique(blocks):
        positions = np.flatnonzero(blocks == block)
        result[positions] = rng.permutation(result[positions])
    return result


def partition_statistics(labels, pairs, lookup, horizons, weights=None, cutoff=None):
    rows, cluster_ids = lookup.within(labels)
    # Empty within-cluster sets deliberately give undefined precision/compactness.
    hops = pairs.tree_hops.to_numpy()[rows]
    m_values = pairs.M.iloc[rows].to_numpy(dtype=float, na_value=np.nan)
    rel = pairs.relationship.cat.codes.to_numpy()[rows]
    connected = hops >= 0
    target = rel < 2
    direct_total = int((pairs.relationship.cat.codes.to_numpy() == 0).sum())
    target_total = int(pairs.IsRelated.sum())
    cluster_values, sizes = np.unique(labels, return_counts=True)
    counts = np.bincount(rel, minlength=len(RELATIONSHIPS))
    summary = {
        "n_clusters": len(sizes), "n_singletons": int(np.sum(sizes == 1)),
        "singleton_case_fraction": float(np.sum(sizes == 1) / len(labels)),
        "largest_cluster": int(sizes.max()), "within_pairs": len(rows),
        "target_pair_precision": float(target.mean()) if len(rows) else np.nan,
        "target_pair_recall": int(target.sum()) / target_total if target_total else np.nan,
        "direct_edge_retention": int(counts[0]) / direct_total if direct_total else np.nan,
        "median_tree_hops_connected": float(np.median(hops[connected])) if np.any(connected) else np.nan,
        "p90_tree_hops_connected": float(np.quantile(hops[connected], .9)) if np.any(connected) else np.nan,
        "median_M_connected": float(np.median(m_values[connected])) if np.any(connected) else np.nan,
        "p90_M_connected": float(np.quantile(m_values[connected], .9)) if np.any(connected) else np.nan,
        "separate_introduction_fraction": float(np.mean(~connected)) if len(rows) else np.nan,
    }
    for horizon in horizons:
        summary[f"pair_fraction_within_{horizon}_hops"] = float(np.mean(connected & (hops <= horizon))) if len(rows) else np.nan
    per_cluster = []
    for cluster, size in zip(cluster_values, sizes):
        selected = cluster_ids == cluster
        h, r = hops[selected], rel[selected]
        m = m_values[selected]
        finite = h >= 0
        row = {"cluster_id": int(cluster), "n_cases": int(size), "n_pairs": len(h),
               "target_pair_precision": float(np.mean(r < 2)) if len(h) else np.nan,
               "median_tree_hops_connected": float(np.median(h[finite])) if np.any(finite) else np.nan,
               "p90_tree_hops_connected": float(np.quantile(h[finite], .9)) if np.any(finite) else np.nan,
               "max_tree_hops_connected": int(h[finite].max()) if np.any(finite) else np.nan,
               "median_M_connected": float(np.median(m[finite])) if np.any(finite) else np.nan,
               "p90_M_connected": float(np.quantile(m[finite], .9)) if np.any(finite) else np.nan,
               "max_M_connected": int(m[finite].max()) if np.any(finite) else np.nan,
               "separate_introduction_fraction": float(np.mean(~finite)) if len(h) else np.nan}
        for code, relation in enumerate(RELATIONSHIPS):
            row[f"n_{relation}"] = int(np.sum(r == code))
        if weights is not None:
            observed_weights = weights[rows[selected]]
            row["retained_graph_edges"] = int(np.sum(observed_weights >= cutoff))
            row["fraction_pairs_with_retained_edge"] = float(np.mean(observed_weights >= cutoff)) if len(h) else np.nan
        per_cluster.append(row)
    frame = pd.DataFrame(per_cluster)
    non_singletons = frame.loc[frame.n_cases > 1]
    summary["cluster_mean_target_pair_precision"] = float(non_singletons.target_pair_precision.mean())
    summary["cluster_mean_median_tree_hops_connected"] = float(non_singletons.median_tree_hops_connected.mean())
    summary["cluster_mean_median_M_connected"] = float(non_singletons.median_M_connected.mean())
    summary["non_singleton_clusters"] = len(non_singletons)
    composition = [{"relationship": relationship, "n_pairs": int(count),
                    "pair_weighted_proportion": count / len(rows) if len(rows) else np.nan,
                    "cluster_weighted_proportion": float((non_singletons[f"n_{relationship}"] / non_singletons.n_pairs).mean())}
                   for relationship, count in zip(RELATIONSHIPS, counts)]
    return summary, frame, composition


def investigate_clusters(pairs, cases, scores, run, directory, settings, seed):
    log("3/4: cluster composition, reference diagnostic, and size/time nulls")
    config = settings["clusters"]
    lookup = PairLookup(pairs, cases)
    reference = get_reference_memberships(nx.read_gml(run.tree_path))
    cutoff = float(config["minimum_weight"])
    summaries, compositions, cluster_tables, memberships, nulls, encodings = [], [], [], [], [], []
    names = settings["models"] + ["true_target_graph"]
    for name in names:
        weights = pairs.IsRelated.to_numpy(float) if name == "true_target_graph" else scores[name].to_numpy()
        keep = weights >= (1.0 if name == "true_target_graph" else cutoff)
        graph = ig.Graph(n=lookup.n, edges=np.column_stack((lookup.a[keep], lookup.b[keep])).tolist())
        graph.es["weight"] = weights[keep].tolist()
        for resolution in config["resolutions"]:
            log(f"Cluster {name}, resolution={resolution}")
            # igraph receives an explicit seeded generator; no dependence on external RNG state.
            ig.set_random_number_generator(random.Random(seed))
            if graph.ecount():
                partition, quality = run_leiden_partition(
                    graph, "weight", float(resolution), int(config["n_restarts"]), seed)
                labels = np.asarray(partition.membership, dtype=np.int32)
            else:
                labels, quality = np.arange(lookup.n, dtype=np.int32), np.nan
            base = {"model": name, "resolution": resolution}
            summary, clusters, composition = partition_statistics(
                labels, pairs, lookup, config["hop_horizons"], weights, 1.0 if name == "true_target_graph" else cutoff)
            predicted = {int(case): {int(label)} for case, label in zip(lookup.cases.case_id, labels)}
            precision, recall, f1 = bcubed_scores(predicted, reference)
            summaries.append({**base, **summary, "retained_graph_edges": graph.ecount(),
                "restart_selection_modularity": quality, "bcubed_precision": precision,
                "bcubed_recall": recall, "bcubed_f1": f1})
            compositions.extend({**base, **row} for row in composition)
            within_rows, _ = lookup.within(labels)
            encoded = canonical_encoding(pairs.iloc[within_rows]).groupby(
                ENCODING_COLUMNS, observed=True, dropna=False
            ).size().rename("n_pairs").reset_index()
            encodings.append(encoded.assign(**base))
            cluster_tables.append(clusters.assign(**base))
            memberships.append(pd.DataFrame({"case_id": lookup.cases.case_id,
                "cluster_id": labels, **base}))
            if np.isclose(resolution, config["primary_resolution"]):
                rng = np.random.default_rng(seed + 1000)
                for null_kind, window in [("size_preserving", None),
                                           ("size_and_time_preserving", config["temporal_block_days"])]:
                    for replicate in range(config["null_replicates"]):
                        random_labels = permute_memberships(labels, lookup.cases.sample_date, rng, window)
                        null_summary, _, null_composition = partition_statistics(
                            random_labels, pairs, lookup, config["hop_horizons"])
                        nulls.append({**base, "null_kind": null_kind, "replicate": replicate,
                            **null_summary, **{f"proportion_{row['relationship']}": row["pair_weighted_proportion"] for row in null_composition}})
        # Keep progress reviewable even if a later model fails.
        save_table(directory, "partition_summary", summaries)
        save_table(directory, "relationship_composition", compositions)
        save_table(directory, "clusters", pd.concat(cluster_tables, ignore_index=True))
        save_table(directory, "memberships", pd.concat(memberships, ignore_index=True))
        save_table(directory, "null_replicates", nulls)
        save_table(directory, "within_cluster_epilink_encodings", pd.concat(encodings, ignore_index=True))
        save_table(directory, "within_cluster_M", pd.concat(encodings, ignore_index=True).groupby(
            ["model", "resolution", "AD", "CA", "M"], observed=True, dropna=False
        ).n_pairs.sum().reset_index())
    ig.set_random_number_generator(None)
