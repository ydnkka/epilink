from __future__ import annotations

import random
from typing import Any

import igraph as ig
import numpy as np
import pandas as pd


def build_weighted_graph(
        minimum_weight: float,
        pairwise_frame: pd.DataFrame | None = None,
        weight_column: str | None = None,
        source_column: str = "CaseA",
        target_column: str = "CaseB",
        vertex_ids: list[str] | pd.Index | None = None,
        source_ids: list[Any] | np.ndarray | pd.Index | None = None,
        target_ids: list[Any] | np.ndarray | pd.Index | None = None,
        weights: list[float] | np.ndarray | pd.Series | None = None,
) -> ig.Graph:
    """Build an undirected weighted graph from a pairwise edge table."""

    if pairwise_frame is not None:
        if weight_column is None:
            raise ValueError("weight_column is required when pairwise_frame is provided.")
        source_values = pairwise_frame[source_column].to_numpy(copy=False)
        target_values = pairwise_frame[target_column].to_numpy(copy=False)
        weight_values = pairwise_frame[weight_column].to_numpy(copy=False)
    else:
        if source_ids is None or target_ids is None or weights is None:
            raise ValueError("Provide pairwise_frame or all of source_ids, target_ids, and weights.")
        source_values = np.asarray(source_ids)
        target_values = np.asarray(target_ids)
        weight_values = np.asarray(weights, dtype=float)

    if len(source_values) != len(target_values) or len(source_values) != len(weight_values):
        raise ValueError("source_ids, target_ids, and weights must have the same length.")

    valid_mask = pd.notna(source_values) & pd.notna(target_values) & ~np.isnan(weight_values)
    if minimum_weight > 0:
        valid_mask &= weight_values >= minimum_weight

    edge_tuples = list(
        zip(
            source_values[valid_mask].astype(str, copy=False),
            target_values[valid_mask].astype(str, copy=False),
            weight_values[valid_mask].astype(float, copy=False).tolist(),
        )
    )

    edge_attribute_name = weight_column or "weight"
    graph = ig.Graph.TupleList(
        edges=edge_tuples,
        directed=False,
        vertex_name_attr="case_id",
        edge_attrs=edge_attribute_name,
    )

    if vertex_ids is not None:
        vertex_id_list = [str(vertex_id) for vertex_id in vertex_ids]
        present = set(graph.vs["case_id"]) if graph.vcount() and "case_id" in graph.vs.attributes() else set()
        missing = [vertex_id for vertex_id in vertex_id_list if vertex_id not in present]
        if missing:
            start_index = graph.vcount()
            graph.add_vertices(len(missing))
            if "case_id" not in graph.vs.attributes():
                graph.vs["case_id"] = [None] * graph.vcount()
            for offset, vertex_id in enumerate(missing):
                graph.vs[start_index + offset]["case_id"] = vertex_id

    return graph


def run_leiden_partition(
        graph: ig.Graph,
        weight_column: str,
        resolution: float,
        num_restarts: int,
        rng_seed: int | None = None,
) -> tuple[ig.VertexClustering, float]:
    """Run Leiden repeatedly and keep the highest-modularity partition."""

    if rng_seed is not None:
        random.seed(rng_seed)

    best_partition: ig.VertexClustering | None = None
    best_modularity = -np.inf

    for _ in range(num_restarts):
        partition = graph.community_leiden(
            weights=weight_column,
            resolution=resolution,
            n_iterations=-1,
        )
        modularity = graph.modularity(
            membership=partition,
            weights=weight_column,
            resolution=resolution,
            directed=False,
        )
        if modularity > best_modularity:
            best_modularity = modularity
            best_partition = partition

    if best_partition is None:
        raise RuntimeError("Leiden did not produce a partition.")
    return best_partition, float(best_modularity)


def partition_to_frame(
        graph: ig.Graph,
        partition: ig.VertexClustering,
        weight_column: str,
        resolution: float,
) -> pd.DataFrame:
    """Convert an igraph partition into a tidy output table."""

    return pd.DataFrame(
        {
            "case_id": graph.vs["case_id"],
            "cluster_id": np.asarray(partition.membership, dtype=int),
            "resolution": float(resolution),
            "weight_col": weight_column,
        }
    )


def total_edge_weight(pairwise_frame: pd.DataFrame, *, weight_column: str) -> float:
    """Compute total retained edge weight for a given score column."""

    if pairwise_frame.empty:
        return 0.0
    return float(pairwise_frame[weight_column].sum())


def subset_pairs_for_nodes(pairwise_frame: pd.DataFrame, node_ids: set[Any]) -> pd.DataFrame:
    """Return the induced pairwise subgraph for a set of nodes."""

    mask = pairwise_frame["CaseA"].isin(node_ids) & pairwise_frame["CaseB"].isin(node_ids)
    return pairwise_frame.loc[mask].copy()
