"""Endpoint-oracle graphs, suitable for the existing graph clusterers."""

from __future__ import annotations

from numbers import Integral

import igraph as ig
import numpy as np
import pandas as pd

from ..schemas import ENDPOINTS
from .observations import _aligned_truth


def oracle_graph(
    observations: pd.DataFrame, n_cases: int, truth: pd.DataFrame, endpoint: str
) -> ig.Graph:
    """Connect precisely supplied pairs with finite M <= the endpoint horizon.

    Vertices are sampled-case positions ``0 .. n_cases - 1``, including every
    isolate. Observation ``a``/``b`` indices refer to that universe, not the
    full transmission tree. Pair IDs must align exactly with truth. No feature
    distances are needed. Edges are undirected and have unit ``weight``.

    Only supplied pairs can become edges; use the complete sampled unordered
    pair universe for a complete endpoint oracle. This graph need not be a
    union of cliques: evaluate clusterers' full within-cluster pairs with the
    existing ``PartitionEvaluator`` rather than treating edges as a partition.
    """
    if endpoint not in ENDPOINTS:
        raise ValueError(f"Unknown endpoint: {endpoint}; expected {tuple(ENDPOINTS)}")
    if isinstance(n_cases, bool) or not isinstance(n_cases, Integral) or n_cases < 0:
        raise ValueError("n_cases must be a nonnegative integer")
    pair_truth = _aligned_truth(observations, truth)
    indices = []
    for column in ("a", "b"):
        if column not in observations or observations[column].isna().any():
            raise ValueError(f"Missing sampled-case pair indices: {column}")
        series = observations[column]
        values = series.to_numpy(dtype=getattr(series.dtype, "numpy_dtype", None))
        if values.dtype.kind not in "iu" or np.any((values < 0) | (values >= n_cases)):
            raise ValueError(f"Invalid sampled-case pair indices: {column}")
        indices.append(values)
    a, b = indices
    if np.any(a == b):
        raise ValueError("Self-pairs are not valid oracle edges")
    keep = pair_truth.masks[endpoint]
    graph = ig.Graph(n=int(n_cases), edges=np.column_stack((a[keep], b[keep])))
    if not graph.is_simple():
        raise ValueError("Duplicate unordered target pairs in oracle graph")
    graph.es["weight"] = np.ones(int(keep.sum())).tolist()
    graph["endpoint"] = endpoint
    graph["horizon"] = ENDPOINTS[endpoint]
    return graph


def graph_summary(graph: ig.Graph, *, include_open_wedges: bool = False) -> dict:
    """Summarize a simple undirected oracle graph, including isolates.

    ``n_wedges`` counts centered unordered neighbor pairs, sum choose(degree, 2).
    Basic summaries require only degrees and connected components. Opt into
    ``triangles`` and ``open_wedges`` (n_wedges - 3 * triangles) to also run
    igraph's triangle-counting transitivity calculation. An open wedge is
    counted once per center; it is not a count of distinct missing edges.
    ``n_target_edges`` describes the supplied graph, assumed to be an oracle.
    """
    if graph.is_directed() or not graph.is_simple():
        raise ValueError("Graph diagnostics require a simple undirected graph")
    degrees = np.asarray(graph.degree(), dtype=np.int64)
    sizes = graph.connected_components().sizes()
    n_cases = graph.vcount()
    largest = max(sizes, default=0)
    wedges = int((degrees * (degrees - 1) // 2).sum())
    result = {
        "n_cases": n_cases,
        "n_target_edges": graph.ecount(),
        "n_components": len(sizes),
        "n_isolates": int(np.count_nonzero(degrees == 0)),
        "largest_component": largest,
        "largest_component_fraction": largest / n_cases if n_cases else np.nan,
        "n_wedges": wedges,
    }
    if include_open_wedges:
        # Recover integer counts from igraph's floating-point ratio without
        # materializing a potentially enormous list of triangles.
        transitivity = graph.transitivity_undirected(mode="zero") if wedges else 0.0
        triangles = round(transitivity * wedges / 3)
        result.update(triangles=triangles, open_wedges=wedges - 3 * triangles)
    return result
