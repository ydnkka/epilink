"""
Pairwise distance table generation utilities.
"""

from __future__ import annotations

from collections.abc import Mapping

import networkx as nx
import numpy as np
import pandas as pd

from ..sequence.packing import PackedGenomicData


def build_pairwise_case_table(
    packed_genomic_data: Mapping[str, PackedGenomicData], tree: nx.DiGraph
) -> pd.DataFrame:
    """Generate a long-format pairwise distance table.

    Parameters
    ----------
    packed_genomic_data : mapping
        Mapping containing packed genomic data for the ``"linear"`` and
        ``"poisson"`` simulation outputs.
    tree : networkx.DiGraph
        Directed transmission tree with epidemic annotations.

    Returns
    -------
    pandas.DataFrame
        Pairwise table of genetic distances, temporal distances, and simple
        topological relationships.
    """

    packed_linear = packed_genomic_data["linear"]
    packed_poisson = packed_genomic_data["poisson"]
    node_map = packed_linear.node_to_idx
    n_nodes = packed_linear.n_seqs
    idx_to_node = {v: k for k, v in node_map.items()}

    mat_linear = packed_linear.compute_hamming_distances()
    mat_poisson = packed_poisson.compute_hamming_distances()

    sample_dates = np.zeros(n_nodes)
    for node, idx in node_map.items():
        sample_dates[idx] = tree.nodes[node].get("sample_date", np.nan)

    diff_matrix = sample_dates[:, np.newaxis] - sample_dates
    mat_temporal = np.abs(diff_matrix).round()

    mat_related = np.zeros((n_nodes, n_nodes), dtype=bool)

    for u, v in tree.edges():
        if u in node_map and v in node_map:
            i, j = node_map[u], node_map[v]
            mat_related[i, j] = True
            mat_related[j, i] = True

    for node in tree.nodes():
        children = list(tree.successors(node))
        if len(children) > 1:
            child_indices = [node_map[c] for c in children if c in node_map]
            if child_indices:
                grid_x, grid_y = np.meshgrid(child_indices, child_indices)
                mat_related[grid_x, grid_y] = True

    rows, cols = np.triu_indices(n_nodes, k=1)
    id_array = np.array([idx_to_node[i] for i in range(n_nodes)])
    sampled_status = np.array([tree.nodes[node].get("sampled", False) for node in id_array])

    return pd.DataFrame(
        {
            "CaseA": id_array[rows],
            "CaseB": id_array[cols],
            "IsRelated": mat_related[rows, cols],
            "BothSampled": sampled_status[rows] & sampled_status[cols],
            "DeterministicDistance": mat_linear[rows, cols],
            "StochasticDistance": mat_poisson[rows, cols],
            "SamplingDateDistanceDays": mat_temporal[rows, cols],
        }
    )


__all__ = ["build_pairwise_case_table"]
