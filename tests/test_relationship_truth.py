from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from epilink_evaluation.truth import (
    TreeIndex,
    reference_memberships,
    relationship_table,
)


@pytest.mark.parametrize("seed", [11, 29, 47])
def test_relationships_match_independent_graph_paths(seed):
    rng = np.random.default_rng(seed)
    tree = nx.DiGraph()
    # Deliberately use non-topological node order and more than one introduction.
    tree.add_nodes_from(rng.permutation(30).tolist())
    tree.add_edges_from(
        (int(rng.integers(child)), child)
        for child in range(1, 30)
        if child not in (10, 20)
    )
    frame = relationship_table(tree)
    nodes = list(tree)
    undirected = tree.to_undirected()
    assert len(frame) == 30 * 29 // 2
    assert frame.pair_id.tolist() == list(range(len(frame)))
    for row in frame.to_dict("records"):
        a, b = nodes[row["node_a"]], nodes[row["node_b"]]
        if not nx.has_path(undirected, a, b):
            assert row["AD"] == row["CA"] == 0
            assert all(
                pd.isna(value)
                for value in (
                    row["m"],
                    row["m1"],
                    row["m2"],
                    row["M"],
                    row["tree_hops"],
                )
            )
            continue
        hops = nx.shortest_path_length(undirected, a, b)
        assert row["tree_hops"] == hops
        if nx.has_path(tree, a, b) or nx.has_path(tree, b, a):
            assert (row["AD"], row["CA"], row["M"], row["m"]) == (
                1,
                0,
                hops - 1,
                hops - 1,
            )
            assert pd.isna(row["m1"]) and pd.isna(row["m2"])
        else:
            common = (nx.ancestors(tree, a) | {a}) & (nx.ancestors(tree, b) | {b})
            ancestor = min(
                common, key=lambda node: nx.shortest_path_length(tree, node, a)
            )
            assert (row["AD"], row["CA"], row["M"]) == (0, 1, hops - 2)
            assert row["m1"] == nx.shortest_path_length(tree, ancestor, a) - 1
            assert row["m2"] == nx.shortest_path_length(tree, ancestor, b) - 1
            assert pd.isna(row["m"])


def test_reversing_pairs_preserves_m_and_swaps_ca_branches():
    tree = nx.DiGraph([("r", "a"), ("r", "b"), ("b", "c"), ("c", "d")])
    index = TreeIndex(tree)
    a, b = np.triu_indices(len(tree), 1)
    forward, reverse = index.classify(a, b), index.classify(b, a)
    pd.testing.assert_series_equal(forward.M, reverse.M)
    pd.testing.assert_series_equal(forward.m1, reverse.m2, check_names=False)
    pd.testing.assert_series_equal(forward.m2, reverse.m1, check_names=False)
    with pytest.raises(ValueError, match="Self-pairs"):
        index.classify([0], [0])


def test_unsampled_infectors_remain_in_reference_memberships():
    tree: nx.DiGraph[str | int] = nx.DiGraph(
        [("r", "a"), ("r", "b"), ("a", "c"), ("c", "d")]
    )
    selected = ["b", "c", "d"]
    reference = reference_memberships(tree, selected)
    full_reference = reference_memberships(tree)
    assert reference == {case: full_reference[case] for case in selected}
    # The observed b/c pair is CA(0,1); pruning the unobserved a must not shorten it.
    index = TreeIndex(tree)
    row = index.classify([index.lookup["b"]], [index.lookup["c"]]).iloc[0]
    assert row.M == 1
    for a, b in combinations(selected, 2):
        target = (
            tree.has_edge(a, b)
            or tree.has_edge(b, a)
            or bool(set(tree.predecessors(a)) & set(tree.predecessors(b)))
        )
        assert bool(reference[a] & reference[b]) == target


@pytest.mark.parametrize(
    "tree",
    [
        nx.DiGraph(),
        nx.Graph([(0, 1)]),
        nx.DiGraph([(0, 1), (1, 0)]),
        nx.DiGraph([(0, 2), (1, 2)]),
    ],
)
def test_invalid_truth_graphs_are_rejected(tree):
    with pytest.raises(ValueError, match="single-parent forest"):
        TreeIndex(tree)
