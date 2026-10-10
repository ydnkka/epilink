from typing import cast

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from epilink_evaluation.clusterers.graph import components, leiden
from epilink_evaluation.diagnostics import (
    graph_summary,
    observation_diagnostics,
    oracle_graph,
)
from epilink_evaluation.metrics.pairwise import CATEGORIES
from epilink_evaluation.metrics.partitions import PartitionEvaluator
from epilink_evaluation.schemas import DISTANCES, ENDPOINTS
from epilink_evaluation.truth.relationships import TreeIndex


def category_truth(categories):
    definitions = {
        "AD0": (1, 0, 0, None, None),
        "AD1": (1, 0, 1, None, None),
        "AD2": (1, 0, 2, None, None),
        "ADge3": (1, 0, 3, None, None),
        "CA00": (0, 1, 0, 0, 0),
        "CA01": (0, 1, 1, 0, 1),
        "CA02": (0, 1, 2, 0, 2),
        "CA11": (0, 1, 2, 1, 1),
        "CAge3": (0, 1, 3, 1, 2),
        "separate": (0, 0, None, None, None),
    }
    truth = pd.DataFrame(
        [definitions[category] for category in categories],
        columns=["AD", "CA", "M", "m1", "m2"],
    ).astype({"AD": "int8", "CA": "int8", "M": "Int32", "m1": "Int32", "m2": "Int32"})
    truth.insert(0, "pair_id", np.arange(len(truth), dtype=np.int64))
    return truth


def observations_for(truth, gd=None, td=None, stochastic=None):
    n = len(truth)
    return pd.DataFrame(
        {
            "pair_id": truth.pair_id,
            "GD_deterministic": np.zeros(n) if gd is None else gd,
            "GD_stochastic": np.zeros(n) if stochastic is None else stochastic,
            "TD": np.zeros(n) if td is None else td,
        }
    )


def sampled_pairs(tree, case_ids):
    index = TreeIndex(tree)
    a, b = np.triu_indices(len(case_ids), k=1)
    nodes = np.array([index.lookup[case] for case in case_ids], dtype=np.int32)
    truth = index.classify(nodes[a], nodes[b])
    truth.insert(0, "pair_id", np.arange(len(truth), dtype=np.int64))
    observations = observations_for(truth).assign(a=a, b=b)
    return observations, truth


def test_hand_counted_exact_cells_and_mixed_denominators():
    truth = category_truth(
        [
            "AD0",
            "CA00",
            "AD1",
            "AD2",
            "AD0",
            "CA01",
            "CA00",
            "ADge3",
            "CA02",
            "CA11",
            "CAge3",
            "separate",
        ]
    )
    observations = observations_for(
        truth,
        gd=[0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 4],
        td=[0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
    )
    cells, summary, prevalence, relationships = observation_diagnostics(
        observations, truth
    )
    assert len(summary) == 12
    assert set(summary.process) == set(DISTANCES)
    assert set(summary.feature_set) == {"GD", "GD_TD"}
    assert set(summary.endpoint) == set(ENDPOINTS)
    rows = summary.set_index(["process", "feature_set", "endpoint"])
    expected = {
        "GD": {
            "n_cells": 5,
            "mixed_cells": 2,
            "n_pairs_in_mixed_cells": 6,
            "n_target_in_mixed_cells": 3,
            "n_other_in_mixed_cells": 3,
            "mixed_cell_fraction": 2 / 5,
            "pair_fraction_in_mixed_cells": 6 / 12,
            "target_fraction_in_mixed_cells": 3 / 4,
            "target_prevalence_in_mixed_cells": 3 / 6,
            "non_target_fraction_in_mixed_cells": 3 / 8,
            "minimum_feature_only_misclassifications": 3,
            "minimum_feature_only_misclassification_rate": 3 / 12,
            "class_conditional_overlap": 3 / 8,
        },
        "GD_TD": {
            "n_cells": 8,
            "mixed_cells": 1,
            "n_pairs_in_mixed_cells": 3,
            "n_target_in_mixed_cells": 2,
            "n_other_in_mixed_cells": 1,
            "mixed_cell_fraction": 1 / 8,
            "pair_fraction_in_mixed_cells": 3 / 12,
            "target_fraction_in_mixed_cells": 2 / 4,
            "target_prevalence_in_mixed_cells": 2 / 3,
            "non_target_fraction_in_mixed_cells": 1 / 8,
            "minimum_feature_only_misclassifications": 1,
            "minimum_feature_only_misclassification_rate": 1 / 12,
            "class_conditional_overlap": 1 / 8,
        },
    }
    for feature_set, metrics in expected.items():
        row = rows.loc[[("deterministic", feature_set, "M0")]].iloc[0]
        assert row.n_pairs == 12
        assert row.n_target == 4
        assert row.n_other == 8
        for metric, value in metrics.items():
            assert row[metric] == pytest.approx(value), metric
    stochastic = rows.loc[[("stochastic", "GD", "M0")]].iloc[0]
    assert stochastic.n_cells == stochastic.mixed_cells == 1
    assert stochastic.pair_fraction_in_mixed_cells == 1
    assert stochastic.target_fraction_in_mixed_cells == 1
    assert stochastic.target_prevalence_in_mixed_cells == pytest.approx(1 / 3)
    assert stochastic.minimum_feature_only_misclassifications == 4
    assert stochastic.class_conditional_overlap == 1

    assert cells.loc[cells.feature_set == "GD", "TD"].isna().all()
    cell = cells.query(
        "process == 'deterministic' and feature_set == 'GD_TD' "
        "and endpoint == 'M0' and GD == 0 and TD == 0"
    ).iloc[0]
    assert (cell.n_pairs, cell.n_target, cell.n_other, cell.mixed) == (3, 2, 1, True)
    assert cell.target_fraction == pytest.approx(2 / 3)
    assert cell.n_AD0 == cell.n_CA00 == cell.n_AD1 == 1
    category_columns = [f"n_{category}" for category in CATEGORIES]
    np.testing.assert_array_equal(cells[category_columns].sum(axis=1), cells.n_pairs)
    expected_categories = [2, 1, 1, 1, 2, 1, 1, 1, 1, 1]
    for (_, _, endpoint), group in cells.groupby(
        ["process", "feature_set", "endpoint"]
    ):
        assert group.n_pairs.sum() == 12
        assert group.n_target.sum() == {"M0": 4, "Mle1": 6, "Mle2": 9}[str(endpoint)]
        assert group[category_columns].sum().tolist() == expected_categories
    for endpoint, targets in {
        "M0": [2, 1, 1, 0, 0],
        "Mle1": [3, 2, 1, 0, 0],
        "Mle2": [4, 2, 1, 2, 0],
    }.items():
        selected = cells.query(
            "process == 'deterministic' and feature_set == 'GD' "
            "and endpoint == @endpoint"
        )
        assert selected.sort_values("GD").n_target.tolist() == targets
    assert prevalence.endpoint.tolist() == list(ENDPOINTS)
    assert prevalence.n_target.tolist() == [4, 6, 9]
    assert prevalence.n_other.tolist() == [8, 6, 3]
    np.testing.assert_allclose(prevalence.target_prevalence, [4 / 12, 6 / 12, 9 / 12])
    assert relationships.relationship.tolist() == list(CATEGORIES)
    assert relationships.n_pairs.tolist() == expected_categories
    assert relationships.pair_fraction.sum() == pytest.approx(1)
    changed = observations.assign(TD=7, GD_deterministic=8, GD_stochastic=9)
    _, _, changed_prevalence, changed_relationships = observation_diagnostics(
        changed, truth
    )
    pd.testing.assert_frame_equal(prevalence, changed_prevalence)
    pd.testing.assert_frame_equal(relationships, changed_relationships)


@pytest.mark.parametrize("categories", [["ADge3", "separate"], ["AD0", "CA00"], []])
def test_absent_classes_and_empty_universe_have_defined_denominators(categories):
    truth = category_truth(categories)
    cells, summary, prevalence, relationships = observation_diagnostics(
        observations_for(truth), truth
    )
    assert (summary.mixed_cells == 0).all()
    assert (summary.minimum_feature_only_misclassifications == 0).all()
    assert summary.target_prevalence_in_mixed_cells.isna().all()
    assert summary.class_conditional_overlap.isna().all()
    for row in summary.to_dict("records"):
        if row["n_target"]:
            assert row["target_fraction_in_mixed_cells"] == 0
        else:
            assert np.isnan(row["target_fraction_in_mixed_cells"])
        if row["n_other"]:
            assert row["non_target_fraction_in_mixed_cells"] == 0
        else:
            assert np.isnan(row["non_target_fraction_in_mixed_cells"])
        if categories:
            assert (
                row["mixed_cell_fraction"] == row["pair_fraction_in_mixed_cells"] == 0
            )
            assert row["minimum_feature_only_misclassification_rate"] == 0
        else:
            assert np.isnan(row["mixed_cell_fraction"])
            assert np.isnan(row["pair_fraction_in_mixed_cells"])
            assert np.isnan(row["minimum_feature_only_misclassification_rate"])
    if not categories:
        assert cells.empty
        assert {"GD", "TD", "n_target", "mixed", "n_separate"} <= set(cells)
        assert prevalence.target_prevalence.isna().all()
        assert relationships.pair_fraction.isna().all()


def test_feature_keys_keep_saved_precision_and_integer_values():
    truth = category_truth(["AD0", "AD1", "AD0", "AD1"])
    gd = pd.array([2**53, 2**53 + 1, 2**53, 2**53], dtype="Int64")
    stochastic = [0.0, 1.0, 0.0, 0.0]
    td = [0.0, 0.0, 1.0, np.nextafter(1.0, 2.0)]
    observations = observations_for(truth, gd=gd, td=td, stochastic=stochastic)
    cells, summary, _, _ = observation_diagnostics(observations, truth)
    for process, values in [("deterministic", gd), ("stochastic", stochastic)]:
        selected = cells.query("process == @process and endpoint == 'M0'")
        gd_only = selected[selected.feature_set == "GD"]
        assert gd_only.GD.tolist() == sorted(set(values))
        joint = selected[selected.feature_set == "GD_TD"]
        assert set(zip(joint.GD, joint.TD)) == set(zip(values, td))
        assert len(joint) == 4
        assert not joint.mixed.any()
    assert (summary.loc[summary.feature_set == "GD_TD", "n_cells"] == 4).all()


def test_large_integer_td_survives_concatenation_with_gd_only_missing_values():
    truth = category_truth(["AD0", "AD1"])
    observations = observations_for(truth, td=[2**53, 2**53 + 1])
    cells, _, _, _ = observation_diagnostics(observations, truth)
    assert cells.loc[cells.feature_set == "GD", "TD"].isna().all()
    joint = cells[cells.feature_set == "GD_TD"]
    assert set(joint.TD) == {2**53, 2**53 + 1}
    assert (joint.n_pairs == 1).all()


@pytest.mark.parametrize("column", ["TD", *DISTANCES.values()])
@pytest.mark.parametrize("value", [np.nan, np.inf, -1.0])
def test_invalid_distances_are_rejected(column, value):
    truth = category_truth(["AD0"])
    observations = observations_for(truth)
    observations.loc[0, column] = value
    with pytest.raises(ValueError, match="Invalid observed distance"):
        observation_diagnostics(observations, truth)


@pytest.mark.parametrize("problem", ["order", "index", "duplicate", "missing"])
def test_diagnostic_pair_alignment_is_exact(problem):
    observations, truth = sampled_pairs(
        nx.DiGraph([("A", "B"), ("B", "C")]), ["A", "B", "C"]
    )
    if problem == "order":
        truth = truth.iloc[::-1].reset_index(drop=True)
    elif problem == "index":
        truth.index = truth.index + 10
    elif problem == "duplicate":
        observations.loc[1, "pair_id"] = truth.loc[1, "pair_id"] = 0
    else:
        observations["pair_id"] = truth["pair_id"] = [0, 1, None]
    with pytest.raises(ValueError, match="aligned|nonmissing and unique"):
        observation_diagnostics(observations, truth)
    with pytest.raises(ValueError, match="aligned|nonmissing and unique"):
        oracle_graph(observations, 3, truth, "M0")


def test_nontransitive_oracle_graph_and_within_cluster_false_positive():
    tree = nx.DiGraph([("A", "B"), ("B", "C")])
    cases = pd.DataFrame({"case_id": ["A", "B", "C"]})
    observations, truth = sampled_pairs(tree, cases.case_id)
    graph = oracle_graph(observations, 3, truth, "M0")
    assert set(graph.get_edgelist()) == {(0, 1), (1, 2)}
    assert graph.es["weight"] == [1.0, 1.0]
    summary = graph_summary(graph, include_open_wedges=True)
    assert summary == {
        "n_cases": 3,
        "n_target_edges": 2,
        "n_components": 1,
        "n_isolates": 0,
        "largest_component": 3,
        "largest_component_fraction": 1,
        "n_wedges": 1,
        "triangles": 0,
        "open_wedges": 1,
    }
    assert "open_wedges" not in graph_summary(graph)
    labels, _ = components(graph)
    evaluator = PartitionEvaluator(observations, cases, truth)
    metrics, clusters = evaluator.evaluate(labels)
    assert metrics["within_pairs"] == 3
    assert metrics["M0_precision"] == pytest.approx(2 / 3)
    assert metrics["M0_recall"] == 1
    assert clusters.n_AD1.sum() == 1  # A--C is a within-cluster false positive.
    leiden_labels, metadata = leiden(
        graph, resolution=0.1, objective="CPM", restarts=2, seed=7
    )
    assert leiden_labels is not None
    assert len(leiden_labels) == 3
    assert np.isfinite(metadata["quality"])
    evaluator.evaluate(leiden_labels)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_oracle_horizons_unsampled_intermediates_and_isolates(endpoint):
    tree: nx.DiGraph[str] = nx.DiGraph(
        [
            ("R", "A"),
            ("A", "u"),
            ("u", "B"),
            ("B", "C"),
            ("C", "v"),
            ("v", "D"),
            ("R", "S"),
        ]
    )
    tree.add_node("I")
    cases = ["D", "A", "C", "B", "S", "I"]
    observations, truth = sampled_pairs(tree, cases)
    graph = oracle_graph(observations, len(cases), truth, endpoint)
    expected = {frozenset(pair) for pair in [("B", "C"), ("A", "S")]}
    if endpoint in ("Mle1", "Mle2"):
        expected.update(frozenset(pair) for pair in [("A", "B"), ("C", "D")])
    if endpoint == "Mle2":
        expected.update(
            frozenset(pair) for pair in [("A", "C"), ("B", "D"), ("B", "S")]
        )
    actual = {frozenset((cases[a], cases[b])) for a, b in graph.get_edgelist()}
    assert actual == expected
    assert graph.vcount() == len(cases)
    assert graph.degree(cases.index("I")) == 0
    assert graph.es["weight"] == [1.0] * len(expected)
    summary = graph_summary(graph)
    assert summary["n_isolates"] == (2 if endpoint == "M0" else 1)
    assert summary["n_components"] == (4 if endpoint == "M0" else 2)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_oracle_excludes_all_nonfinite_M(endpoint):
    truth = category_truth(["separate"] * 3)
    truth["M"] = [np.nan, np.inf, -np.inf]
    observations = observations_for(truth).assign(a=[0, 0, 1], b=[1, 2, 2])
    graph = oracle_graph(observations, 3, truth, endpoint)
    assert graph.ecount() == 0
    assert graph_summary(graph)["n_isolates"] == 3


@pytest.mark.parametrize("n_cases", [0, 1, 4])
def test_empty_oracle_keeps_every_vertex(n_cases):
    truth = category_truth([])
    observations = observations_for(truth).assign(
        a=pd.Series(dtype="int64"), b=pd.Series(dtype="int64")
    )
    graph = oracle_graph(observations, n_cases, truth, "M0")
    assert graph.vcount() == n_cases
    assert graph.es["weight"] == []
    summary = graph_summary(graph, include_open_wedges=True)
    assert summary["n_components"] == summary["n_isolates"] == n_cases
    assert summary["largest_component"] == (1 if n_cases else 0)
    assert summary["n_target_edges"] == 0
    assert summary["open_wedges"] == summary["triangles"] == 0
    if not n_cases:
        assert np.isnan(summary["largest_component_fraction"])


def test_closed_wedges_and_graph_validation():
    summary = graph_summary(ig.Graph.Full(3), include_open_wedges=True)
    assert summary["n_wedges"] == 3
    assert summary["triangles"] == 1
    assert summary["open_wedges"] == 0
    for graph in [
        ig.Graph(n=2, edges=[(0, 1)], directed=True),
        ig.Graph(n=1, edges=[(0, 0)]),
    ]:
        with pytest.raises(ValueError, match="simple undirected"):
            graph_summary(graph)
    truth = category_truth(["AD0"])
    observations = observations_for(truth).assign(a=0, b=1)
    with pytest.raises(ValueError, match="Unknown endpoint"):
        oracle_graph(observations, 2, truth, "M3")
    for n_cases in (-1, 2.5, True):
        with pytest.raises(ValueError, match="nonnegative integer"):
            oracle_graph(observations, cast(int, n_cases), truth, "M0")
    for a, b in [(-1, 1), (0, 2), (0.5, 1), (None, 1), (0, 0)]:
        with pytest.raises(ValueError, match="indices|Self-pairs"):
            oracle_graph(observations.assign(a=a, b=b), 2, truth, "M0")
    truth = category_truth(["AD0", "AD0"])
    repeated = observations_for(truth).assign(a=[0, 1], b=[1, 0])
    with pytest.raises(ValueError, match="Duplicate unordered target pairs"):
        oracle_graph(repeated, 2, truth, "M0")
