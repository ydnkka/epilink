import bcubed
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from epilink_evaluation.metrics.pairwise import PairTruth, precision_recall_curve
from epilink_evaluation.metrics.partitions import PartitionEvaluator, ReferenceIndex
from epilink_evaluation.schemas import ScoreSpec
from epilink_evaluation.truth import reference_memberships, relationship_table


@pytest.mark.parametrize("higher_is_better", [True, False])
def test_tied_precision_recall_matches_sklearn(higher_is_better):
    tree = nx.DiGraph([(0, 1), (0, 2), (1, 3), (3, 4)])
    frame = relationship_table(tree)
    truth = PairTruth(frame)
    values = np.array([0, 0, 1, 2, 0, 1, 2, 1, 1, 2], dtype=float)
    spec = ScoreSpec("test", "genetic", "deterministic", higher_is_better=higher_is_better)
    curve, summary = precision_recall_curve(values, spec, truth)
    assert len(curve) == len(np.unique(values))
    assert curve.ties_at_threshold.sum() == len(values)
    for endpoint, horizon in (("M0", 0), ("Mle1", 1), ("Mle2", 2)):
        target = frame.M.le(horizon).to_numpy(bool)
        oriented = values if higher_is_better else -values
        assert summary[f"{endpoint}_AP"] == pytest.approx(average_precision_score(target, oriented))
        for row in curve.to_dict("records"):
            selected = values >= row["threshold"] if higher_is_better else values <= row["threshold"]
            assert row["selected_pairs"] == selected.sum()
            assert row[f"{endpoint}_precision"] == pytest.approx(target[selected].mean())
            assert row[f"{endpoint}_recall"] == pytest.approx(target[selected].sum() / target.sum())


def test_false_positives_include_close_non_targets_and_separate_introductions():
    tree = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4)])
    tree.add_node(5)
    truth = PairTruth(relationship_table(tree))
    all_pairs = truth.statistics(np.ones(15, dtype=bool))
    assert all_pairs["M0_precision"] == pytest.approx(4 / 15)
    assert all_pairs["Mge3_contamination"] == pytest.approx(1 / 15)
    assert all_pairs["separate_fraction"] == pytest.approx(5 / 15)
    empty = truth.statistics(np.zeros(15, dtype=bool))
    assert np.isnan(empty["M0_precision"])
    assert empty["M0_recall"] == empty["M0_f1"] == 0


def test_partitions_evaluate_transitive_pairs_not_only_graph_edges():
    tree = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d")])
    truth = relationship_table(tree)
    observations = truth[["pair_id", "node_a", "node_b"]].rename(
        columns={"node_a": "a", "node_b": "b"}
    )
    cases = pd.DataFrame({"case_id": list(tree)})
    evaluator = PartitionEvaluator(observations, cases, truth, reference_memberships(tree))
    summary, clusters = evaluator.evaluate([0, 0, 0, 0])
    # Three true transmission edges induce six within-cluster pairs.
    assert summary["within_pairs"] == 6
    assert summary["M0_precision"] == 0.5
    assert summary["M0_recall"] == 1
    assert summary["M0_f1"] == pytest.approx(2 / 3)
    assert clusters.within_pairs.sum() == 6
    singleton, _ = evaluator.evaluate([0, 1, 2, 3])
    assert singleton["singleton_fraction"] == 1
    assert singleton["within_pairs"] == 0
    assert np.isnan(singleton["M0_precision"])
    assert singleton["M0_recall"] == 0


@pytest.mark.parametrize("labels", [[0, 0, 0, 0], [0, 1, 2, 3], [0, 0, 1, 1], [7, 3, 7, 7]])
def test_extended_bcubed_matches_independent_package(labels):
    reference = {"a": {0}, "b": {0, 1}, "c": {0, 2}, "d": {1, 3}}
    predicted = {case: {label} for case, label in zip(reference, labels)}
    expected_precision = bcubed.precision(predicted, reference)
    expected_recall = bcubed.recall(predicted, reference)
    actual = ReferenceIndex(list(reference), reference).score(labels)
    assert actual["bcubed_precision"] == pytest.approx(expected_precision)
    assert actual["bcubed_recall"] == pytest.approx(expected_recall)
    assert actual["bcubed_f1"] == pytest.approx(bcubed.fscore(expected_precision, expected_recall))
