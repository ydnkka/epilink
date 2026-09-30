"""Small independent examples protect the scientific definitions and denominators."""
import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from synthetic_exploration.truth import TreeIndex, encode_epilink, canonical_encoding, ENCODING_COLUMNS
from synthetic_exploration.scores import ranking_table, average_precision, thresholds_for_selections
from synthetic_exploration.observations import ambiguity_summary, feature_cells
from synthetic_exploration.clusters import PairLookup, partition_statistics, permute_memberships
from synthetic_exploration.validation import fit_feature_benchmarks
from synthetic_exploration.table import export_analysis_table, read_analysis_table


def table(tree):
    nodes = np.array(list(tree))
    a, b = np.triu_indices(len(nodes), k=1)
    pairs = TreeIndex(tree).annotate(pd.DataFrame({"CaseA": nodes[a], "CaseB": nodes[b]}))
    pairs["pair_id"] = np.arange(len(pairs))
    cases = pd.DataFrame({"case_id": nodes, "node_index": np.arange(len(nodes)), "sampled": True})
    return pairs, cases


class RelationshipTests(unittest.TestCase):
    def setUp(self):
        self.tree = nx.DiGraph([(0, 1), (0, 2), (1, 3), (3, 4), (2, 5)])
        self.tree.add_node(6)

    def test_exact_epilink_encoding_and_inactive_nulls(self):
        pairs, _ = table(self.tree)
        lookup = pairs.set_index(["CaseID1", "CaseID2"])
        for a, b, expected in [(0, 1, (1, 0, 0, None, None)),
                               (0, 4, (1, 0, 2, None, None)),
                               (1, 2, (0, 1, None, 0, 0)),
                               (2, 3, (0, 1, None, 0, 1)),
                               (4, 5, (0, 1, None, 2, 1)),
                               (0, 6, (0, 0, None, None, None))]:
            row = lookup.loc[(a, b)]
            actual = tuple(None if pd.isna(row[c]) else int(row[c]) for c in ["AD", "CA", "m", "m1", "m2"])
            self.assertEqual(actual, expected)

    def test_lca_and_paths_against_networkx(self):
        rng = np.random.default_rng(83)
        tree = nx.DiGraph()
        tree.add_nodes_from(range(35))
        tree.add_edges_from((int(rng.integers(i)), i) for i in range(1, 35))
        pairs, _ = table(tree)
        nodes = list(tree)
        for row in pairs.itertuples():
            ancestor = nx.lowest_common_ancestor(tree, row.CaseID1, row.CaseID2)
            self.assertEqual(nodes[row.lca_index], ancestor)
            self.assertEqual(row.tree_hops, nx.shortest_path_length(tree.to_undirected(), row.CaseID1, row.CaseID2))
            self.assertEqual(row.AD + row.CA, 1)

    def test_CA_relationships_are_invariant_to_case_order(self):
        original = TreeIndex(self.tree).annotate(pd.DataFrame({
            "CaseA": [2, 4, 0, 0], "CaseB": [3, 5, 1, 6]}))
        reversed_pairs = TreeIndex(self.tree).annotate(pd.DataFrame({
            "CaseA": [3, 5, 1, 6], "CaseB": [2, 4, 0, 0]}))
        saved = original.copy(deep=True)
        pd.testing.assert_frame_equal(canonical_encoding(original), canonical_encoding(reversed_pairs))
        pd.testing.assert_frame_equal(original, saved)
        combined = canonical_encoding(pd.concat([original, reversed_pairs], ignore_index=True))
        counts = combined.groupby(ENCODING_COLUMNS, observed=True, dropna=False).size()
        self.assertEqual(counts.tolist(), [2, 2, 2, 2])
        self.assertEqual(tuple(combined.loc[0, ["m1", "m2"]]), (0, 1))
        self.assertEqual(tuple(combined.loc[1, ["m1", "m2"]]), (1, 2))

    def test_same_M_does_not_merge_different_CA_branch_geometries(self):
        pairs = TreeIndex(self.tree).annotate(pd.DataFrame({
            "CaseA": [2, 4, 3], "CaseB": [4, 2, 5]}))
        keys = canonical_encoding(pairs)
        self.assertEqual(keys.M.tolist(), [2, 2, 2])
        counts = keys.groupby(ENCODING_COLUMNS, observed=True, dropna=False).size()
        self.assertEqual(counts.tolist(), [2, 1])  # CA(0,2), CA(1,1)

    def test_case_order_is_not_ancestor_order(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["child", "root", "sibling", "grandchild"])
        tree.add_edges_from([("root", "child"), ("root", "sibling"), ("child", "grandchild")])
        pairs, _ = table(tree)
        row = pairs.iloc[0]
        self.assertEqual((row.CaseID1, row.CaseID2, row.AD, row.m), ("child", "root", 1, 0))
        self.assertEqual(row.lca_index, 1)

    def test_invalid_graphs_rejected(self):
        for edges in [[(1, 2), (2, 1)], [(1, 3), (2, 3)]]:
            with self.assertRaises(ValueError):
                TreeIndex(nx.DiGraph(edges))

    def test_M_unifies_immediate_targets_and_keeps_edge_distance_distinct(self):
        pairs, _ = table(self.tree)
        lookup = pairs.set_index(["CaseID1", "CaseID2"])
        for pair, expected in [((0, 1), 0), ((1, 2), 0), ((0, 4), 2), ((2, 3), 1), ((4, 5), 3)]:
            self.assertEqual(lookup.loc[pair, "M"], expected)
        self.assertTrue(pd.isna(lookup.loc[(0, 6), "M"]))
        np.testing.assert_array_equal(pairs.M.eq(0).fillna(False).to_numpy(bool), pairs.IsRelated)
        connected = pairs.loc[pairs.M.notna()]
        np.testing.assert_array_equal(connected.M + connected.AD + 2 * connected.CA, connected.tree_hops)


class EvaluationTests(unittest.TestCase):
    def test_tied_scores_preserve_full_ties_and_exact_ap(self):
        y = np.array([1, 0, 1, 0, 0, 1])
        score = np.array([.8, .8, .4, .4, .1, .1])
        ranks = ranking_table(y, score)
        self.assertAlmostEqual(average_precision(ranks), average_precision_score(y, score))
        selections = thresholds_for_selections(ranks, {"thresholds": [], "selected_fractions": [.1], "recall_levels": [.5]})
        self.assertEqual(selections[0][2], .8)
        self.assertEqual(int(ranks.iloc[0].selected_pairs), 2)
        self.assertEqual(selections[1][2], .4)

    def test_observation_ambiguity_with_known_collision(self):
        cells = feature_cells(pd.DataFrame({"SamplingDateDistanceDays": [1, 1, 2, 3],
            "StochasticDistance": [0, 0, 1, 2], "IsRelated": [True, False, True, False]}), "StochasticDistance")
        result = ambiguity_summary(cells)
        self.assertEqual(result["mixed_cells"], 1)
        self.assertEqual(result["target_fraction_in_mixed_cells"], .5)
        self.assertEqual(result["minimum_feature_only_misclassifications"], 1)

    def test_cluster_includes_missing_edge_and_singletons_not_pairs(self):
        pairs, cases = table(nx.DiGraph([(0, 1), (1, 2)]))
        lookup = PairLookup(pairs, cases)
        result, clusters, _ = partition_statistics(np.zeros(3, dtype=int), pairs, lookup, [2])
        self.assertEqual(result["within_pairs"], 3)
        self.assertAlmostEqual(result["target_pair_precision"], 2 / 3)
        self.assertEqual(result["direct_edge_retention"], 1)
        self.assertEqual(result["median_M_connected"], 0)
        self.assertEqual(int(clusters.iloc[0].max_M_connected), 1)
        single, _, _ = partition_statistics(np.arange(3), pairs, lookup, [2])
        self.assertEqual(single["within_pairs"], 0)
        self.assertTrue(np.isnan(single["target_pair_precision"]))

    def test_unsampled_intermediates_kept_in_truth(self):
        pairs, cases = table(nx.DiGraph([(0, 1), (1, 2), (2, 3)]))
        cases["sampled"] = cases.case_id.isin([0, 2, 3])
        pairs = pairs.loc[pairs.CaseID1.isin([0, 2, 3]) & pairs.CaseID2.isin([0, 2, 3])].reset_index(drop=True)
        lookup = PairLookup(pairs, cases)
        result, _, _ = partition_statistics(np.zeros(3, dtype=int), pairs, lookup, [2])
        self.assertEqual(result["within_pairs"], 3)
        self.assertEqual(pairs.iloc[0].m, 1)

    def test_temporal_null_preserves_each_cluster_time_histogram(self):
        labels = np.array([0, 1, 0, 2, 1, 2, 2, 0])
        dates = np.array([1, 2, 3, 8, 9, 10, 11, 12])
        permuted = permute_memberships(labels, dates, np.random.default_rng(9), 7)
        for block in [0, 1]:
            mask = dates // 7 == block
            np.testing.assert_array_equal(np.sort(labels[mask]), np.sort(permuted[mask]))

    def test_heldout_predictions_do_not_use_test_labels(self):
        train = pd.DataFrame({"time_days": [0, 2], "genetic_distance": [0, 1],
                              "n_target": [8, 1], "n_other": [2, 9], "n_pairs": [10, 10]})
        test = train.copy()
        first = fit_feature_benchmarks(train, test, 10)
        test["n_target"] = [0, 10]
        test["n_other"] = [10, 0]
        second = fit_feature_benchmarks(train, test, 10)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_unified_table_round_trip_selects_correct_genetics_and_scores(self):
        pairs, _ = table(nx.DiGraph([(0, 1), (0, 2), (1, 3)]))
        pairs["DeterministicDistance"] = np.arange(len(pairs), dtype=np.int32)
        pairs["StochasticDistance"] = np.arange(len(pairs), dtype=np.int32) + 10
        pairs["SamplingDateDistanceDays"] = np.arange(len(pairs), dtype=float) + 20
        scores = pd.DataFrame({"pair_id": pairs.pair_id, "EDD": .2, "ESS": .7})
        run = SimpleNamespace(condition="matched", scenario_name="baseline")
        with tempfile.TemporaryDirectory() as tmp:
            export_analysis_table(pairs, scores, run, 12345, tmp, ["EDD", "ESS"])
            for model, original in [("EDD", "DeterministicDistance"), ("ESS", "StochasticDistance")]:
                result = read_analysis_table(tmp, model=model)
                self.assertEqual(len(result), len(pairs))
                np.testing.assert_array_equal(result.GD, pairs[original])
                np.testing.assert_array_equal(result.TD, pairs.SamplingDateDistanceDays)
                np.testing.assert_array_equal(result.CS, scores[model])
                pd.testing.assert_series_equal(result.m, pairs.m, check_names=False)
                pd.testing.assert_series_equal(result.M, pairs.M, check_names=False)
                self.assertTrue(result.model.eq(model).all())


if __name__ == "__main__":
    unittest.main()
