"""Regression tests for overlapping transmission-neighbourhood references."""

from pathlib import Path
import sys
import tempfile
import unittest

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/evaluation")]

from evaluation.evaluate import _reference_memberships
from evaluation.metrics import bcubed_scores, get_reference_memberships


class ReferenceMembershipTests(unittest.TestCase):
    def tree(self, integer_ids=False):
        labels = [111, 222, 333, 444] if integer_ids else ["111", "222", "333", "444"]
        tree = nx.DiGraph()
        tree.add_nodes_from(labels)
        tree.add_edges_from([(labels[0], labels[1]), (labels[0], labels[2]), (labels[1], labels[3])])
        return tree

    def test_string_and_integer_ids_roots_leaves_and_overlap(self):
        expected = {111: {0}, 222: {0, 1}, 333: {0, 2}, 444: {1, 3}}
        for integer_ids in (False, True):
            with self.subTest(integer_ids=integer_ids):
                self.assertEqual(get_reference_memberships(self.tree(integer_ids)), expected)

    def test_single_node_and_empty_tree(self):
        self.assertEqual(get_reference_memberships(nx.DiGraph()), {})
        tree = nx.DiGraph()
        tree.add_node("12121")
        self.assertEqual(get_reference_memberships(tree), {12121: {0}})

    def test_cached_file_entrypoint_matches_shared_builder(self):
        tree = self.tree()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "tree.gml")
            nx.write_gml(tree, path)
            self.assertEqual(_reference_memberships(path), get_reference_memberships(tree))

    def test_consistent_case_renaming_preserves_scores(self):
        tree = self.tree()
        reference = get_reference_memberships(tree)
        prediction = {111: {0}, 222: {0}, 333: {0}, 444: {1}}
        rename = {"111": "98765", "222": "10010", "333": "55555", "444": "67890"}
        renamed_reference = get_reference_memberships(nx.relabel_nodes(tree, rename))
        renamed_prediction = {int(rename[str(case)]): labels for case, labels in prediction.items()}
        self.assertEqual(
            bcubed_scores(prediction, reference),
            bcubed_scores(renamed_prediction, renamed_reference),
        )
        self.assertEqual(bcubed_scores(reference, reference), (1.0, 1.0, 1.0))

    def test_saved_tree_has_exact_case_universe_and_memberships(self):
        path = ROOT / "results/scovmod/scovmod_tree.gml"
        tree = nx.read_gml(path)
        reference = get_reference_memberships(tree)
        self.assertEqual(len(reference), 4990)
        self.assertEqual(set(reference), {int(node) for node in tree})
        roots = [node for node in tree if tree.in_degree(node) == 0]
        self.assertEqual(roots, ["4537061"])
        cluster_ids = {node: i for i, node in enumerate(tree)}
        for node in tree:
            expected = {cluster_ids[node]} | {cluster_ids[parent] for parent in tree.predecessors(node)}
            self.assertEqual(reference[int(node)], expected)
            self.assertEqual(len(reference[int(node)]), 1 if node in roots else 2)
        self.assertEqual(_reference_memberships(str(path)), reference)


if __name__ == "__main__":
    unittest.main()
