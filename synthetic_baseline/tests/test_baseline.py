"""Focused tests for synthetic baseline assessment."""
import unittest

import numpy as np
import pandas as pd

from synthetic_baseline.common import (
    ENDPOINTS,
    endpoint_mask,
    m_category_counts,
    relationship_category,
    RELATIONSHIP_CATEGORIES,
    score_threshold_for_fraction,
    selected_m_summary,
)


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.pairs = pd.DataFrame({
            "M": [0, 1, 2, 3, 6, np.nan],
            "AD": [1, 1, 1, 1, 0, 0],
            "CA": [0, 0, 0, 0, 1, 0],
            "m1": [np.nan, np.nan, np.nan, np.nan, 0, np.nan],
            "m2": [np.nan, np.nan, np.nan, np.nan, 0, np.nan],
            "SamplingDateDistanceDays": [0, 0, 1, 1, 2, 2],
            "DeterministicDistance": [0, 0, 1, 1, 2, 2],
        })
    
    def test_endpoint_masks_treat_Mge3_as_negative(self):
        masks = {endpoint.key: endpoint_mask(self.pairs, endpoint).tolist() for endpoint in ENDPOINTS}
        self.assertEqual(masks["M0"], [True, False, False, False, False, False])
        self.assertEqual(masks["Mle1"], [True, True, False, False, False, False])
        self.assertEqual(masks["Mle2"], [True, True, True, False, False, False])
    
    def test_M_composition_counts_contamination(self):
        counts = m_category_counts(self.pairs)
        self.assertEqual(counts, {"M0": 1, "M1": 1, "M2": 1, "Mge3": 2, "undefined_M": 1})
        summary = selected_m_summary(self.pairs, np.array([True, True, True, True, False, False]))
        self.assertEqual(summary["selected_pairs"], 4)
        self.assertEqual(summary["Mge3_contamination_fraction"], 0.25)
        self.assertEqual(summary["Mle2_fraction"], 0.75)
    
    def test_tie_aware_top_fraction_threshold(self):
        values = np.array([0.9, 0.8, 0.8, 0.3, 0.1])
        self.assertEqual(score_threshold_for_fraction(values, 0.2), 0.9)
        self.assertEqual(score_threshold_for_fraction(values, 0.4), 0.8)
        self.assertEqual(score_threshold_for_fraction(values, 0.5), 0.8)


class RelationshipCategoryTests(unittest.TestCase):
    def setUp(self):
        # Create test pairs covering all 8 categories
        self.pairs = pd.DataFrame({
            "AD": [1, 1, 1, 1, 0, 0, 0, 0],
            "CA": [0, 0, 0, 0, 1, 1, 1, 1],
            "M": [0, 1, 2, 5, 0, 1, 2, 4],
            "m1": [np.nan, np.nan, np.nan, np.nan, 0, 0, 1, 2],
            "m2": [np.nan, np.nan, np.nan, np.nan, 0, 1, 1, 2],
        })
    
    def test_all_eight_categories_assigned(self):
        cats = relationship_category(self.pairs)
        expected = ["AD0", "AD1", "AD2", "ADge3", "CA00", "CA01", "CA11", "CAge3"]
        self.assertEqual(cats.tolist(), expected)
    
    def test_category_counts_match(self):
        from synthetic_baseline.common import relationship_category_counts
        mask = np.ones(len(self.pairs), dtype=bool)
        counts = relationship_category_counts(self.pairs, mask)
        for cat in ["AD0", "AD1", "AD2", "ADge3", "CA00", "CA01", "CA11", "CAge3"]:
            self.assertEqual(counts.get(cat, 0), 1)


class GraphWeightTests(unittest.TestCase):
    def test_genetic_only_graph_weights_are_non_negative(self):
        """Graph clustering requires non-negative weights even for -GD ranking."""
        from synthetic_baseline.stage_03_clusters import graph_from_scores
        from synthetic_exploration.clusters import PairLookup
        from synthetic_exploration.truth import TreeIndex
        import networkx as nx
        
        # Create minimal lookup using proper TreeIndex annotation
        tree = nx.DiGraph([(0, 1), (1, 2)])
        nodes = np.array(list(tree))
        a, b = np.triu_indices(len(nodes), k=1)
        pairs = TreeIndex(tree).annotate(pd.DataFrame({"CaseA": nodes[a], "CaseB": nodes[b]}))
        pairs["pair_id"] = np.arange(len(pairs))
        cases = pd.DataFrame({"case_id": nodes, "node_index": np.arange(len(nodes)), "sampled": True})
        
        lookup = PairLookup(pairs, cases)
        values = np.array([0.0, -1.0, -2.0])  # Genetic-only scores
        graph = graph_from_scores(lookup, values, -1.0)
        
        self.assertEqual(graph.ecount(), 2)
        self.assertTrue(all(weight >= 0 for weight in graph.es["weight"]))
        self.assertGreater(graph.es["weight"][0], graph.es["weight"][1])  # Order preserved


if __name__ == "__main__":
    unittest.main()
