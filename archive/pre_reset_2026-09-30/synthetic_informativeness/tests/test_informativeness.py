"""Focused checks for horizon targets and contamination accounting."""
import unittest

import numpy as np
import pandas as pd

from synthetic_informativeness.common import (
    ENDPOINTS,
    endpoint_mask,
    m_category_counts,
    score_threshold_for_fraction,
    selected_m_summary,
)
from synthetic_informativeness.clusters import graph_from_scores
from synthetic_informativeness.pairwise import feature_cells_for_target


class TinyLookup:
    n = 3
    a = np.array([0, 0, 1])
    b = np.array([1, 2, 2])


class InformativenessTests(unittest.TestCase):
    def setUp(self):
        self.pairs = pd.DataFrame({
            "M": [0, 1, 2, 3, 6, np.nan],
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
        self.assertEqual(summary["Mge3_contamination_fraction"], .25)
        self.assertEqual(summary["Mle2_fraction"], .75)

    def test_tie_aware_top_fraction_threshold(self):
        values = np.array([.9, .8, .8, .3, .1])
        self.assertEqual(score_threshold_for_fraction(values, .2), .9)
        self.assertEqual(score_threshold_for_fraction(values, .4), .8)
        self.assertEqual(score_threshold_for_fraction(values, .5), .8)

    def test_feature_cells_use_requested_endpoint(self):
        target = endpoint_mask(self.pairs, ENDPOINTS[1])
        cells = feature_cells_for_target(self.pairs, "DeterministicDistance", target)
        first = cells.loc[(cells.time_days == 0) & (cells.genetic_distance == 0)].iloc[0]
        self.assertEqual(first.n_pairs, 2)
        self.assertEqual(first.n_target, 2)
        mixed = cells.loc[(cells.time_days == 1) & (cells.genetic_distance == 1)].iloc[0]
        self.assertEqual(mixed.n_pairs, 2)
        self.assertEqual(mixed.n_target, 0)

    def test_genetic_only_graph_weights_are_non_negative(self):
        graph = graph_from_scores(TinyLookup(), np.array([0.0, -1.0, -2.0]), -1.0)
        self.assertEqual(graph.ecount(), 2)
        self.assertTrue(all(weight >= 0 for weight in graph.es["weight"]))
        self.assertGreater(graph.es["weight"][0], graph.es["weight"][1])


if __name__ == "__main__":
    unittest.main()
