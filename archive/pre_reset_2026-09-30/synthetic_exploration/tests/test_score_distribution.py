"""Scientific checks for count weighting, conditioning, and score-bin edges."""
import unittest

import numpy as np
import pandas as pd

from synthetic_exploration.score_distribution import distribution_tables


class ScoreDistributionTests(unittest.TestCase):
    def test_weighted_distribution_zero_boundary_and_class_conditioning(self):
        atoms = pd.DataFrame({"score": [0, .05, .05, .10, .20],
                              "M": [30, 0, 10, 1, 2], "AD": [0, 1, 0, 1, 0],
                              "CA": [1, 0, 1, 0, 1], "n_pairs": [1000, 10, 90, 5, 1]})
        dist, stats, _ = distribution_tables({"test": atoms}, width=.05)
        pooled = dist.loc[dist.relationship_class == "pooled"]
        self.assertEqual(int(pooled.n_pairs.sum()), 1106)
        np.testing.assert_allclose(pooled.groupby("score_band").probability.sum(), 1)
        first = stats.loc[(stats.relationship_class == "pooled") & (stats.score_band == 1)].iloc[0]
        self.assertEqual(first.n_pairs, 100)
        self.assertEqual(first.M_q50, 10)  # uses multiplicities, not two equally weighted rows
        self.assertEqual(first.M0_fraction, .1)
        self.assertEqual(stats.loc[(stats.relationship_class == "AD") & (stats.score_band == 1), "M_q50"].iloc[0], 0)
        empty = stats.loc[(stats.relationship_class == "pooled") & (stats.score_band == 3)].iloc[0]
        self.assertEqual(empty.n_pairs, 0)
        self.assertTrue(np.isnan(empty.M_q50))
        self.assertEqual(pooled.loc[pooled.score_band == 0, "n_pairs"].sum(), 1000)
        self.assertEqual(pooled.loc[pooled.score_band == 2, "n_pairs"].sum(), 5)

    def test_undefined_M_is_explicitly_excluded(self):
        atoms = pd.DataFrame({"score": [.1, .1], "M": [0, np.nan],
                              "AD": [1, 0], "CA": [0, 0], "n_pairs": [7, 13]})
        dist, _, meta = distribution_tables({"test": atoms})
        self.assertEqual(meta["undefined_M_pairs"]["test"], 13)
        self.assertEqual(dist.loc[dist.relationship_class == "pooled", "n_pairs"].sum(), 7)


if __name__ == "__main__":
    unittest.main()
