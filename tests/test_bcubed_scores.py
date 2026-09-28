"""Check sparse extended BCubed against the independently implemented package."""

from pathlib import Path
import sys
import unittest

import bcubed
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/evaluation")]
from evaluation.metrics import bcubed_scores


class BCubedTests(unittest.TestCase):
    def test_random_overlapping_and_hard_assignments_match_package(self):
        rng = np.random.default_rng(54321)
        for trial in range(30):
            predicted = {i: set(rng.choice(9, size=1 if trial % 2 else int(rng.integers(1, 4)), replace=False)) for i in range(30)}
            reference = {i: set(rng.choice(11, size=int(rng.integers(1, 5)), replace=False)) for i in range(30)}
            p, r = bcubed.precision(predicted, reference), bcubed.recall(predicted, reference)
            np.testing.assert_allclose(bcubed_scores(predicted, reference), (p, r, bcubed.fscore(p, r)), rtol=1e-14, atol=1e-15)

    def test_intersection_filtering_and_empty_memberships(self):
        predicted = {1: {0}, 2: {0}, 3: set(), 4: {1}, None: {3}}
        reference = {1: {1, 2}, 2: {2}, 3: {3}, 5: {4}, None: {3}}
        p, r = bcubed.precision({1: {0}, 2: {0}}, {1: {1, 2}, 2: {2}}), bcubed.recall({1: {0}, 2: {0}}, {1: {1, 2}, 2: {2}})
        np.testing.assert_allclose(bcubed_scores(predicted, reference), (p, r, bcubed.fscore(p, r)))
        with self.assertRaises(ValueError):
            bcubed_scores({1: set()}, {1: {1}})


if __name__ == "__main__":
    unittest.main()
