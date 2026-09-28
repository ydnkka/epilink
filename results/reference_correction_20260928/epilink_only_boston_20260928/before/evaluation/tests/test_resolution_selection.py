"""Tests for the shared Boston resolution-selection rule."""

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/evaluation")]

from evaluation.specs import MODEL_KEYS
from evaluation.stability import select_shared_resolution


class ResolutionSelectionTests(unittest.TestCase):
    def selection(self):
        return pd.DataFrame([
            {"weight": model, "resolution": resolution, "f1_score": score}
            for model in MODEL_KEYS
            for resolution, score in [(0.3, 0.8), (0.1, 0.4), (0.2, 0.7)]
        ])

    def test_selects_minimum_mean_shortfall(self):
        self.assertEqual(select_shared_resolution(self.selection()), 0.3)

    def test_ties_choose_lower_resolution(self):
        selection = self.selection()
        selection.loc[selection.resolution == 0.2, "f1_score"] = 0.8
        self.assertEqual(select_shared_resolution(selection), 0.2)

    def test_missing_or_nonfinite_scores_are_rejected(self):
        selection = self.selection().iloc[1:]
        with self.assertRaises(ValueError):
            select_shared_resolution(selection)


if __name__ == "__main__":
    unittest.main()
