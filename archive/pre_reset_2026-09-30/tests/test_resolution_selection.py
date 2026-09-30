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

    def test_logistic_comparators_do_not_influence_selection(self):
        selection = self.selection()
        logistic = selection.weight.isin(["LD", "LS"])
        selection.loc[logistic, "f1_score"] = 0.0
        selection.loc[logistic & (selection.resolution == 0.1), "f1_score"] = 1.0
        self.assertEqual(select_shared_resolution(selection), 0.3)
        self.assertEqual(select_shared_resolution(selection.loc[~logistic]), 0.3)

    def test_missing_or_nonfinite_scores_are_rejected(self):
        selection = self.selection()
        nonfinite = selection.copy()
        nonfinite.loc[0, "f1_score"] = float("nan")
        for invalid in [selection.iloc[1:], selection.loc[selection.weight != "ESS"],
                        nonfinite, selection.iloc[:0]]:
            with self.subTest(rows=len(invalid)):
                with self.assertRaises(ValueError):
                    select_shared_resolution(invalid)


if __name__ == "__main__":
    unittest.main()
