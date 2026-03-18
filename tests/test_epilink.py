from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from epilink.epilink import EpiLink


class CountingProfile:
    def __init__(self) -> None:
        self.rng = np.random.default_rng(1234)
        self.call_counts = {
            "sample_incubation_periods": 0,
            "sample_testing_delays": 0,
            "sample_generation_intervals": 0,
            "sample_clock_rate": 0,
        }

    @staticmethod
    def _sequence(size: int | tuple[int, ...], start: float, step: float = 0.1) -> np.ndarray:
        shape = size if isinstance(size, tuple) else (size,)
        total = int(np.prod(shape))
        values = start + step * np.arange(total, dtype=float)
        return values.reshape(shape)

    def sample_incubation_periods(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        self.call_counts["sample_incubation_periods"] += 1
        return self._sequence(size, start=1.0)

    def sample_testing_delays(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        self.call_counts["sample_testing_delays"] += 1
        return self._sequence(size, start=0.5, step=0.05)

    def sample_generation_intervals(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        self.call_counts["sample_generation_intervals"] += 1
        return self._sequence(size, start=2.0, step=0.2)

    def sample_clock_rate(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        self.call_counts["sample_clock_rate"] += 1
        return self._sequence(size, start=0.2, step=0.01)


class TestEpiLink(unittest.TestCase):
    def test_score_pair_uses_cached_draws(self) -> None:
        profile = CountingProfile()
        model = EpiLink(
            transmission_profile=profile,
            maximum_depth=1,
            mc_samples=8,
            target="ad(0)",
            mutation_process="deterministic",
        )

        call_counts_after_init = dict(profile.call_counts)

        first_result = model.score_pair(sample_time_difference=2.0, genetic_distance=1.0)
        second_result = model.score_pair(sample_time_difference=3.0, genetic_distance=2.0)

        self.assertEqual(profile.call_counts, call_counts_after_init)
        self.assertEqual(first_result["target"], "ad(0)")
        self.assertIn("scenario_scores", second_result)

    def test_score_pair_returns_target_compatibility_and_per_scenario_scores(self) -> None:
        profile = CountingProfile()
        model = EpiLink(
            transmission_profile=profile,
            maximum_depth=0,
            mc_samples=4,
            target="ad(0)",
            mutation_process="deterministic",
        )

        model.draws_by_scenario = {
            "ad(0)": {
                "time_draws": np.array([1.0, 2.0, 3.0, 4.0]),
                "branch_draws": np.array([1.0, 1.0, 1.0, 1.0]),
                "genetic_draws": np.array([0.0, 1.0, 2.0, 3.0]),
            },
            "ca(0,0)": {
                "time_draws": np.array([3.0, 4.0, 5.0, 6.0]),
                "branch_draws": np.array([2.0, 2.0, 2.0, 2.0]),
                "genetic_draws": np.array([4.0, 5.0, 6.0, 7.0]),
            },
        }

        result = model.score_pair(sample_time_difference=2.5, genetic_distance=1.5)

        self.assertAlmostEqual(result["scenario_scores"]["ad(0)"]["time_percentile"], 0.5)
        self.assertAlmostEqual(result["scenario_scores"]["ad(0)"]["genetic_percentile"], 0.5)
        self.assertAlmostEqual(result["scenario_scores"]["ad(0)"]["compatibility"], 1.0)
        self.assertAlmostEqual(result["scenario_scores"]["ca(0,0)"]["compatibility"], 0.0)
        self.assertAlmostEqual(result["target_compatibility"], 1.0)
        self.assertAlmostEqual(model.score_target(2.5, 1.5), 1.0)

    def test_score_pair_sums_compatibility_across_target_subset(self) -> None:
        profile = CountingProfile()
        model = EpiLink(
            transmission_profile=profile,
            maximum_depth=0,
            mc_samples=4,
            target=["ad(0)", "ca(0,0)", "ad(0)"],
            mutation_process="deterministic",
        )

        model.draws_by_scenario = {
            "ad(0)": {
                "time_draws": np.array([1.0, 2.0, 3.0, 4.0]),
                "branch_draws": np.array([1.0, 1.0, 1.0, 1.0]),
                "genetic_draws": np.array([0.0, 1.0, 2.0, 3.0]),
            },
            "ca(0,0)": {
                "time_draws": np.array([2.0, 3.0, 4.0, 5.0]),
                "branch_draws": np.array([2.0, 2.0, 2.0, 2.0]),
                "genetic_draws": np.array([1.0, 2.0, 3.0, 4.0]),
            },
        }

        result = model.score_pair(sample_time_difference=2.5, genetic_distance=1.5)

        self.assertEqual(model.target_labels, ("ad(0)", "ca(0,0)"))
        self.assertIsNone(model.target_label)
        self.assertEqual(result["target"], ["ad(0)", "ca(0,0)"])
        self.assertEqual(result["target_labels"], ["ad(0)", "ca(0,0)"])
        self.assertAlmostEqual(result["scenario_scores"]["ad(0)"]["compatibility"], 1.0)
        self.assertAlmostEqual(result["scenario_scores"]["ca(0,0)"]["compatibility"], 0.25)
        self.assertAlmostEqual(result["target_compatibility"], 1.25)
        self.assertAlmostEqual(model.score_target(2.5, 1.5), 1.25)

    def test_stochastic_mutation_process_precomputes_integer_genetic_draws(self) -> None:
        profile = CountingProfile()
        model = EpiLink(
            transmission_profile=profile,
            maximum_depth=0,
            mc_samples=16,
            mutation_process="stochastic",
        )

        genetic_draws = model.draws_by_scenario["ad(0)"]["genetic_draws"]

        self.assertTrue(np.issubdtype(genetic_draws.dtype, np.integer))

    def test_legacy_target_labels_raise_value_error(self) -> None:
        profile = CountingProfile()

        with self.assertRaises(ValueError):
            EpiLink(
                transmission_profile=profile,
                maximum_depth=0,
                mc_samples=4,
                target="HCA(0,0)",
                mutation_process="deterministic",
            )

    def test_legacy_score_pair_keywords_raise_type_error(self) -> None:
        profile = CountingProfile()
        model = EpiLink(
            transmission_profile=profile,
            maximum_depth=0,
            mc_samples=4,
            mutation_process="deterministic",
        )

        with self.assertRaises(TypeError):
            model.score_pair(t_ij=2.5, g_ij=1.5)

    def test_unknown_target_raises_value_error(self) -> None:
        profile = CountingProfile()

        with self.assertRaises(ValueError):
            EpiLink(
                transmission_profile=profile,
                maximum_depth=0,
                mc_samples=4,
                target="ad(3)",
            )

    def test_empty_target_subset_raises_value_error(self) -> None:
        profile = CountingProfile()

        with self.assertRaises(ValueError):
            EpiLink(
                transmission_profile=profile,
                maximum_depth=0,
                mc_samples=4,
                target=[],
            )


if __name__ == "__main__":
    unittest.main()
