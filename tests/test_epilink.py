from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from epilink import EpiLink  # noqa: E402


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
    @staticmethod
    def _manual_draws() -> dict[str, dict[str, np.ndarray]]:
        return {
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

        model.draws_by_scenario = self._manual_draws()
        model.draws_by_scenario["ca(0,0)"]["time_draws"] = np.array([3.0, 4.0, 5.0, 6.0])
        model.draws_by_scenario["ca(0,0)"]["genetic_draws"] = np.array([4.0, 5.0, 6.0, 7.0])

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
            target=["ca(0,0)", "ad(0)", "ad(0)"],
            mutation_process="deterministic",
        )

        model.draws_by_scenario = self._manual_draws()

        result = model.score_pair(sample_time_difference=2.5, genetic_distance=1.5)

        self.assertEqual(model.target_labels, ("ad(0)", "ca(0,0)"))
        self.assertIsNone(model.target_label)
        self.assertEqual(result["target"], ["ad(0)", "ca(0,0)"])
        self.assertEqual(result["target_labels"], ["ad(0)", "ca(0,0)"])
        self.assertAlmostEqual(result["scenario_scores"]["ad(0)"]["compatibility"], 1.0)
        self.assertAlmostEqual(result["scenario_scores"]["ca(0,0)"]["compatibility"], 0.25)
        self.assertAlmostEqual(result["target_compatibility"], 1.25)
        self.assertAlmostEqual(model.score_target(2.5, 1.5), 1.25)

    def test_pairwise_model_accepts_arraylike_inputs_and_returns_broadcast_shape(self) -> None:
        profile = CountingProfile()
        model = EpiLink(
            transmission_profile=profile,
            maximum_depth=0,
            mc_samples=4,
            target=["ad(0)", "ca(0,0)"],
            mutation_process="deterministic",
        )
        model.draws_by_scenario = self._manual_draws()

        pairwise_model = model.pairwise_model()
        sample_time_difference = np.array([[2.5], [4.5]])
        genetic_distance = np.array([[1.5, 2.5, 3.5]])

        scores = pairwise_model(sample_time_difference, genetic_distance)
        broadcast_time, broadcast_genetic = np.broadcast_arrays(
            sample_time_difference,
            genetic_distance,
        )
        expected = np.empty_like(broadcast_time, dtype=float)

        for index in np.ndindex(expected.shape):
            expected[index] = model.score_pair(
                sample_time_difference=float(broadcast_time[index]),
                genetic_distance=float(broadcast_genetic[index]),
            )["target_compatibility"]

        np.testing.assert_allclose(scores, expected)
        self.assertEqual(scores.shape, expected.shape)
        np.testing.assert_allclose(
            model.score_target(sample_time_difference, genetic_distance),
            expected,
        )

    def test_pairwise_model_caches_equivalent_target_subsets(self) -> None:
        profile = CountingProfile()
        model = EpiLink(
            transmission_profile=profile,
            maximum_depth=0,
            mc_samples=4,
            mutation_process="deterministic",
        )

        first_model = model.pairwise_model(["ca(0,0)", "ad(0)"])
        second_model = model.pairwise_model(["ad(0)", "ca(0,0)", "ad(0)"])

        self.assertIs(first_model, second_model)
        self.assertEqual(first_model.target_labels, ("ad(0)", "ca(0,0)"))

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

    def test_score_pair_rejects_arraylike_inputs(self) -> None:
        profile = CountingProfile()
        model = EpiLink(
            transmission_profile=profile,
            maximum_depth=0,
            mc_samples=4,
            mutation_process="deterministic",
        )

        with self.assertRaises(TypeError):
            model.score_pair(sample_time_difference=np.array([2.5]), genetic_distance=1.5)

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
