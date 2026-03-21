from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from epilink import EpiLink, PairCompatibilityResult, Scenario, ScenarioScore  # noqa: E402


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
        self.assertIsInstance(result, PairCompatibilityResult)
        self.assertIsInstance(result.scenario_scores["ad(0)"], ScenarioScore)
        self.assertEqual(result.target_labels, ("ad(0)",))
        self.assertAlmostEqual(result.scenario_scores["ad(0)"].compatibility, 1.0)

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

    def test_draws_by_scenario_setter_invalidates_cached_pairwise_models(self) -> None:
        profile = CountingProfile()
        model = EpiLink(
            transmission_profile=profile,
            maximum_depth=0,
            mc_samples=4,
            mutation_process="deterministic",
        )

        first_model = model.pairwise_model(["ad(0)"])
        replacement_draws = {
            label: {name: values.copy() for name, values in payload.items()}
            for label, payload in self._manual_draws().items()
        }

        model.draws_by_scenario = replacement_draws
        second_model = model.pairwise_model(["ad(0)"])

        self.assertIsNot(first_model, second_model)

    def test_target_accepts_scenario_objects_and_preserves_canonical_order(self) -> None:
        profile = CountingProfile()
        model = EpiLink(
            transmission_profile=profile,
            maximum_depth=0,
            mc_samples=4,
            target=[
                Scenario(kind="ca", branch_to_i=0, branch_to_j=0),
                Scenario(kind="ad", intermediates=0),
            ],
            mutation_process="deterministic",
        )

        self.assertEqual(model.target_labels, ("ad(0)", "ca(0,0)"))
        self.assertIsNone(model.target_label)

    def test_scenario_parse_normalizes_and_round_trips_labels(self) -> None:
        parsed = Scenario.parse("  CA(1,2) ")

        self.assertEqual(parsed, Scenario.common_ancestor(1, 2))
        self.assertEqual(parsed.label(), "ca(1,2)")
        self.assertEqual(str(Scenario.parse("ad(3)")), "ad(3)")

    def test_invalid_scenario_combinations_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            Scenario(kind="ad", branch_to_i=0, branch_to_j=0)

        with self.assertRaises(ValueError):
            Scenario(kind="ca", intermediates=0, branch_to_i=0, branch_to_j=0)

        with self.assertRaises(ValueError):
            Scenario.parse("ca(1)")

        with self.assertRaises(ValueError):
            Scenario.parse("weird(0)")

    def test_score_target_returns_scalar_float_for_scalar_inputs(self) -> None:
        profile = CountingProfile()
        model = EpiLink(
            transmission_profile=profile,
            maximum_depth=0,
            mc_samples=4,
            mutation_process="deterministic",
        )
        model.draws_by_scenario = self._manual_draws()

        score = model.score_target(2.5, 1.5, target="ad(0)")

        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 1.0)

    def test_score_target_broadcasts_scalar_against_vector_inputs(self) -> None:
        profile = CountingProfile()
        model = EpiLink(
            transmission_profile=profile,
            maximum_depth=0,
            mc_samples=4,
            target="ad(0)",
            mutation_process="deterministic",
        )
        model.draws_by_scenario = self._manual_draws()

        scores = model.score_target(2.5, np.array([0.5, 1.5, 2.5]))

        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(scores.shape, (3,))

    def test_deterministic_mutation_process_uses_expected_mutation_counts(self) -> None:
        profile = CountingProfile()
        model = EpiLink(
            transmission_profile=profile,
            maximum_depth=0,
            mc_samples=4,
            mutation_process="deterministic",
        )

        draws = model.draws_by_scenario["ad(0)"]
        expected_branch_draws = np.array([2.0, 2.2, 2.4, 2.6])
        expected_clock_rate_draws = np.array([0.2, 0.21, 0.22, 0.23])

        np.testing.assert_allclose(draws["branch_draws"], expected_branch_draws)
        np.testing.assert_allclose(
            draws["genetic_draws"],
            expected_branch_draws * expected_clock_rate_draws,
        )
        self.assertTrue(np.issubdtype(draws["genetic_draws"].dtype, np.floating))

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

    def test_simulate_common_ancestor_draws_match_manual_expectation(self) -> None:
        profile = CountingProfile()
        model = EpiLink(
            transmission_profile=profile,
            maximum_depth=1,
            mc_samples=3,
            mutation_process="deterministic",
        )

        time_draws, branch_draws = model.simulate_scenario_draws(
            Scenario(kind="ca", branch_to_i=0, branch_to_j=1),
            sample_count=3,
        )

        np.testing.assert_allclose(time_draws, np.array([2.6, 2.8, 3.0]))
        np.testing.assert_allclose(branch_draws, np.array([9.6, 10.5, 11.4]))

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

    def test_invalid_configuration_raises_value_error(self) -> None:
        profile = CountingProfile()

        with self.subTest(parameter="maximum_depth"):
            with self.assertRaises(ValueError):
                EpiLink(
                    transmission_profile=profile,
                    maximum_depth=-1,
                    mc_samples=4,
                    mutation_process="deterministic",
                )

        with self.subTest(parameter="mc_samples"):
            with self.assertRaises(ValueError):
                EpiLink(
                    transmission_profile=profile,
                    maximum_depth=0,
                    mc_samples=0,
                    mutation_process="deterministic",
                )

        with self.subTest(parameter="mutation_process"):
            with self.assertRaises(ValueError):
                EpiLink(
                    transmission_profile=profile,
                    maximum_depth=0,
                    mc_samples=4,
                    mutation_process="not-a-process",
                )

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

    def test_score_pair_requires_both_observations(self) -> None:
        profile = CountingProfile()
        model = EpiLink(
            transmission_profile=profile,
            maximum_depth=0,
            mc_samples=4,
            mutation_process="deterministic",
        )

        with self.assertRaises(TypeError):
            model.score_pair(sample_time_difference=2.5)

        with self.assertRaises(TypeError):
            model.score_pair(genetic_distance=1.5)

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
