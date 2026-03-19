from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from epilink import (  # noqa: E402
    InfectiousnessToTransmission,
    NaturalHistoryParameters,
    SymptomOnsetToTransmission,
)
from epilink.profiles import BaseTransmissionProfile  # noqa: E402


class ConstantDensityProfile(BaseTransmissionProfile):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("grid_min_days", 0.0)
        kwargs.setdefault("grid_max_days", 2.0)
        kwargs.setdefault("grid_points", 5)
        super().__init__(**kwargs)
        self.pdf_calls = 0

    def pdf(self, times_in_days: np.typing.ArrayLike) -> np.ndarray:
        self.pdf_calls += 1
        evaluation_points = np.asarray(times_in_days, dtype=float)
        density = np.where(
            (evaluation_points >= self.grid_min_days) & (evaluation_points <= self.grid_max_days),
            0.5,
            0.0,
        )
        return density[()] if evaluation_points.ndim == 0 else density

    def rvs(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        sample_shape = (size,) if isinstance(size, int) else size
        return np.full(sample_shape, 1.0, dtype=float)


class ZeroDensityProfile(BaseTransmissionProfile):
    def __init__(self, **kwargs) -> None:
        super().__init__(grid_min_days=0.0, grid_max_days=2.0, grid_points=5, **kwargs)

    def pdf(self, times_in_days: np.typing.ArrayLike) -> np.ndarray:
        evaluation_points = np.asarray(times_in_days, dtype=float)
        density = np.zeros_like(evaluation_points)
        return density[()] if evaluation_points.ndim == 0 else density

    def rvs(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        sample_shape = (size,) if isinstance(size, int) else size
        return np.zeros(sample_shape, dtype=float)


class TestBaseTransmissionProfile(unittest.TestCase):
    def test_invalid_grid_configuration_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            ConstantDensityProfile(grid_min_days=2.0, grid_max_days=1.0)

        with self.assertRaises(ValueError):
            ConstantDensityProfile(grid_points=1)

    def test_cdf_uses_cached_numerical_grid(self) -> None:
        profile = ConstantDensityProfile()

        first = profile.cdf(np.array([0.0, 1.0, 2.0]))
        second = profile.cdf(np.array([0.5, 1.5]))

        np.testing.assert_allclose(first, np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(second, np.array([0.25, 0.75]))
        self.assertEqual(profile.pdf_calls, 1)

    def test_cdf_falls_back_to_linear_interpolation_when_density_has_no_mass(self) -> None:
        profile = ZeroDensityProfile()

        cdf = profile.cdf(np.array([0.0, 1.0, 2.0]))

        np.testing.assert_allclose(cdf, np.array([0.0, 0.5, 1.0]))

    def test_sample_incubation_periods_adds_latent_and_presymptomatic_periods(self) -> None:
        profile = ConstantDensityProfile()
        latent_periods = np.array([1.0, 2.0, 3.0])
        presymptomatic_periods = np.array([0.25, 0.5, 0.75])

        with (
            patch.object(profile, "sample_latent_periods", return_value=latent_periods),
            patch.object(
                profile,
                "sample_presymptomatic_periods",
                return_value=presymptomatic_periods,
            ),
        ):
            incubation_periods = profile.sample_incubation_periods(size=3)

        np.testing.assert_allclose(incubation_periods, latent_periods + presymptomatic_periods)

    def test_mean_and_repr_reflect_profile_configuration(self) -> None:
        profile = ConstantDensityProfile()

        self.assertTrue(math.isclose(profile.mean(), 1.0, rel_tol=1e-3))
        self.assertIn("ConstantDensityProfile", repr(profile))
        self.assertIn("grid_min_days=0.0", repr(profile))
        self.assertIn("grid_max_days=2.0", repr(profile))

    def test_zero_mass_profile_has_nan_mean(self) -> None:
        profile = ZeroDensityProfile()

        self.assertTrue(math.isnan(profile.mean()))

    def test_strict_clock_sampling_and_expected_mutations_are_deterministic(self) -> None:
        parameters = NaturalHistoryParameters(
            substitution_rate=2.0e-3,
            relaxation=0.0,
            genome_length=1000,
        )
        profile = ConstantDensityProfile(parameters=parameters, rng_seed=123)

        expected_rate_per_day = (parameters.substitution_rate * parameters.genome_length) / 365.0
        clock_rate = profile.sample_clock_rate(size=(2, 2))
        expected_mutations = profile.expected_mutations(np.array([0.0, 5.0, 10.0]))

        np.testing.assert_allclose(clock_rate, np.full((2, 2), expected_rate_per_day))
        np.testing.assert_allclose(
            expected_mutations,
            np.array([0.0, 5.0 * expected_rate_per_day, 10.0 * expected_rate_per_day]),
        )


class TestInfectiousnessToTransmission(unittest.TestCase):
    def test_pdf_returns_all_zeros_when_every_time_is_negative(self) -> None:
        profile = InfectiousnessToTransmission(
            rng_seed=123,
            integration_grid_points=64,
            sampling_grid_points=64,
        )

        values = profile.pdf(np.array([-3.0, -1.5, -0.2]))

        np.testing.assert_allclose(values, np.zeros(3))

    def test_pdf_returns_zero_for_negative_times_and_scalar_for_scalar_input(self) -> None:
        profile = InfectiousnessToTransmission(
            rng_seed=123,
            integration_grid_points=64,
            sampling_grid_points=64,
        )

        values = profile.pdf(np.array([-2.0, -0.5, 0.0, 1.0]))
        scalar = profile.pdf(0.0)

        np.testing.assert_allclose(values[:2], np.array([0.0, 0.0]))
        self.assertTrue(np.all(values >= 0.0))
        self.assertIsInstance(scalar, float)

    def test_rvs_respects_shape_and_grid_bounds(self) -> None:
        profile = InfectiousnessToTransmission(
            rng_seed=123,
            integration_grid_points=64,
            sampling_grid_points=64,
        )

        draws = profile.rvs(size=(3, 4))

        self.assertEqual(draws.shape, (3, 4))
        self.assertTrue(np.all(draws >= profile.grid_min_days))
        self.assertTrue(np.all(draws <= profile.grid_max_days))

    def test_sample_generation_intervals_adds_latent_periods_and_transmission_times(self) -> None:
        profile = InfectiousnessToTransmission(rng_seed=123)
        latent_periods = np.array([1.0, 2.0, 3.0])
        transmission_times = np.array([0.5, 0.75, 1.25])

        with (
            patch.object(profile, "sample_latent_periods", return_value=latent_periods),
            patch.object(profile, "rvs", return_value=transmission_times),
        ):
            generation_intervals = profile.sample_generation_intervals(size=3)

        np.testing.assert_allclose(generation_intervals, latent_periods + transmission_times)


class TestSymptomOnsetToTransmission(unittest.TestCase):
    def test_pdf_matches_documented_piecewise_definition(self) -> None:
        profile = SymptomOnsetToTransmission(rng_seed=123, grid_points=128)
        points = np.array([-2.0, -0.5, 0.0, 1.5])
        parameters = profile.parameters

        expected = np.where(
            points < 0.0,
            parameters.transmission_rate_ratio
            * parameters.infectiousness_normalisation
            * (1.0 - profile.presymptomatic.cdf(-points)),
            parameters.infectiousness_normalisation
            * (1.0 - profile.symptomatic.cdf(points)),
        )

        np.testing.assert_allclose(profile.pdf(points), expected)
        self.assertIsInstance(profile.pdf(0.0), float)

    def test_cdf_and_rvs_respect_configured_bounds(self) -> None:
        profile = SymptomOnsetToTransmission(rng_seed=123, grid_points=128)

        draws = profile.rvs(size=20)

        self.assertEqual(profile.cdf(profile.grid_min_days - 1.0), 0.0)
        self.assertEqual(profile.cdf(profile.grid_max_days + 1.0), 1.0)
        self.assertEqual(draws.shape, (20,))
        self.assertTrue(np.all(draws >= profile.grid_min_days))
        self.assertTrue(np.all(draws <= profile.grid_max_days))
