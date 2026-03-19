from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from epilink import NaturalHistoryParameters  # noqa: E402


class TestNaturalHistoryParameters(unittest.TestCase):
    def test_custom_parameters_update_derived_properties(self) -> None:
        p = NaturalHistoryParameters(
            incubation_shape=6.0,
            incubation_scale=0.75,
            latent_shape=2.5,
            symptomatic_rate=0.5,
            symptomatic_shape=2.0,
            transmission_rate_ratio=1.8,
        )

        self.assertTrue(math.isclose(p.presymptomatic_shape, 3.5))
        self.assertTrue(math.isclose(p.symptomatic_scale, 1.0))
        self.assertTrue(math.isclose(p.incubation_rate, 1.0 / 4.5))

    def test_to_dict_round_trips_through_constructor(self) -> None:
        original = NaturalHistoryParameters(
            incubation_shape=6.2,
            incubation_scale=0.8,
            latent_shape=2.0,
            symptomatic_rate=0.4,
            symptomatic_shape=1.3,
            transmission_rate_ratio=1.9,
            testing_delay_shape=2.5,
            testing_delay_scale=1.2,
            substitution_rate=1.2e-3,
            relaxation=0.15,
            genome_length=12345,
        )

        reconstructed = NaturalHistoryParameters(**original.to_dict())

        self.assertEqual(reconstructed, original)

    def test_default_derived_properties(self) -> None:
        p = NaturalHistoryParameters()

        self.assertTrue(math.isclose(p.presymptomatic_shape, p.incubation_shape - p.latent_shape))
        self.assertTrue(
            math.isclose(
                p.symptomatic_scale,
                1.0 / (p.symptomatic_shape * p.symptomatic_rate),
            )
        )
        self.assertTrue(
            math.isclose(
                p.incubation_rate,
                1.0 / (p.incubation_shape * p.incubation_scale),
            )
        )

    def test_transmission_metrics_match_documented_equations(self) -> None:
        p = NaturalHistoryParameters()

        expected_fraction = (
            p.transmission_rate_ratio * p.presymptomatic_shape * p.symptomatic_rate
        ) / (
            p.transmission_rate_ratio * p.presymptomatic_shape * p.symptomatic_rate
            + p.incubation_shape * p.incubation_rate
        )
        expected_normalisation = (p.incubation_shape * p.incubation_rate * p.symptomatic_rate) / (
            p.transmission_rate_ratio * p.presymptomatic_shape * p.symptomatic_rate
            + p.incubation_shape * p.incubation_rate
        )

        self.assertTrue(math.isclose(p.presymptomatic_transmission_fraction, expected_fraction))
        self.assertTrue(math.isclose(p.infectiousness_normalisation, expected_normalisation))
        self.assertGreaterEqual(p.presymptomatic_transmission_fraction, 0.0)
        self.assertLessEqual(p.presymptomatic_transmission_fraction, 1.0)

    def test_to_dict_returns_only_constructor_fields(self) -> None:
        p = NaturalHistoryParameters()
        data = p.to_dict()

        self.assertEqual(
            set(data.keys()),
            {
                "incubation_shape",
                "incubation_scale",
                "latent_shape",
                "symptomatic_rate",
                "symptomatic_shape",
                "transmission_rate_ratio",
                "testing_delay_shape",
                "testing_delay_scale",
                "substitution_rate",
                "relaxation",
                "genome_length",
            },
        )
        self.assertNotIn("presymptomatic_shape", data)
        self.assertEqual(data["testing_delay_shape"], 2.0)

    def test_bool_and_non_real_values_raise_type_error(self) -> None:
        with self.assertRaises(TypeError):
            NaturalHistoryParameters(symptomatic_rate=True)

        with self.assertRaises(TypeError):
            NaturalHistoryParameters(incubation_shape="5.807")

        with self.assertRaises(TypeError):
            NaturalHistoryParameters(relaxation=True)

    def test_non_positive_values_raise_value_error(self) -> None:
        non_positive_cases = (
            {"incubation_shape": 0.0},
            {"incubation_scale": -1.0},
            {"latent_shape": 0.0},
            {"symptomatic_rate": -0.1},
            {"symptomatic_shape": 0.0},
            {"transmission_rate_ratio": 0.0},
            {"testing_delay_shape": -2.0},
            {"testing_delay_scale": 0.0},
            {"substitution_rate": 0.0},
            {"genome_length": -1},
        )

        for kwargs in non_positive_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    NaturalHistoryParameters(**kwargs)

    def test_relaxation_must_be_non_negative(self) -> None:
        NaturalHistoryParameters(relaxation=0.0)

        with self.assertRaises(ValueError):
            NaturalHistoryParameters(relaxation=-0.1)

    def test_latent_shape_must_be_less_than_incubation_shape(self) -> None:
        with self.assertRaises(ValueError):
            NaturalHistoryParameters(latent_shape=5.807)


if __name__ == "__main__":
    unittest.main()
