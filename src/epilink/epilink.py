from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator

from .profiles import InfectiousnessToTransmission


@dataclass(frozen=True, slots=True)
class Scenario:
    """Latent transmission scenario."""

    kind: str
    intermediates: int | None = None
    branch_to_i: int | None = None
    branch_to_j: int | None = None

    def label(self) -> str:
        if self.kind == "ad":
            return f"ad({self.intermediates})"
        return f"ca({self.branch_to_i},{self.branch_to_j})"


class EpiLink:
    """
    Pairwise scenario compatibility scoring.

    Parameters
    ----------
    transmission_profile :
        Instance of ``InfectiousnessToTransmission`` providing latent-time
        samples and per-day substitution-rate samples.
    maximum_depth :
        Maximum hidden-depth parameter.
    mc_samples :
        Number of Monte Carlo draws precomputed for each scenario.
    target :
        Scenario label or subset of scenario labels returned as the main score.
        Accepts canonical labels such as ``ad(0)`` and ``ca(0,0)``, or
        ``Scenario`` objects.
        When multiple targets are provided, the reported target score is the
        sum of the compatibilities across the configured subset.
    mutation_process :
        Either ``"deterministic"`` to compare against expected mutation counts
        or ``"stochastic"`` to compare against Poisson mutation-count draws.
    rng :
        Random number generator used for stochastic mutation-count draws. When
        omitted, the transmission profile generator is reused.
    """

    def __init__(
        self,
        transmission_profile: InfectiousnessToTransmission,
        maximum_depth: int = 2,
        mc_samples: int = 10000,
        target: str | Scenario | Iterable[str | Scenario] = "ad(0)",
        mutation_process: str = "stochastic",
        rng: Generator | None = None,
    ) -> None:
        if maximum_depth < 0:
            raise ValueError("maximum_depth must be >= 0.")
        if mc_samples <= 0:
            raise ValueError("mc_samples must be > 0.")

        self.profile = transmission_profile
        self.maximum_depth = int(maximum_depth)
        self.sample_count = int(mc_samples)
        self.mutation_process = self._validate_mutation_process(mutation_process)
        self.random_generator = rng if rng is not None else transmission_profile.rng

        self.scenarios = self._enumerate_scenarios()
        self.scenarios_by_label = {scenario.label(): scenario for scenario in self.scenarios}
        self.target_labels = self._resolve_target_labels(target)
        self.target_label = self.target_labels[0] if len(self.target_labels) == 1 else None
        self.draws_by_scenario = self._precompute_draws()

    @staticmethod
    def _validate_mutation_process(mutation_process: str) -> str:
        normalized_process = mutation_process.strip().lower()
        if normalized_process not in {"deterministic", "stochastic"}:
            raise ValueError(
                "mutation_process must be either 'deterministic' or 'stochastic'."
            )
        return normalized_process

    @staticmethod
    def _normalize_target_label(target_label: str) -> str:
        return target_label.strip().lower()

    def _resolve_target_labels(
        self,
        target: str | Scenario | Iterable[str | Scenario],
    ) -> tuple[str, ...]:
        if isinstance(target, (str, Scenario)):
            raw_targets = (target,)
        else:
            raw_targets = tuple(target)

        if not raw_targets:
            raise ValueError("target must contain at least one scenario.")

        available_labels = ", ".join(self.scenarios_by_label)
        resolved_labels: list[str] = []

        for raw_target in raw_targets:
            target_label = raw_target.label() if isinstance(raw_target, Scenario) else str(raw_target)
            normalized_label = self._normalize_target_label(target_label)

            if normalized_label not in self.scenarios_by_label:
                raise ValueError(
                    f"Unknown target scenario '{target_label}'. Available scenarios: {available_labels}."
                )
            if normalized_label not in resolved_labels:
                resolved_labels.append(normalized_label)

        return tuple(resolved_labels)

    def _enumerate_scenarios(self) -> list[Scenario]:
        scenarios: list[Scenario] = []

        for intermediates in range(self.maximum_depth + 1):
            scenarios.append(Scenario(kind="ad", intermediates=intermediates))

        for branch_to_i in range(self.maximum_depth + 1):
            for branch_to_j in range(self.maximum_depth + 1 - branch_to_i):
                scenarios.append(
                    Scenario(kind="ca", branch_to_i=branch_to_i, branch_to_j=branch_to_j)
                )

        return scenarios

    def _sum_generation_intervals(
        self,
        generations_per_path: int,
        sample_count: int,
    ) -> np.ndarray:
        if generations_per_path <= 0:
            return np.zeros(sample_count, dtype=float)

        generation_draws = self.profile.sample_generation_intervals(
            size=(generations_per_path, sample_count)
        )
        return np.asarray(generation_draws, dtype=float).sum(axis=0)

    def simulate_scenario_draws(
        self,
        scenario: Scenario,
        sample_count: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate sample-time and branch-length draws for one scenario.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(time_draws, branch_draws)`` with one draw per Monte Carlo sample.
        """

        sample_count = self.sample_count if sample_count is None else int(sample_count)

        incubation_i = self.profile.sample_incubation_periods(sample_count)
        incubation_j = self.profile.sample_incubation_periods(sample_count)
        testing_delay_i = self.profile.sample_testing_delays(sample_count)
        testing_delay_j = self.profile.sample_testing_delays(sample_count)

        if scenario.kind == "ad":
            elapsed_generations = self._sum_generation_intervals(
                generations_per_path=int(scenario.intermediates) + 1,
                sample_count=sample_count,
            )
            time_draws = (
                elapsed_generations
                + incubation_j
                + testing_delay_j
                - incubation_i
                - testing_delay_i
            )
            branch_draws = time_draws
        elif scenario.kind == "ca":
            path_to_i = self._sum_generation_intervals(
                generations_per_path=int(scenario.branch_to_i) + 1,
                sample_count=sample_count,
            )
            path_to_j = self._sum_generation_intervals(
                generations_per_path=int(scenario.branch_to_j) + 1,
                sample_count=sample_count,
            )
            time_draws = (
                path_to_j
                - path_to_i
                + incubation_j
                + testing_delay_j
                - incubation_i
                - testing_delay_i
            )
            branch_draws = (
                path_to_i
                + path_to_j
                + incubation_i
                + testing_delay_i
                + incubation_j
                + testing_delay_j
            )
        else:
            raise ValueError(f"Unknown scenario kind: {scenario.kind}")

        return np.asarray(time_draws, dtype=float), np.clip(
            np.asarray(branch_draws, dtype=float),
            a_min=0.0,
            a_max=np.inf,
        )

    def _simulate_genetic_draws(self, branch_draws: np.ndarray) -> np.ndarray:
        clock_rate_draws = np.asarray(
            self.profile.sample_clock_rate(size=branch_draws.shape),
            dtype=float,
        )
        expected_mutation_draws = np.clip(branch_draws * clock_rate_draws, a_min=0.0, a_max=np.inf)

        if self.mutation_process == "deterministic":
            return expected_mutation_draws

        return self.random_generator.poisson(expected_mutation_draws)

    def _precompute_draws(self) -> dict[str, dict[str, np.ndarray]]:
        draws_by_scenario: dict[str, dict[str, np.ndarray]] = {}

        for scenario in self.scenarios:
            label = scenario.label()
            time_draws, branch_draws = self.simulate_scenario_draws(scenario)
            genetic_draws = self._simulate_genetic_draws(branch_draws)

            draws_by_scenario[label] = {
                "time_draws": time_draws,
                "branch_draws": branch_draws,
                "genetic_draws": np.asarray(genetic_draws),
            }

        return draws_by_scenario

    @staticmethod
    def percentile_score(observed_value: float, simulated_values: np.ndarray) -> float:
        simulated_values = np.asarray(simulated_values, dtype=float)
        return float(np.mean(simulated_values <= observed_value))

    @staticmethod
    def compatibility_from_percentile(percentile: float) -> float:
        return float(1.0 - 2.0 * abs(percentile - 0.5))

    @classmethod
    def compatibility_score(
        cls,
        observed_value: float,
        simulated_values: np.ndarray,
    ) -> tuple[float, float]:
        percentile = cls.percentile_score(observed_value, simulated_values)
        compatibility = cls.compatibility_from_percentile(percentile)
        return percentile, compatibility

    def score_pair(
        self,
        sample_time_difference: float | None = None,
        genetic_distance: float | None = None,
    ) -> dict[str, Any]:
        """
        Compute compatibility scores for all scenarios for one sampled pair.

        Parameters
        ----------
        sample_time_difference :
            Observed difference in sampling times.
        genetic_distance :
            Observed consensus-level genetic distance.

        Returns
        -------
        dict
            Per-scenario percentile and compatibility summaries together with
            the summed compatibility of the configured target subset.
        """

        if sample_time_difference is None or genetic_distance is None:
            raise TypeError(
                "score_pair() requires both sample_time_difference and genetic_distance."
            )

        scenario_scores = {}

        for label, draws in self.draws_by_scenario.items():
            time_percentile, time_compatibility = self.compatibility_score(
                observed_value=sample_time_difference,
                simulated_values=draws["time_draws"],
            )
            genetic_percentile, genetic_compatibility = self.compatibility_score(
                observed_value=genetic_distance,
                simulated_values=draws["genetic_draws"],
            )
            scenario_scores[label] = {
                "time_percentile": time_percentile,
                "time_compatibility": time_compatibility,
                "genetic_percentile": genetic_percentile,
                "genetic_compatibility": genetic_compatibility,
                "compatibility": time_compatibility * genetic_compatibility,
            }

        target_compatibility = float(
            sum(scenario_scores[label]["compatibility"] for label in self.target_labels)
        )

        return {
            "target": self.target_labels[0] if self.target_label is not None else list(self.target_labels),
            "target_labels": list(self.target_labels),
            "target_compatibility": target_compatibility,
            "scenario_scores": scenario_scores,
        }

    def score_target(
        self,
        sample_time_difference: float,
        genetic_distance: float,
    ) -> float:
        """Return only the summed compatibility of the configured target subset."""

        result = self.score_pair(
            sample_time_difference=sample_time_difference,
            genetic_distance=genetic_distance,
        )
        return float(result["target_compatibility"])


__all__ = ["EpiLink", "Scenario"]
