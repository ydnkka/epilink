from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike

from .exceptions import ConfigurationError, ScenarioError, SimulationError
from .profiles import InfectiousnessToTransmission
from .results import PairCompatibilityResult, ScenarioScore

logger = logging.getLogger(__name__)

_AD_SCENARIO_PATTERN = re.compile(r"ad\((\d+)\)")
_CA_SCENARIO_PATTERN = re.compile(r"ca\((\d+),(\d+)\)")


@dataclass(frozen=True, slots=True)
class Scenario:
    """Latent transmission scenario."""

    kind: str
    intermediates: int | None = None
    branch_to_i: int | None = None
    branch_to_j: int | None = None

    def __post_init__(self) -> None:
        normalized_kind = self.kind.strip().lower()
        object.__setattr__(self, "kind", normalized_kind)

        if normalized_kind == "ad":
            if self.intermediates is None:
                raise ScenarioError("Ancestor-descendant scenarios require intermediates.")
            if self.intermediates < 0:
                raise ScenarioError("Scenario depths must be non-negative.")
            if self.branch_to_i is not None or self.branch_to_j is not None:
                raise ScenarioError("Ancestor-descendant scenarios do not accept branch depths.")
            return

        if normalized_kind == "ca":
            if self.branch_to_i is None or self.branch_to_j is None:
                raise ScenarioError("Common-ancestor scenarios require both branch depths.")
            if self.branch_to_i < 0 or self.branch_to_j < 0:
                raise ScenarioError("Scenario depths must be non-negative.")
            if self.intermediates is not None:
                raise ScenarioError("Common-ancestor scenarios do not accept intermediates.")
            return

        raise ScenarioError(f"Invalid scenario kind: '{self.kind}'. Must be 'ad' or 'ca'.")

    @classmethod
    def ancestor_descendant(cls, intermediates: int) -> Scenario:
        return cls(kind="ad", intermediates=intermediates)

    @classmethod
    def common_ancestor(cls, branch_to_i: int, branch_to_j: int) -> Scenario:
        return cls(kind="ca", branch_to_i=branch_to_i, branch_to_j=branch_to_j)

    @classmethod
    def parse(cls, value: str | Scenario) -> Scenario:
        if isinstance(value, cls):
            return value

        normalized = str(value).strip().lower()
        ad_match = _AD_SCENARIO_PATTERN.fullmatch(normalized)
        if ad_match is not None:
            return cls.ancestor_descendant(intermediates=int(ad_match.group(1)))

        ca_match = _CA_SCENARIO_PATTERN.fullmatch(normalized)
        if ca_match is not None:
            return cls.common_ancestor(
                branch_to_i=int(ca_match.group(1)),
                branch_to_j=int(ca_match.group(2)),
            )

        raise ScenarioError(f"Invalid scenario label '{value}'.")

    def label(self) -> str:
        if self.kind == "ad":
            return f"ad({self.intermediates})"
        return f"ca({self.branch_to_i},{self.branch_to_j})"

    def __str__(self) -> str:
        return self.label()


class PairwiseCompatibilityModel:
    """
    Vectorized pairwise compatibility scorer for a fixed target subset.

    Parameters
    ----------
    draws_by_scenario :
        Cached Monte Carlo draws keyed by scenario label.
    target_labels :
        Canonical scenario labels included in the target subset.
    """

    def __init__(
        self,
        draws_by_scenario: dict[str, dict[str, np.ndarray]],
        target_labels: Iterable[str],
    ) -> None:
        self.target_labels = tuple(target_labels)
        if not self.target_labels:
            raise ConfigurationError("target_labels must contain at least one scenario.")

        self._sorted_time_draws = tuple(
            np.sort(np.asarray(draws_by_scenario[label]["time_draws"], dtype=float))
            for label in self.target_labels
        )
        self._sorted_genetic_draws = tuple(
            np.sort(np.asarray(draws_by_scenario[label]["genetic_draws"], dtype=float))
            for label in self.target_labels
        )

    @staticmethod
    def _broadcast_observations(
        sample_time_difference: ArrayLike,
        genetic_distance: ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
        sample_time_values, genetic_values = np.broadcast_arrays(
            np.asarray(sample_time_difference, dtype=float),
            np.asarray(genetic_distance, dtype=float),
        )
        return (
            sample_time_values.reshape(-1),
            genetic_values.reshape(-1),
            sample_time_values.shape,
        )

    @staticmethod
    def _percentile_scores(
        observed_values: np.ndarray,
        simulated_sorted_values: np.ndarray,
    ) -> np.ndarray:
        return (
            np.searchsorted(simulated_sorted_values, observed_values, side="right").astype(float)
            / simulated_sorted_values.size
        )

    @staticmethod
    def _compatibility_from_percentiles(percentiles: np.ndarray) -> np.ndarray:
        return 1.0 - 2.0 * np.abs(percentiles - 0.5)

    def score(
        self,
        sample_time_difference: ArrayLike,
        genetic_distance: ArrayLike,
    ) -> np.ndarray:
        """
        Compute target-subset compatibility for arraylike observations.

        Returns
        -------
        numpy.ndarray
            Compatibility scores with the broadcast shape of the inputs.
        """

        sample_time_values, genetic_values, output_shape = self._broadcast_observations(
            sample_time_difference=sample_time_difference,
            genetic_distance=genetic_distance,
        )
        target_scores = np.zeros(sample_time_values.shape, dtype=float)

        for time_draws, genetic_draws in zip(
            self._sorted_time_draws,
            self._sorted_genetic_draws,
            strict=True,
        ):
            time_percentiles = self._percentile_scores(sample_time_values, time_draws)
            genetic_percentiles = self._percentile_scores(genetic_values, genetic_draws)
            target_scores += self._compatibility_from_percentiles(
                time_percentiles
            ) * self._compatibility_from_percentiles(genetic_percentiles)

        return target_scores.reshape(output_shape)

    __call__ = score


class EpiLink:
    """
    Pairwise scenario compatibility scoring.

    Parameters
    ----------
    transmission_profile :
        Instance of ``InfectiousnessToTransmission`` providing generation-interval,
        testing-delay, and per-day substitution-rate samples.
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

    Notes
    -----
    Use :meth:`pairwise_model` to obtain a dedicated cached scorer for a
    target subset. The resulting model accepts scalar or arraylike inputs and
    returns compatibility scores with the broadcast shape of the observations.
    """

    def __init__(
        self,
        transmission_profile: InfectiousnessToTransmission,
        maximum_depth: int = 2,
        mc_samples: int = 10000,
        target: str | Scenario | Iterable[str | Scenario] = ("ad(0)", "ca(0,0)"),
        mutation_process: str = "stochastic",
        rng: Generator | None = None,
    ) -> None:
        if maximum_depth < 0:
            raise ConfigurationError("maximum_depth must be >= 0.")
        if mc_samples <= 0:
            raise ConfigurationError("mc_samples must be > 0.")

        self.profile = transmission_profile
        self.maximum_depth = int(maximum_depth)
        self.sample_count = int(mc_samples)
        self.mutation_process = self._validate_mutation_process(mutation_process)
        self.random_generator = rng if rng is not None else transmission_profile.rng

        self.scenarios = self._enumerate_scenarios()
        self.scenarios_by_label = {scenario.label(): scenario for scenario in self.scenarios}
        self.target_labels = self._resolve_target_labels(target)
        self.target_label = self.target_labels[0] if len(self.target_labels) == 1 else None
        self._pairwise_models_by_target: dict[tuple[str, ...], PairwiseCompatibilityModel] = {}
        self.draws_by_scenario = self._precompute_draws()

    @property
    def draws_by_scenario(self) -> dict[str, dict[str, np.ndarray]]:
        return self._draws_by_scenario

    @draws_by_scenario.setter
    def draws_by_scenario(self, draws_by_scenario: dict[str, dict[str, np.ndarray]]) -> None:
        self._draws_by_scenario = draws_by_scenario
        self._pairwise_models_by_target.clear()

    @staticmethod
    def _validate_mutation_process(mutation_process: str) -> str:
        normalized_process = mutation_process.strip().lower()
        if normalized_process not in {"deterministic", "stochastic"}:
            raise ConfigurationError(
                "mutation_process must be either 'deterministic' or 'stochastic'."
            )
        return normalized_process

    def _resolve_target_labels(
        self,
        target: str | Scenario | Iterable[str | Scenario],
    ) -> tuple[str, ...]:
        raw_targets: tuple[str | Scenario, ...]
        if isinstance(target, (str, Scenario)):
            raw_targets = (target,)
        else:
            raw_targets = tuple(target)

        if not raw_targets:
            raise ConfigurationError("target must contain at least one scenario.")

        available_labels = ", ".join(self.scenarios_by_label)
        resolved_labels: set[str] = set()

        for raw_target in raw_targets:
            try:
                normalized_label = Scenario.parse(raw_target).label()
            except ScenarioError as error:
                raise ScenarioError(
                    f"Unknown target scenario '{raw_target}'. Available scenarios: {available_labels}."
                ) from error

            if normalized_label not in self.scenarios_by_label:
                raise ScenarioError(
                    f"Unknown target scenario '{raw_target}'. Available scenarios: {available_labels}."
                )
            resolved_labels.add(normalized_label)

        return tuple(label for label in self.scenarios_by_label if label in resolved_labels)

    def _enumerate_scenarios(self) -> list[Scenario]:
        scenarios: list[Scenario] = []

        for intermediates in range(self.maximum_depth + 1):
            scenarios.append(Scenario.ancestor_descendant(intermediates=intermediates))

        for branch_to_i in range(self.maximum_depth + 1):
            for branch_to_j in range(self.maximum_depth + 1 - branch_to_i):
                scenarios.append(Scenario.common_ancestor(branch_to_i, branch_to_j))

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
            if scenario.intermediates is None:
                raise ScenarioError("Ancestor-descendant scenarios require intermediates.")
            elapsed_generations = self._sum_generation_intervals(
                generations_per_path=scenario.intermediates + 1,
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
            if scenario.branch_to_i is None or scenario.branch_to_j is None:
                raise ScenarioError("Common-ancestor scenarios require both branch depths.")
            path_to_i = self._sum_generation_intervals(
                generations_per_path=scenario.branch_to_i + 1,
                sample_count=sample_count,
            )
            path_to_j = self._sum_generation_intervals(
                generations_per_path=scenario.branch_to_j + 1,
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
            raise ScenarioError(f"Unknown scenario kind: {scenario.kind}")

        return np.asarray(time_draws, dtype=float), np.clip(
            np.asarray(branch_draws, dtype=float),
            a_min=0.0,
            a_max=np.inf,
        )

    def _simulate_genetic_draws(self, branch_draws: np.ndarray) -> np.ndarray:
        expected_mutation_draws = self.profile.expected_mutations(branch_draws)

        if self.mutation_process == "deterministic":
            return expected_mutation_draws

        return self.random_generator.poisson(expected_mutation_draws)

    def _precompute_draws(self) -> dict[str, dict[str, np.ndarray]]:
        logger.info(
            "Precomputing %d draws for %d scenarios.",
            self.sample_count,
            len(self.scenarios),
        )
        draws_by_scenario: dict[str, dict[str, np.ndarray]] = {}

        for scenario in self.scenarios:
            label = scenario.label()
            try:
                time_draws, branch_draws = self.simulate_scenario_draws(scenario)
                genetic_draws = self._simulate_genetic_draws(branch_draws)
            except Exception as error:
                raise SimulationError(
                    f"Failed to precompute draws for scenario '{label}'."
                ) from error

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

    @staticmethod
    def _validate_scalar_observation(observed_value: ArrayLike, name: str) -> float:
        scalar_value = np.asarray(observed_value, dtype=float)
        if scalar_value.ndim != 0:
            raise TypeError(
                f"score_pair() requires scalar {name}; use pairwise_model() or score_target() "
                "for arraylike inputs."
            )
        return float(scalar_value)

    def pairwise_model(
        self,
        target: str | Scenario | Iterable[str | Scenario] | None = None,
    ) -> PairwiseCompatibilityModel:
        """
        Return a cached vectorized scorer for a target subset.

        Parameters
        ----------
        target :
            Optional target subset. When omitted, the scorer uses the subset
            configured at construction time.
        """

        target_labels = (
            self.target_labels if target is None else self._resolve_target_labels(target)
        )
        pairwise_model = self._pairwise_models_by_target.get(target_labels)
        if pairwise_model is None:
            logger.debug("Creating new PairwiseCompatibilityModel for target: %s", target_labels)
            pairwise_model = PairwiseCompatibilityModel(self.draws_by_scenario, target_labels)
            self._pairwise_models_by_target[target_labels] = pairwise_model
        return pairwise_model

    def score_pair(
        self,
        sample_time_difference: float | None = None,
        genetic_distance: float | None = None,
    ) -> PairCompatibilityResult:
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
        sample_time_difference = self._validate_scalar_observation(
            sample_time_difference,
            "sample_time_difference",
        )
        genetic_distance = self._validate_scalar_observation(
            genetic_distance,
            "genetic_distance",
        )

        scenario_scores: dict[str, ScenarioScore] = {}

        for label, draws in self.draws_by_scenario.items():
            time_percentile, time_compatibility = self.compatibility_score(
                observed_value=sample_time_difference,
                simulated_values=draws["time_draws"],
            )
            genetic_percentile, genetic_compatibility = self.compatibility_score(
                observed_value=genetic_distance,
                simulated_values=draws["genetic_draws"],
            )
            scenario_scores[label] = ScenarioScore(
                time_percentile=time_percentile,
                time_compatibility=time_compatibility,
                genetic_percentile=genetic_percentile,
                genetic_compatibility=genetic_compatibility,
                compatibility=time_compatibility * genetic_compatibility,
            )

        target_compatibility = float(
            self.pairwise_model()(sample_time_difference, genetic_distance)
        )

        return PairCompatibilityResult(
            target=self.target_labels[0] if self.target_label is not None else self.target_labels,
            target_labels=self.target_labels,
            target_compatibility=target_compatibility,
            scenario_scores=scenario_scores,
        )

    def score_target(
        self,
        sample_time_difference: ArrayLike,
        genetic_distance: ArrayLike,
        target: str | Scenario | Iterable[str | Scenario] | None = None,
    ) -> float | np.ndarray:
        """
        Return the summed compatibility of a target subset.

        Scalar inputs return a scalar ``float``. Arraylike inputs return a
        ``numpy.ndarray`` with the broadcast shape of the observations.
        """

        result = self.pairwise_model(target).score(
            sample_time_difference=sample_time_difference,
            genetic_distance=genetic_distance,
        )
        return float(result) if result.ndim == 0 else result


__all__ = [
    "EpiLink",
    "PairCompatibilityResult",
    "PairwiseCompatibilityModel",
    "Scenario",
    "ScenarioScore",
]
