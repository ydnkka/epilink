"""Transmission profile distributions used by epilink."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.random import Generator, default_rng
from scipy import stats

from .parameters import NaturalHistoryParameters

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


class BaseTransmissionProfile:
    """Base class for transmission profile distributions.

    Parameters
    ----------
    grid_min_days : float
        Lower bound of the numerical grid used for sampling and numerical
        summaries.
    grid_max_days : float
        Upper bound of the numerical grid used for sampling and numerical
        summaries.
    parameters : NaturalHistoryParameters, optional
        Stage-duration and infectiousness parameters for the E/P/I model.
    rng : Generator, optional
        Random number generator to use for sampling.
    rng_seed : int, optional
        Seed used to initialize ``rng`` when no generator is supplied.
    grid_points : int, default=1024
        Number of grid points used for numerical summaries and sampling.
    """

    def __init__(
        self,
        grid_min_days: float,
        grid_max_days: float,
        parameters: NaturalHistoryParameters | None = None,
        rng: Generator | None = None,
        rng_seed: int | None = None,
        grid_points: int = 1024,
    ) -> None:
        if grid_max_days < grid_min_days:
            raise ValueError("grid_max_days must be greater than or equal to grid_min_days.")
        if grid_points < 2:
            raise ValueError("grid_points must be at least 2.")

        self.grid_min_days = float(grid_min_days)
        self.grid_max_days = float(grid_max_days)
        self.parameters = parameters or NaturalHistoryParameters()
        self.rng: Generator = rng if rng is not None else default_rng(rng_seed)
        self.grid_points = int(grid_points)

        self._sampling_grid: np.ndarray | None = None
        self._sampling_weights: np.ndarray | None = None

        parameters = self.parameters
        self.incubation = stats.gamma(
            a=parameters.incubation_shape, scale=parameters.incubation_scale
        )
        self.latent = stats.gamma(a=parameters.latent_shape, scale=parameters.incubation_scale)
        self.presymptomatic = stats.gamma(
            a=parameters.presymptomatic_shape, scale=parameters.incubation_scale
        )
        self.symptomatic = stats.gamma(
            a=parameters.symptomatic_shape, scale=parameters.symptomatic_scale
        )

    def pdf(self, times_in_days: npt.ArrayLike) -> np.ndarray:
        """Evaluate the probability density function.

        Parameters
        ----------
        times_in_days : array_like
            Evaluation points in days.

        Returns
        -------
        numpy.ndarray
            Probability density values at ``times_in_days``.
        """

        raise NotImplementedError

    def cdf(self, times_in_days: npt.ArrayLike) -> np.ndarray:
        """Evaluate the cumulative distribution function on the numerical grid.

        Parameters
        ----------
        times_in_days : array_like
            Evaluation points in days.

        Returns
        -------
        numpy.ndarray
            Approximate cumulative probabilities clipped to ``[0, 1]``.
        """

        if _trapz is None:
            raise ImportError("Neither np.trapezoid nor np.trapz found in NumPy.")

        evaluation_points = np.asarray(times_in_days, dtype=float)
        flat_evaluation_points = evaluation_points.reshape(-1)
        cumulative_probability = np.zeros_like(flat_evaluation_points)

        for index, time_value in enumerate(flat_evaluation_points):
            if time_value <= self.grid_min_days:
                cumulative_probability[index] = 0.0
            elif time_value >= self.grid_max_days:
                cumulative_probability[index] = 1.0
            else:
                integration_grid = np.linspace(
                    self.grid_min_days, time_value, num=max(100, self.grid_points)
                )
                probability_density = self.pdf(integration_grid)
                cumulative_probability[index] = _trapz(probability_density, integration_grid)

        cumulative_probability = np.clip(cumulative_probability, 0.0, 1.0).reshape(
            evaluation_points.shape
        )
        if evaluation_points.ndim == 0:
            return cumulative_probability[()]
        return cumulative_probability

    def mean(self) -> float:
        """Estimate the mean on the configured numerical grid.

        Returns
        -------
        float
            Numerical estimate of the mean.
        """

        if _trapz is None:
            raise ImportError("Neither np.trapezoid nor np.trapz found in NumPy.")

        integration_grid = np.linspace(
            self.grid_min_days, self.grid_max_days, num=max(100, self.grid_points)
        )
        probability_density = np.clip(self.pdf(integration_grid), a_min=0.0, a_max=np.inf)
        total_mass = float(_trapz(probability_density, integration_grid))
        if not np.isfinite(total_mass) or total_mass <= 0.0:
            return float("nan")
        return float(_trapz(integration_grid * probability_density, integration_grid) / total_mass)

    def rvs(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        """Sample from the profile on the configured numerical grid.

        Parameters
        ----------
        size : int or tuple of int, default=1
            Output shape.

        Returns
        -------
        numpy.ndarray
            Random variates.
        """

        raise NotImplementedError

    def sample_latent_periods(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        """Sample latent-period durations.

        Parameters
        ----------
        size : int or tuple of int, default=1
            Output shape.

        Returns
        -------
        numpy.ndarray
            Latent-period durations in days.
        """

        return self.rng.gamma(
            shape=self.parameters.latent_shape,
            scale=self.parameters.incubation_scale,
            size=size,
        )

    def sample_presymptomatic_periods(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        """Sample presymptomatic-period durations.

        Parameters
        ----------
        size : int or tuple of int, default=1
            Output shape.

        Returns
        -------
        numpy.ndarray
            Presymptomatic-period durations in days.
        """

        return self.rng.gamma(
            shape=self.parameters.presymptomatic_shape,
            scale=self.parameters.incubation_scale,
            size=size,
        )

    def sample_incubation_periods(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        """Sample incubation periods.

        Parameters
        ----------
        size : int or tuple of int, default=1
            Output shape.

        Returns
        -------
        numpy.ndarray
            Incubation-period durations in days.
        """

        return self.sample_latent_periods(size) + self.sample_presymptomatic_periods(size)

    def sample_symptomatic_periods(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        """Sample symptomatic-period durations.

        Parameters
        ----------
        size : int or tuple of int, default=1
            Output shape.

        Returns
        -------
        numpy.ndarray
            Symptomatic-period durations in days.
        """

        return self.rng.gamma(
            shape=self.parameters.symptomatic_shape,
            scale=self.parameters.symptomatic_scale,
            size=size,
        )

    def _ensure_sampling_grid(self) -> tuple[np.ndarray, np.ndarray]:
        if self._sampling_grid is None or self._sampling_weights is None:
            sampling_grid = np.linspace(
                self.grid_min_days, self.grid_max_days, num=max(2, self.grid_points)
            )
            probability_density = np.clip(self.pdf(sampling_grid), a_min=0.0, a_max=np.inf)
            weight_sum = float(probability_density.sum())
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                sampling_weights = np.full_like(
                    sampling_grid, fill_value=1.0 / sampling_grid.size, dtype=float
                )
            else:
                sampling_weights = probability_density / weight_sum
            self._sampling_grid = sampling_grid
            self._sampling_weights = sampling_weights

        return self._sampling_grid, self._sampling_weights

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"grid_min_days={self.grid_min_days}, "
            f"grid_max_days={self.grid_max_days}, "
            f"parameters={self.parameters})"
        )


class InfectiousnessToTransmissionTime(BaseTransmissionProfile):
    """Transmission-time distribution from infectiousness onset.

    Parameters
    ----------
    grid_min_days : float, default=0.0
        Lower bound of the numerical sampling grid in days.
    grid_max_days : float, default=60.0
        Upper bound of the numerical sampling grid in days.
    parameters : NaturalHistoryParameters, optional
        Stage-duration and infectiousness parameters for the E/P/I model.
    rng : Generator, optional
        Random number generator to use for sampling.
    rng_seed : int, optional
        Seed used to initialize ``rng`` when no generator is supplied.
    integration_grid_points : int, default=2048
        Number of integration points used inside the PDF evaluation.
    sampling_grid_points : int, default=1024
        Number of numerical grid points used for sampling from the profile.
    """

    def __init__(
        self,
        grid_min_days: float = 0.0,
        grid_max_days: float = 60.0,
        parameters: NaturalHistoryParameters | None = None,
        rng: Generator | None = None,
        rng_seed: int | None = None,
        integration_grid_points: int = 2048,
        sampling_grid_points: int = 1024,
    ) -> None:
        super().__init__(
            grid_min_days=grid_min_days,
            grid_max_days=grid_max_days,
            parameters=parameters,
            rng=rng,
            rng_seed=rng_seed,
            grid_points=sampling_grid_points,
        )
        self.integration_grid_points = int(integration_grid_points)

    def sample_generation_intervals(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        """Sample generation intervals.

        Parameters
        ----------
        size : int or tuple of int, default=1
            Output shape.

        Returns
        -------
        numpy.ndarray
            Generation intervals in days.
        """

        return self.sample_latent_periods(size=size) + self.rvs(size=size)

    def pdf(self, times_in_days: npt.ArrayLike) -> np.ndarray:
        """Evaluate the infectiousness-to-transmission density.

        Parameters
        ----------
        times_in_days : array_like
            Evaluation points in days.

        Returns
        -------
        numpy.ndarray
            Probability density values at ``times_in_days``.
        """

        if _trapz is None:
            raise ImportError("Neither np.trapezoid nor np.trapz found in NumPy.")

        evaluation_points = np.asarray(times_in_days, dtype=float)
        flat_evaluation_points = evaluation_points.reshape(-1)
        probability_density = np.zeros_like(flat_evaluation_points)

        valid_indices = np.flatnonzero(flat_evaluation_points >= 0.0)
        if valid_indices.size == 0:
            if evaluation_points.ndim == 0:
                return probability_density.reshape(evaluation_points.shape)[()]
            return probability_density.reshape(evaluation_points.shape)

        parameters = self.parameters
        for index in valid_indices:
            time_value = float(flat_evaluation_points[index])
            if time_value == 0.0:
                symptomatic_contribution = 0.0
            else:
                presymptomatic_grid = np.linspace(
                    0.0,
                    time_value,
                    num=max(2, self.integration_grid_points),
                )
                presymptomatic_density = self.presymptomatic.pdf(presymptomatic_grid)
                symptomatic_survival = 1.0 - self.symptomatic.cdf(time_value - presymptomatic_grid)
                symptomatic_contribution = float(
                    _trapz(symptomatic_survival * presymptomatic_density, presymptomatic_grid)
                )

            probability_density[index] = parameters.infectiousness_normalisation * (
                parameters.rel_presymptomatic_infectiousness
                * (1.0 - self.presymptomatic.cdf(time_value))
                + symptomatic_contribution
            )

        probability_density = probability_density.reshape(evaluation_points.shape)
        if evaluation_points.ndim == 0:
            return probability_density[()]
        return probability_density

    def rvs(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        """Sample transmission times from the configured numerical grid.

        Parameters
        ----------
        size : int or tuple of int, default=1
            Output shape.

        Returns
        -------
        numpy.ndarray
            Random variates from the profile.
        """

        sample_shape = (size,) if isinstance(size, int) else size
        sampling_grid, sampling_weights = self._ensure_sampling_grid()
        return self.rng.choice(sampling_grid, size=sample_shape, p=sampling_weights)


class SymptomOnsetToTransmissionTime(BaseTransmissionProfile):
    """Transmission-time distribution relative to symptom onset.

    Parameters
    ----------
    grid_min_days : float, default=-30.0
        Lower bound of the numerical sampling grid in days.
    grid_max_days : float, default=30.0
        Upper bound of the numerical sampling grid in days.
    parameters : NaturalHistoryParameters, optional
        Stage-duration and infectiousness parameters for the E/P/I model.
    rng : Generator, optional
        Random number generator to use for sampling.
    rng_seed : int, optional
        Seed used to initialize ``rng`` when no generator is supplied.
    grid_points : int, default=2048
        Number of numerical grid points used for summaries and sampling.
    """

    def __init__(
        self,
        grid_min_days: float = -30.0,
        grid_max_days: float = 30.0,
        parameters: NaturalHistoryParameters | None = None,
        rng: Generator | None = None,
        rng_seed: int | None = None,
        grid_points: int = 2048,
    ) -> None:
        super().__init__(
            grid_min_days=grid_min_days,
            grid_max_days=grid_max_days,
            parameters=parameters,
            rng=rng,
            rng_seed=rng_seed,
            grid_points=grid_points,
        )

    def pdf(self, times_in_days: npt.ArrayLike) -> np.ndarray:
        """Evaluate the symptom-onset-to-transmission density.

        Parameters
        ----------
        times_in_days : array_like
            Evaluation points in days.

        Returns
        -------
        numpy.ndarray
            Probability density values at ``times_in_days``.
        """

        evaluation_points = np.asarray(times_in_days, dtype=float)
        parameters = self.parameters
        probability_density = np.where(
            evaluation_points < 0.0,
            parameters.rel_presymptomatic_infectiousness
            * parameters.infectiousness_normalisation
            * (1.0 - self.presymptomatic.cdf(-evaluation_points)),
            parameters.infectiousness_normalisation
            * (1.0 - self.symptomatic.cdf(evaluation_points)),
        )
        probability_density = np.clip(probability_density, a_min=0.0, a_max=np.inf)
        if evaluation_points.ndim == 0:
            return probability_density[()]
        return probability_density

    def rvs(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        """Sample onset-to-transmission times from the configured grid.

        Parameters
        ----------
        size : int or tuple of int, default=1
            Output shape.

        Returns
        -------
        numpy.ndarray
            Random variates from the profile.
        """

        sample_shape = (size,) if isinstance(size, int) else size
        sampling_grid, sampling_weights = self._ensure_sampling_grid()
        return self.rng.choice(sampling_grid, size=sample_shape, p=sampling_weights)


__all__ = [
    "BaseTransmissionProfile",
    "InfectiousnessToTransmissionTime",
    "SymptomOnsetToTransmissionTime",
    "_trapz",
]
