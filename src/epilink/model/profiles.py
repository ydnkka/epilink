from __future__ import annotations

import numpy as np
from numpy.random import Generator, default_rng
from scipy import stats
from scipy.integrate import cumulative_trapezoid

from .parameters import NaturalHistoryParameters


class BaseTransmissionProfile:
    """Base class for transmission profile distributions.

    Parameters
    ----------
    parameters : NaturalHistoryParameters, optional
        Stage-duration and infectiousness parameters for the E/P/I model.
    grid_min_days : float, default=0.0
        Lower bound of the numerical sampling grid in days.
    grid_max_days : float, default=60.0
    rng : Generator, optional
        Random number generator to use for sampling.
    rng_seed : int, optional
        Seed used to initialize ``rng`` when no generator is supplied.
    grid_points : int, default=1024
        grid points used for numerical summaries and sampling.
    """

    def __init__(
        self,
        parameters: NaturalHistoryParameters | None = None,
        grid_min_days: float = 0.0,
        grid_max_days: float = 100.0,
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
        self._cdf_grid: np.ndarray | None = None

        # Frozen gamma distributions.
        param = self.parameters
        self.incubation = stats.gamma(a=param.incubation_shape, scale=param.incubation_scale)
        self.latent = stats.gamma(a=param.latent_shape, scale=param.incubation_scale)
        self.presymptomatic = stats.gamma(
            a=param.presymptomatic_shape, scale=param.incubation_scale
        )
        self.symptomatic = stats.gamma(a=param.symptomatic_shape, scale=param.symptomatic_scale)

    def _ensure_numerical_cdf(self) -> tuple[np.ndarray, np.ndarray]:
        if self._sampling_grid is None or self._cdf_grid is None:
            x = np.linspace(self.grid_min_days, self.grid_max_days, num=max(2, self.grid_points))
            y = np.clip(np.asarray(self.pdf(x), dtype=float), 0.0, np.inf)

            cdf = cumulative_trapezoid(y, x, initial=0)
            total = float(cdf[-1])

            if not np.isfinite(total) or total <= 0.0:
                cdf = (x - x[0]) / (x[-1] - x[0])
            else:
                cdf /= total
                cdf = np.maximum.accumulate(cdf)
                cdf[0] = 0.0
                cdf[-1] = 1.0

            self._sampling_grid = x
            self._cdf_grid = cdf

        return self._sampling_grid, self._cdf_grid

    def pdf(self, times_in_days: np.typing.ArrayLike) -> np.ndarray:
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

    def cdf(self, times_in_days: np.typing.ArrayLike) -> np.ndarray:
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

        evaluation_points = np.asarray(times_in_days, dtype=float)
        x, cdf = self._ensure_numerical_cdf()
        values = np.interp(evaluation_points, x, cdf, left=0.0, right=1.0)
        return values[()] if evaluation_points.ndim == 0 else values

    def mean(self) -> float:
        """Estimate the mean on the configured numerical grid.

        Returns
        -------
        float
            Numerical estimate of the mean.
        """

        integration_grid = np.linspace(
            self.grid_min_days, self.grid_max_days, num=max(100, self.grid_points)
        )
        probability_density = np.clip(self.pdf(integration_grid), a_min=0.0, a_max=np.inf)
        total_mass = float(np.trapezoid(probability_density, integration_grid))
        if not np.isfinite(total_mass) or total_mass <= 0.0:
            return float("nan")
        return float(
            np.trapezoid(integration_grid * probability_density, integration_grid) / total_mass
        )

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

    def sample_testing_delays(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        """Sample testing delays.

        Parameters
        ----------
        size : int or tuple of int, default=1
            Output shape.

        Returns
        -------
        numpy.ndarray
            Testing delays in days.
        """

        return self.rng.gamma(
            shape=self.parameters.testing_delay_shape,
            scale=self.parameters.testing_delay_scale,
            size=size,
        )

    def sample_clock_rate(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        """Sample substitution rates per day.

        Parameters
        ----------
        size : int or tuple of int, default=1
            Output shape.

        Returns
        -------
        numpy.ndarray
            Substitution rates in mutations per day.
        """
        per_site_per_year = self.rng.lognormal(
            np.log(self.parameters.substitution_rate), self.parameters.relaxation, size=size
        )
        return (per_site_per_year * self.parameters.genome_length) / 365.0

    def expected_mutations(
        self,
        times_in_days: np.typing.ArrayLike,
    ) -> np.ndarray:
        """Compute expected mutation counts for elapsed times.

        Parameters
        ----------
        times_in_days : array_like
            Times in days.
        size : int or tuple of int, default=1
            Shape of sampled rates when ``rates_per_day`` is None.

        Returns
        -------
        numpy.ndarray
            Expected mutation counts clipped to be non-negative.
        """

        times = np.asarray(times_in_days, dtype=float)
        clock_rate = self.sample_clock_rate(size=times.shape)
        return times * clock_rate

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"grid_min_days={self.grid_min_days}, "
            f"grid_max_days={self.grid_max_days}, "
            f"parameters={self.parameters})"
        )


class InfectiousnessToTransmission(BaseTransmissionProfile):
    r"""Distribution of time from the onset of infectiousness to transmission (:math:`y^*`).

    Density of the distribution is given by:

    .. math::

        f^*(y^*) =
        \begin{cases}
            0, & y^* < 0, \\
            C \left[ \alpha \left(1 - F_P(y^*)\right)
            + \int_0^{y^*} \left(1 - F_I(y^*-y_P)\right) f_P(y_P) \, d_{y_P} \right], & y^* \ge 0,
        \end{cases}

    where :math:`f_P` and :math:`F_P` are the presymptomatic PDF/CDF,
    :math:`F_I` is the symptomatic CDF, :math:`\alpha` is relative
    presymptomatic infectiousness, and :math:`C` is the normalisation
    constant from :class:`NaturalHistoryParameters`.

    Parameters
    ----------
    parameters : NaturalHistoryParameters, optional
        Stage-duration and infectiousness parameters for the E/P/I model.
    grid_min_days : float, default=0.0
        Lower bound of the numerical sampling grid in days.
    grid_max_days : float, default=60.0
        Upper bound of the numerical sampling grid in days.
    rng : Generator, optional
        Random number generator to use for sampling.
    rng_seed : int, optional
        Seed used to initialize ``rng`` when no generator is supplied.
    integration_grid_points : int, default=2048
         integration points used inside the PDF evaluation.
    sampling_grid_points : int, default=1024
         numerical grid points used for sampling from the profile.
    """

    def __init__(
        self,
        parameters: NaturalHistoryParameters | None = None,
        grid_min_days: float = 0.0,
        grid_max_days: float = 100.0,
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

    def pdf(self, times_in_days: np.typing.ArrayLike) -> np.ndarray:
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

        evaluation_points = np.asarray(times_in_days, dtype=float)
        flat_evaluation_points = evaluation_points.reshape(-1)
        probability_density = np.zeros_like(flat_evaluation_points)

        valid_indices = np.flatnonzero(flat_evaluation_points >= 0.0)
        if valid_indices.size == 0:
            if evaluation_points.ndim == 0:
                return probability_density.reshape(evaluation_points.shape)[()]
            return probability_density.reshape(evaluation_points.shape)

        param = self.parameters
        for index in valid_indices:
            toit_value = float(flat_evaluation_points[index])
            if toit_value == 0.0:
                symptomatic_contribution = 0.0
            else:
                presymptomatic_grid = np.linspace(
                    0.0,
                    toit_value,
                    num=max(2, self.integration_grid_points),
                )
                presymptomatic_density = self.presymptomatic.pdf(presymptomatic_grid)
                symptomatic_survival = 1.0 - self.symptomatic.cdf(toit_value - presymptomatic_grid)
                symptomatic_contribution = float(
                    np.trapezoid(symptomatic_survival * presymptomatic_density, presymptomatic_grid)
                )

            probability_density[index] = param.infectiousness_normalisation * (
                param.transmission_rate_ratio * (1.0 - self.presymptomatic.cdf(toit_value))
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
        x, cdf = self._ensure_numerical_cdf()
        u = self.rng.uniform(0.0, 1.0, size=sample_shape)
        return np.interp(u, cdf, x)


class SymptomOnsetToTransmission(BaseTransmissionProfile):
    r"""Distribution of time from the onset symptoms to transmission (:math:`x_{tost}`).

    Density of the distribution is given by:

    .. math::

        f_{tost}(x_{tost}) =
        \begin{cases}
            \alpha C \left(1 - F_P(-x_{tost})\right), & x_{tost} < 0, \\
            C \left(1 - F_I(x_{tost})\right), & x_{tost} \ge 0,
        \end{cases}

    where :math:`F_P` is the presymptomatic CDF, :math:`F_I` is the
    symptomatic CDF, :math:`\alpha` is relative presymptomatic
    infectiousness, and :math:`C` is the normalisation constant from
    :class:`NaturalHistoryParameters`.

    Parameters
    ----------
    parameters : NaturalHistoryParameters, optional
        Stage-duration and infectiousness parameters for the E/P/I model.
    grid_min_days : float, default=-30.0
        Lower bound of the numerical sampling grid in days.
    grid_max_days : float, default=30.0
        Upper bound of the numerical sampling grid in days.
    rng : Generator, optional
        Random number generator to use for sampling.
    rng_seed : int, optional
        Seed used to initialize ``rng`` when no generator is supplied.
    grid_points : int, default=2048
        numerical grid points used for summaries and sampling.
    """

    def __init__(
        self,
        parameters: NaturalHistoryParameters | None = None,
        grid_min_days: float = -30.0,
        grid_max_days: float = 30.0,
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

    def pdf(self, times_in_days: np.typing.ArrayLike) -> np.ndarray:
        r"""Evaluate the symptom-onset-to-transmission density.

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
            parameters.transmission_rate_ratio
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
        x, cdf = self._ensure_numerical_cdf()
        u = self.rng.uniform(0.0, 1.0, size=sample_shape)
        return np.interp(u, cdf, x)


__all__ = [
    "BaseTransmissionProfile",
    "InfectiousnessToTransmission",
    "SymptomOnsetToTransmission",
]
