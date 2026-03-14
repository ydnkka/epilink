"""Molecular clock models used by epilink."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.random import Generator, default_rng


class MolecularClock:
    """Molecular clock model for substitution-rate sampling.

    Parameters
    ----------
    substitution_rate : float, default=1e-3
        Median substitution rate in substitutions per site per year.
    use_relaxed_clock : bool, default=True
        If ``True``, sample branch-specific rates from a lognormal
        distribution. If ``False``, use a strict clock.
    relaxed_clock_sigma : float, default=0.33
        Lognormal standard deviation for the relaxed clock.
    genome_length : int, default=29903
        Genome length in sites.
    rng : Generator, optional
        Random number generator to use for sampling.
    rng_seed : int, optional
        Seed used to initialize ``rng`` when no generator is supplied.
    """

    def __init__(
        self,
        substitution_rate: float = 1e-3,
        use_relaxed_clock: bool = True,
        relaxed_clock_sigma: float = 0.33,
        genome_length: int = 29903,
        rng: Generator | None = None,
        rng_seed: int | None = None,
    ) -> None:
        self.substitution_rate = float(substitution_rate)
        self.use_relaxed_clock = bool(use_relaxed_clock)
        self.relaxed_clock_sigma = float(relaxed_clock_sigma)
        self.genome_length = int(genome_length)
        self.rng: Generator = rng if rng is not None else default_rng(rng_seed)

        if self.substitution_rate <= 0:
            raise ValueError("substitution_rate must be positive.")
        if self.relaxed_clock_sigma < 0:
            raise ValueError("relaxed_clock_sigma must be non-negative.")
        if self.genome_length <= 0:
            raise ValueError("genome_length must be positive.")

        self.log_substitution_rate_location = np.log(self.substitution_rate)

    def sample_substitution_rate_per_day(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
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

        if self.use_relaxed_clock:
            per_site_per_year = self.rng.lognormal(
                self.log_substitution_rate_location, self.relaxed_clock_sigma, size=size
            )
        else:
            per_site_per_year = np.full(size, self.substitution_rate, dtype=float)
        return (per_site_per_year * self.genome_length) / 365.0

    def estimate_expected_mutations(
        self,
        times_in_days: npt.ArrayLike,
        rates_per_day: npt.ArrayLike | None = None,
        size: int | tuple[int, ...] = 1,
    ) -> np.ndarray:
        """Compute expected mutation counts for elapsed times.

        Parameters
        ----------
        times_in_days : array_like
            Times in days.
        rates_per_day : array_like, optional
            Per-day substitution rates. If None, sample them from the clock.
        size : int or tuple of int, default=1
            Shape of sampled rates when ``rates_per_day`` is None.

        Returns
        -------
        numpy.ndarray
            Expected mutation counts clipped to be non-negative.
        """

        times = np.asarray(times_in_days, dtype=float)
        if rates_per_day is None:
            if isinstance(size, int) and size == 1:
                rate = float(self.sample_substitution_rate_per_day(size=1).item())
                mut = rate * times
            else:
                rates = np.asarray(self.sample_substitution_rate_per_day(size=size), dtype=float)
                expand_axes = (None,) * times.ndim
                mut = rates[(...,) + expand_axes] * times
        else:
            mut = np.asarray(rates_per_day, dtype=float) * times
        return np.clip(mut, a_min=0.0, a_max=np.inf)

    def __repr__(self) -> str:
        return (
            f"MolecularClock("
            f"substitution_rate={self.substitution_rate}, "
            f"use_relaxed_clock={self.use_relaxed_clock}, "
            f"relaxed_clock_sigma={self.relaxed_clock_sigma}, "
            f"genome_length={self.genome_length})"
        )


__all__ = ["MolecularClock"]
