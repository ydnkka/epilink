"""Monte Carlo draw containers for linkage inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..model.clock import MolecularClock
from ..model.profiles import InfectiousnessToTransmissionTime
from .kernels import genetic_kernel, temporal_kernel


@dataclass(frozen=True)
class LinkageMonteCarloSamples:
    """Monte Carlo draws used by the temporal and genetic kernels.

    Attributes
    ----------
    incubation_periods : numpy.ndarray
        Simulated incubation periods for paired cases with shape ``(N, 2)``.
    generation_intervals : numpy.ndarray
        Simulated generation-interval draws with shape ``(N, M + 1)``.
    sampling_delay_i : numpy.ndarray
        Simulated delay from infection to sampling for the first case.
    sampling_delay_j : numpy.ndarray
        Simulated delay from infection to sampling for the second case.
    diff_incubation_ij : numpy.ndarray
        Pairwise incubation-period differences.
    diff_infection_ij : numpy.ndarray
        Pairwise infection-time differences.
    clock_rates : numpy.ndarray
        Simulated substitution rates per day.
    """

    incubation_periods: npt.NDArray[np.float64]
    generation_intervals: npt.NDArray[np.float64]
    sampling_delay_i: npt.NDArray[np.float64]
    sampling_delay_j: npt.NDArray[np.float64]
    diff_incubation_ij: npt.NDArray[np.float64]
    diff_infection_ij: npt.NDArray[np.float64]
    clock_rates: npt.NDArray[np.float64]

    temporal_kernel = staticmethod(temporal_kernel)
    genetic_kernel = staticmethod(genetic_kernel)

    @classmethod
    def run_simulations(
        cls,
        transmission_profile: InfectiousnessToTransmissionTime,
        molecular_clock: MolecularClock,
        num_simulations: int,
        max_intermediate_hosts: int,
    ) -> LinkageMonteCarloSamples:
        """Sample epidemiological quantities for Monte Carlo inference.

        Parameters
        ----------
        transmission_profile : InfectiousnessToTransmissionTime
            Transmission-time model used to sample natural-history quantities.
        molecular_clock : MolecularClock
            Molecular clock used to sample substitution rates.
        num_simulations : int
            Number of Monte Carlo draws.
        max_intermediate_hosts : int
            Maximum number of intermediate hosts to simulate.

        Returns
        -------
        LinkageMonteCarloSamples
            Container with Monte Carlo draws used by the linkage kernels.
        """

        num_draws = int(num_simulations)
        max_intermediates = int(max_intermediate_hosts)

        incubation_periods = transmission_profile.sample_incubation_periods(size=(num_draws, 2))
        generation_intervals = transmission_profile.sample_generation_intervals(
            size=(num_draws, max_intermediates + 1)
        )
        transmission_offsets = transmission_profile.rvs(size=(num_draws, 2))
        clock_rates = molecular_clock.sample_substitution_rate_per_day(size=num_draws)

        return cls(
            incubation_periods=incubation_periods,
            generation_intervals=generation_intervals,
            sampling_delay_i=np.abs(generation_intervals[:, 0] - incubation_periods[:, 0]),
            sampling_delay_j=incubation_periods[:, 1],
            diff_incubation_ij=incubation_periods[:, 0] - incubation_periods[:, 1],
            diff_infection_ij=np.abs(transmission_offsets[:, 1] - transmission_offsets[:, 0]),
            clock_rates=clock_rates,
        )


__all__ = ["LinkageMonteCarloSamples"]
