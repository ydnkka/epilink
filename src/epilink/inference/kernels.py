"""
Numba kernels used by transmission linkage inference.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, fastmath=True)
def temporal_kernel(
    temporal_distance_ij: np.ndarray,
    diff_incubation_ij: np.ndarray,
    generation_interval: np.ndarray,
) -> np.ndarray:
    """
    Estimate temporal evidence from Monte Carlo samples.
    """

    K = temporal_distance_ij.shape[0]
    N = diff_incubation_ij.shape[0]
    out = np.zeros(K, dtype=np.float64)

    for i in prange(K):
        t = temporal_distance_ij[i]
        cnt = 0
        for j in range(N):
            if math.fabs(t + diff_incubation_ij[j]) <= generation_interval[j]:
                cnt += 1
        out[i] = cnt / N
    return out


@njit(cache=True, parallel=True, fastmath=True)
def genetic_kernel(
    genetic_distance_ij: np.ndarray,
    clock_rates: np.ndarray,
    sampling_delay_i: np.ndarray,
    sampling_delay_j: np.ndarray,
    generation_intervals: np.ndarray,
    max_intermediate_hosts: int,
    diff_infection_ij: np.ndarray,
    incubation_periods: np.ndarray,
) -> np.ndarray:
    """
    Estimate genetic evidence across intermediate scenarios.
    """

    K = genetic_distance_ij.shape[0]
    N = clock_rates.shape[0]
    M = max_intermediate_hosts

    out = np.zeros((K, M + 1), dtype=np.float64)

    direct_tmrca_expected = sampling_delay_i + sampling_delay_j
    incubation_period_sum = incubation_periods[:, 0] + incubation_periods[:, 1]
    inv_2_clock = 0.5 / clock_rates

    suffix = np.empty((N, M + 2), dtype=np.float64)
    for j in prange(N):
        suffix[j, M + 1] = 0.0
        for c in range(M, -1, -1):
            suffix[j, c] = suffix[j, c + 1] + generation_intervals[j, c]

    for k in prange(K):
        d = genetic_distance_ij[k]
        if d < 0:
            continue

        dir0 = 0
        common_source0 = 0
        for j in range(N):
            tmrca_obs = d * inv_2_clock[j]
            tmrca_dir = direct_tmrca_expected[j]
            tmrca_cs = diff_infection_ij[j] + incubation_period_sum[j]
            if math.fabs(tmrca_obs - tmrca_dir) <= generation_intervals[j, 0]:
                dir0 += 1
            if math.fabs(tmrca_obs - tmrca_cs) <= generation_intervals[j, 0]:
                common_source0 += 1
        out[k, 0] = (dir0 / N) + (common_source0 / N)

        for m in range(1, M + 1):
            idx = M - (m - 1)

            chain = 0
            common_source = 0
            for j in range(N):
                tmrca_obs = d * inv_2_clock[j]
                tmrca_chain = direct_tmrca_expected[j] + suffix[j, idx]
                tmrca_cs = diff_infection_ij[j] + incubation_period_sum[j] + suffix[j, idx]

                if math.fabs(tmrca_obs - tmrca_chain) <= generation_intervals[j, 0]:
                    chain += 1

                if math.fabs(tmrca_obs - tmrca_cs) <= generation_intervals[j, 0]:
                    common_source += 1

            out[k, m] = (chain / N) + (common_source / N)

    return out


def _ensure_pyfunc(func: Any) -> None:
    """Ensure a .py_func attribute for non-numba callables."""

    if not hasattr(func, "py_func"):
        func.py_func = func


_ensure_pyfunc(temporal_kernel)
_ensure_pyfunc(genetic_kernel)


__all__ = ["genetic_kernel", "temporal_kernel"]
