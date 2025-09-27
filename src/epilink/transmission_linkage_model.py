"""
Estimate the probability of a transmission link between two cases from
genetic distance (SNPs) and temporal distance (days).

This module uses:
- A mechanistic infectiousness model (TOIT/TOST from infectiousness_profile.py)
- Monte Carlo simulation for epidemiological components
- Numba-accelerated kernels for the probability calculations

Public API:
- estimate_linkage_probability(...)
- pairwise_linkage_probability_matrix(...)
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np

# Optional numba JIT with safe fallback
try:
    import numba  # type: ignore

    JIT = numba.njit(cache=True, fastmath=True)
except ImportError:  # pragma: no cover
    def JIT(*args, **kwargs):  # type: ignore
        def wrapper(func):
            return func
        return wrapper

from .infectiousness_profile import InfectiousnessParams, TOIT

ArrayLike = Union[float, Iterable[float], np.ndarray]


# =============================================================================
# 1) Numba kernels (nopython) — fast and allocation-conscious
# =============================================================================

@JIT
def _temporal_kernel(
    sampling_interval: np.ndarray,  # shape (K,)
    diff_inc: np.ndarray,  # shape (N,)
    generation_interval0: np.ndarray,  # shape (N,), direct generation interval
) -> np.ndarray:
    """
    Compute temporal evidence P_t for each sampling interval by Monte Carlo.
    P_t(i) = mean_j [ |t_i + diff_inc_j| <= generation_interval0_j ]
    """
    K = sampling_interval.shape[0]
    N = diff_inc.shape[0]
    out = np.empty(K, dtype=np.float64)
    for i in range(K):
        t = sampling_interval[i]
        count = 0
        for j in range(N):
            if abs(t + diff_inc[j]) <= generation_interval0[j]:
                count += 1
        out[i] = count / N
    return out


@JIT
def _mean_prob_kernel(
    tmrca: np.ndarray,  # shape (N,)
    tmrca_expected: np.ndarray,  # shape (N,)
    tolerance: np.ndarray,  # shape (N,)
) -> float:
    """
    Mean over simulations of the event:
    |tmrca_j - tmrca_expected_j| <= tolerance_j
    """
    N = tmrca.shape[0]
    count = 0
    for j in range(N):
        if abs(tmrca[j] - tmrca_expected[j]) <= tolerance[j]:
            count += 1
    return count / N


@JIT
def _genetic_kernel_many(
    dists: np.ndarray,  # shape (K,)
    clock_rates: np.ndarray,  # shape (N,)
    psi_sA: np.ndarray,  # shape (N,)
    psi_sB: np.ndarray,  # shape (N,)
    generation_interval: np.ndarray,  # shape (N, M+1)
    toit_difference: np.ndarray,  # shape (N,)
    incubation_period: np.ndarray,  # shape (N, 2)
    caseX_to_caseA: np.ndarray,  # shape (N,)
    intermediates: int,  # M (max number of intermediates)
) -> np.ndarray:
    """
    For each genetic distance d in dists, compute vector p_m for m=0..M,
    where p_m is the genetic evidence under the scenario of m intermediates.
    Returns an array of shape (K, M+1).
    """
    K = dists.shape[0]
    N = clock_rates.shape[0]
    M = intermediates

    out = np.zeros((K, M + 1), dtype=np.float64)

    # Invariants across m
    dir_tmrca_exp = psi_sA + psi_sB  # shape (N,)
    inc_period_sum = incubation_period[:, 0] + incubation_period[:, 1]  # shape (N,)

    for k in range(K):
        d = dists[k]
        # tmrca per simulation for this distance
        tmrca = d / (2.0 * clock_rates)  # shape (N,)

        # m = 0: direct
        p_direct = _mean_prob_kernel(tmrca, dir_tmrca_exp, generation_interval[:, 0])
        out[k, 0] = p_direct

        # m > 0
        for m in range(1, M + 1):
            idx = M - (m - 1)

            # Successive transmission path
            # sum generation intervals from idx to end (inclusive)
            suc_sum = np.zeros(N, dtype=np.float64)
            for j in range(N):
                s = 0.0
                for col in range(idx, M + 1):
                    s += generation_interval[j, col]
                suc_sum[j] = s
            suc_tmrca = dir_tmrca_exp + suc_sum
            p_suc = _mean_prob_kernel(tmrca, suc_tmrca, generation_interval[:, 0])

            # Common ancestor path: sum from idx to M-1 (exclusive of last col)
            com_sum = np.zeros(N, dtype=np.float64)
            for j in range(N):
                s = 0.0
                for col in range(idx, M):
                    s += generation_interval[j, col]
                com_sum[j] = s
            common_tmrca = toit_difference + inc_period_sum + com_sum
            p_common = _mean_prob_kernel(tmrca, common_tmrca, caseX_to_caseA)

            # Inclusion-exclusion
            out[k, m] = p_common + p_suc - (p_common * p_suc)

    return out


# =============================================================================
# 2) Simulation
# =============================================================================

def _run_simulations(toit: TOIT, num_simulations: int, no_intermediates: int) -> Dict[str, np.ndarray]:
    """
    Draw all random epidemiological quantities once.
    """
    N = int(num_simulations)
    M = int(no_intermediates)

    inc_period = toit.sample_incubation(size=(N, 2))  # (N, 2)
    gen_interval = toit.generation_time(size=(N, M + 1))  # (N, M+1)
    toit_values = toit.rvs(size=(N, 2))  # (N, 2)
    latent_period = toit.sample_E(size=N)  # (N,)
    clock_rates = toit.sample_clock_rate_per_day(size=N)  # (N,)

    sim = {
        "incubation_period": inc_period,
        "generation_interval": gen_interval,
        "psi_sA": np.abs(gen_interval[:, 0] - inc_period[:, 0]),
        "psi_sB": inc_period[:, 1],
        "diff_inc": inc_period[:, 0] - inc_period[:, 1],
        "caseX_to_caseA": latent_period + np.minimum(toit_values[:, 0], toit_values[:, 1]),
        "toit_difference": np.abs(toit_values[:, 1] - toit_values[:, 0]),
        "clock_rates": clock_rates,
    }
    return sim


# =============================================================================
# 3) Public API
# =============================================================================

def estimate_linkage_probability(
    genetic_distance: ArrayLike,  # SNPs; scalar or array
    sampling_interval: ArrayLike,  # days; scalar or array (same length as genetic_distance)
    intermediate_generations: Tuple[int, ...] = (0,),  # which m to include in final mixture
    no_intermediates: int = 10,  # max intermediates M used in simulation/kernel
    infectiousness_profile: Optional[InfectiousnessParams] = None,  # Use default InfectiousnessParams
    subs_rate: float = 1e-3,  # subs/site/year (median)
    subs_rate_sigma: float = 0.33,  # lognormal sigma for relaxed clock
    relax_rate: bool = False,  # relaxed clock on/off
    num_simulations: int = 10000,  # Monte Carlo draws
    rng_seed: int = 12345,  # passed into TOIT
) -> Union[float, np.ndarray]:
    """
    Estimate P(link | g, t) combining:
      - Temporal evidence P_t(given t)
      - Genetic evidence P_d(given g), marginalized over selected intermediate counts

    The genetic evidence per m is normalized:
      - Compute p_m for m=0....M
      - total = 1 - prod_m (1 - p_m)  [prob of at least one link]
      - p'_m = p_m / total
      - Normalize across m to sum to 1, then sum only for requested `intermediate_generations`.

    Returns:
      - float if inputs are scalars
      - 1D numpy array if inputs are arrays
    """
    # 1) Prepare inputs as 1D arrays of same length
    g = np.atleast_1d(np.asarray(genetic_distance, dtype=float))
    t = np.atleast_1d(np.asarray(sampling_interval, dtype=float))
    if g.shape[0] != t.shape[0]:
        raise ValueError(
            f"genetic_distance and sampling_interval must have the same length, got {g.shape[0]} vs {t.shape[0]}."
        )

    K = g.shape[0]
    M = int(no_intermediates)

    # 2) Initialize model and run simulations once
    toit = TOIT(
        params=infectiousness_profile,
        rng_seed=int(rng_seed),
        subs_rate=float(subs_rate),
        subs_rate_sigma=float(subs_rate_sigma),
        relax_rate=bool(relax_rate),
    )
    sim = _run_simulations(toit, int(num_simulations), M)

    # 3) Temporal evidence
    p_t = _temporal_kernel(
        sampling_interval=t,
        diff_inc=sim["diff_inc"],
        generation_interval0=sim["generation_interval"][:, 0],
    )  # shape (K,)

    # 4) Genetic evidence: p_m for each m=0..M
    p_m = _genetic_kernel_many(
        dists=g,
        clock_rates=sim["clock_rates"],
        psi_sA=sim["psi_sA"],
        psi_sB=sim["psi_sB"],
        generation_interval=sim["generation_interval"],
        toit_difference=sim["toit_difference"],
        incubation_period=sim["incubation_period"],
        caseX_to_caseA=sim["caseX_to_caseA"],
        intermediates=M,
    )  # shape (K, M+1)

    # 5) Normalize genetic evidence across m following the original approach
    # total probability of at least one link per row
    total = 1.0 - np.prod(1.0 - p_m, axis=1)  # shape (K,)
    # Avoid division by zero: where total==0, keep zeros
    with np.errstate(divide="ignore", invalid="ignore"):
        p_rel = np.where(total[:, None] > 0.0, p_m / total[:, None], 0.0)  # shape (K, M+1)
    row_sums = p_rel.sum(axis=1)  # shape (K,)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_norm = np.where(row_sums[:, None] > 0.0, p_rel / row_sums[:, None], 0.0)

    # Select columns m specified by intermediate_generations
    cols = np.array(intermediate_generations, dtype=np.int64)
    if cols.min() < 0 or cols.max() > M:
        raise ValueError(f"intermediate_generations must be within [0, {M}], got {intermediate_generations}.")
    selected = p_norm[:, cols].sum(axis=1)  # shape (K,)

    # 6) Combine temporal and genetic evidence
    out = p_t * selected  # shape (K,)

    # 7) Return scalar if scalar input
    if np.isscalar(genetic_distance) and np.isscalar(sampling_interval):
        return float(out[0])
    return out


def pairwise_linkage_probability_matrix(
    genetic_distances: np.ndarray,  # 1D
    temporal_distances: np.ndarray,  # 1D
    intermediate_generations: Tuple[int, ...] = (0,),
    no_intermediates: int = 10,
    **kwargs,
) -> np.ndarray:
    """
    Compute a (len(genetic_distances) x len(temporal_distances)) matrix of P(link | g, t).
    """
    gd = np.asarray(genetic_distances, dtype=float)
    td = np.asarray(temporal_distances, dtype=float)

    g_grid, t_grid = np.meshgrid(gd, td, indexing="ij")  # each shape (G, T)
    flat_g = g_grid.ravel()
    flat_t = t_grid.ravel()

    flat_p = estimate_linkage_probability(
        genetic_distance=flat_g,
        sampling_interval=flat_t,
        intermediate_generations=intermediate_generations,
        no_intermediates=no_intermediates,
        **kwargs,
    )
    flat_p = np.asarray(flat_p, dtype=float)
    return flat_p.reshape(g_grid.shape)
