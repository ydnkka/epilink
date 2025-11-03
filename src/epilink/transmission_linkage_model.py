"""
Estimate the probability of a transmission link between two cases from
genetic distance (SNPs) and temporal distance (days).

This module uses:
- A mechanistic infectiousness model (TOIT/TOST from infectiousness_profile.py)
- Monte Carlo simulation for epidemiological components
- Numba-accelerated kernels for the probability calculations

Public API:
- estimate_linkage_probability(...)
- estimate_linkage_probabilities(...)
- pairwise_linkage_probability_matrix(...)
- temporal_linkage_probability(...)
- genetic_linkage_probability(...)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

# Optional numba JIT with safe fallback
try:
    import numba

    JIT = numba.njit(cache=True, fastmath=True)
except ImportError:

    def JIT(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


from .infectiousness_profile import TOIT, InfectiousnessParams

ArrayLike = npt.ArrayLike


# =============================================================================
# 1) Numba kernels (nopython) — fast and allocation-conscious
# =============================================================================


@JIT
def _temporal_kernel(
    temporal_distance: np.ndarray,  # shape (K,)
    diff_inc: np.ndarray,  # shape (N,)
    generation_interval: np.ndarray,  # shape (N,), direct generation interval
) -> np.ndarray:
    """
    Compute temporal evidence P_t for each sampling interval by Monte Carlo.
    P_t(i) = mean_j [ |t_i + diff_inc_j| <= generation_interval0_j ]
    """
    K = temporal_distance.shape[0]
    N = diff_inc.shape[0]
    out = np.empty(K, dtype=np.float64)
    for i in range(K):
        t = temporal_distance[i]
        count = 0
        for j in range(N):
            if abs(t + diff_inc[j]) <= generation_interval[j]:
                count += 1
        out[i] = count / N
    return out


@JIT
def _mean_genetic_prob_kernel(
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
def _genetic_kernel(
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
        p_direct = _mean_genetic_prob_kernel(tmrca, dir_tmrca_exp, generation_interval[:, 0])
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
            p_suc = _mean_genetic_prob_kernel(tmrca, suc_tmrca, generation_interval[:, 0])

            # Common ancestor path: sum from idx to M-1 (exclusive of last col)
            com_sum = np.zeros(N, dtype=np.float64)
            for j in range(N):
                s = 0.0
                for col in range(idx, M):
                    s += generation_interval[j, col]
                com_sum[j] = s
            common_tmrca = toit_difference + inc_period_sum + com_sum
            p_common = _mean_genetic_prob_kernel(tmrca, common_tmrca, caseX_to_caseA)

            # Inclusion-exclusion
            out[k, m] = p_common + p_suc - (p_common * p_suc)

    return out


# =============================================================================
# 2) Simulation
# =============================================================================


def _run_simulations(
    toit: TOIT, num_simulations: int, no_intermediates: int
) -> dict[str, np.ndarray]:
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
    temporal_distance: ArrayLike,  # days; scalar or array (same length as genetic_distance)
    intermediate_generations: tuple[int, ...] = (0,),  # which m to include in final mixture
    no_intermediates: int = 10,  # max intermediates M used in simulation/kernel
    infectiousness_profile: InfectiousnessParams | None = None,  # Use default InfectiousnessParams
    subs_rate: float = 1e-3,  # subs/site/year (median)
    subs_rate_sigma: float = 0.33,  # lognormal sigma for relaxed clock
    relax_rate: bool = True,  # relaxed clock on/off
    num_simulations: int = 10000,  # Monte Carlo draws
    rng_seed: int = 12345,  # passed into TOIT
) -> float | np.ndarray:
    """
    End-to-end estimate of P(link | g, t) combining temporal and genetic evidence.

    This function mixes over intermediate-generation counts m using a two-step
    normalization of the genetic evidence and multiplies by the temporal evidence.

    Algorithm (per observation)
    ---------------------------
    1) Temporal evidence P_t(t):
         P_t = mean_j [ |t + (inc_A_j - inc_B_j)| <= GI_j ]
       via Monte Carlo draws from the TOIT model.
    2) Genetic evidence p_m(g) for m = 0..M:
         - Compute p_m via _genetic_kernel over simulated epidemiological quantities.
         - total = 1 - prod_m (1 - p_m)   [probability that at least one link occurs]
         - p_rel_m = p_m / total          [“relative” evidence; guards against no-link]
         - p_norm_m = p_rel_m / sum_m p_rel_m
       The final genetic mixture over selected m is:
         - sum_m p_norm_m for m in intermediate_generations
    3) Combine:
         P(link | g, t) = P_t * (sum over selected m of p_norm_m)

    Parameters
    ----------
    genetic_distance : ArrayLike
        SNP distance(s). Scalar or 1D array-like.
    temporal_distance : ArrayLike
        Temporal distance(s) in days. Must be the same length and shape-broadcast
        as genetic_distance (scalar or 1D array-like).
    intermediate_generations : tuple[int, ...], default=(0,)
        Which m values (0 = direct, 1 = one intermediate, ...) to include in the
        final normalized genetic mixture.
    no_intermediates : int, default=10
        Maximum m (inclusive) to simulate, i.e., M in m=0..M.
    infectiousness_profile : InfectiousnessParams | None, default=None
        Parameters passed to TOIT; None uses the module defaults.
    subs_rate : float, default=1e-3
        Substitutions per site per year (median of the clock-rate prior).
    subs_rate_sigma : float, default=0.33
        Lognormal sigma of the relaxed clock.
    relax_rate : bool, default=False
        If True, use a relaxed molecular clock; otherwise strict.
    num_simulations : int, default=10000
        Number of Monte Carlo draws.
    rng_seed : int, default=12345
        Seed passed to TOIT for reproducibility.

    Returns
    -------
    float or np.ndarray
        - float, if both inputs are scalars
        - 1D numpy array of shape (K,), where K = len(np.atleast_1d(genetic_distance)),
          if either input is array-like.
        - np.nan if K == 0 (empty input).

    Raises
    ------
    ValueError
        - If genetic_distance and temporal_distance lengths differ
        - If any requested intermediate generation lies outside [0, M]

    Notes
    -----
    - Temporal and genetic components are computed via numba-accelerated kernels.
    - For direct access to components, see:
        temporal_linkage_probability(...) and genetic_linkage_probability(...).
    - The “normalized” genetic mixture sums to 1 across m and is then restricted
      to the requested m values before combining with P_t.

    Examples
    --------
    >>> # Scalar inputs -> scalar probability
    >>> p = estimate_linkage_probability(2, 7, intermediate_generations=(0,1), rng_seed=1)
    >>> isinstance(p, float)
    True

    >>> # Vectorized inputs -> 1D array of probabilities
    >>> gd = [0, 1, 2]
    >>> td = [2, 5, 10]
    >>> estimate_linkage_probability(gd, td, intermediate_generations=(0,)).shape
    (3,)
    """

    # 1) Prepare inputs as 1D arrays of same length
    g = np.atleast_1d(np.asarray(genetic_distance, dtype=float))
    t = np.atleast_1d(np.asarray(temporal_distance, dtype=float))
    if g.size != t.size:
        raise ValueError(
            f"genetic_distance and temporal_distance must have the same length, got {g.size} vs {t.size}."
        )

    # Early return for empty input
    if g.size == 0:
        return np.nan

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
        temporal_distance=t,
        diff_inc=sim["diff_inc"],
        generation_interval=sim["generation_interval"][:, 0],
    )  # shape (K,)

    # 4) Genetic evidence: p_m for each m=0..M
    p_m = _genetic_kernel(
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

    # 5) Normalize genetic evidence across m
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
        raise ValueError(
            f"intermediate_generations must be within [0, {M}], got {intermediate_generations}."
        )
    selected = p_norm[:, cols].sum(axis=1)  # shape (K,)

    # 6) Combine temporal and genetic evidence
    out = p_t * selected  # shape (K,)

    # 7) Return scalar if scalar input
    if np.isscalar(genetic_distance) and np.isscalar(temporal_distance):
        return float(out[0])
    return out


def estimate_linkage_probabilities(
    genetic_distance: ArrayLike,
    temporal_distance: ArrayLike,
    intermediate_generations: tuple[int, ...] = (0,),
    no_intermediates: int = 10,
    **kwargs: Any,
) -> np.ndarray:
    """
    Vectorized helper to compute P(link | g_i, t_i) for many observations efficiently.

    This function:
      1) extracts sorted unique values from the inputs,
      2) computes a full P(link | g, t) matrix over the unique grid via
         pairwise_linkage_probability_matrix, and
      3) maps each observation back to its probability.

    Parameters
    ----------
    genetic_distance : ArrayLike
        SNP distances; numeric/coercible to float. Length N.
    temporal_distance : ArrayLike
        Temporal distances in days; numeric/coercible to float. Length N.
    intermediate_generations : tuple[int, ...], default=(0,)
        Which m values to include in the final normalized genetic mixture.
        Passed through to pairwise_linkage_probability_matrix (and ultimately
        estimate_linkage_probability).
    no_intermediates : int, default=10
        Maximum number of intermediates M used in simulation/kernel. Passed through.
    **kwargs : Any
        Additional keyword arguments forwarded to pairwise_linkage_probability_matrix,
        e.g.:
          - infectiousness_profile: InfectiousnessParams | None
          - subs_rate: float
          - subs_rate_sigma: float
          - relax_rate: bool
          - num_simulations: int
          - rng_seed: int

    Returns
    -------
    np.ndarray
        A 1D array of probabilities (dtype float), shape (N,), aligned with the
        input ordering. Returns an empty array if N == 0.

    Raises
    ------
    ValueError
        - If the input lengths differ
        - If the probability matrix from pairwise_linkage_probability_matrix has
          an unexpected shape

    Notes
    -----
    - Using unique grids avoids recomputing probabilities for repeated (g, t) pairs.
    - The stochastic components (Monte Carlo draws) are controlled by the rng_seed
      and other kwargs passed into the underlying estimator.

    Examples
    --------
    >>> gd = [0, 0, 1, 2]
    >>> td = [5, 5, 5, 10]
    >>> probs = estimate_linkage_probabilities(gd, td, intermediate_generations=(0,1), rng_seed=7)
    >>> probs.shape
    (4,)
    """

    g = np.atleast_1d(np.asarray(genetic_distance, dtype=float))
    t = np.atleast_1d(np.asarray(temporal_distance, dtype=float))
    if g.size != t.size:
        raise ValueError(
            f"genetic_distance and temporal_distance must have the same length, got {g.size} vs {t.size}."
        )

    # Early return for empty input
    if g.size == 0:
        return np.array([], dtype=float)

    # Get sorted unique values and inverse indices for vectorized lookup
    gd_unique, gi = np.unique(g, return_inverse=True)
    td_unique, tj = np.unique(t, return_inverse=True)

    # Compute full probability matrix over the unique grid
    prob_matrix = pairwise_linkage_probability_matrix(
        genetic_distances=gd_unique,
        temporal_distances=td_unique,
        intermediate_generations=intermediate_generations,
        no_intermediates=no_intermediates,
        **kwargs,
    )

    expected_shape = (gd_unique.size, td_unique.size)
    if getattr(prob_matrix, "shape", None) != expected_shape:
        raise ValueError(
            "pairwise_linkage_probability_matrix returned shape "
            f"{getattr(prob_matrix, 'shape', None)}, expected {expected_shape}"
        )

    # Vectorized mapping to per-observation probabilities
    probs = np.asarray(prob_matrix, dtype=float)[gi, tj]
    return probs


def pairwise_linkage_probability_matrix(
    genetic_distances: np.ndarray,  # 1D
    temporal_distances: np.ndarray,  # 1D
    intermediate_generations: tuple[int, ...] = (0,),
    no_intermediates: int = 10,
    **kwargs,
) -> np.ndarray:
    """
    Compute a (G x T) matrix of P(link | g, t) over a grid of genetic and temporal distances.

    Parameters
    ----------
    genetic_distances : array-like of float, shape (G,)
        Unique or non-unique SNP distances to place on the grid’s rows.
        Duplicates are allowed (rows will be duplicated accordingly).
    temporal_distances : array-like of float, shape (T,)
        Unique or non-unique temporal distances in days to place on the grid’s columns.
        Duplicates are allowed (columns will be duplicated accordingly).
    intermediate_generations : tuple[int, ...], default=(0,)
        Which m values to include in the final normalized genetic mixture.
        Forwarded to estimate_linkage_probability.
    no_intermediates : int, default=10
        Maximum number of intermediates M used in simulation/kernel. Forwarded through.
    **kwargs
        Additional keyword arguments forwarded to estimate_linkage_probability, e.g.:
          - infectiousness_profile: InfectiousnessParams | None
          - subs_rate: float
          - subs_rate_sigma: float
          - relax_rate: bool
          - num_simulations: int
          - rng_seed: int

    Returns
    -------
    np.ndarray
        Matrix of shape (G, T), where entry [i, j] is P(link | g=genetic_distances[i],
        t=temporal_distances[j]) computed using the mixture logic over m and the
        specified kwargs.

    Raises
    ------
    ValueError
        Propagated from estimate_linkage_probability for invalid inputs, e.g., if
        any requested intermediate generation lies outside [0, M].

    Notes
    -----
    - Internally uses a meshgrid and flattens the grid to make a single call to
      estimate_linkage_probability, which runs the simulation once and evaluates
      all pairs in a vectorized manner.
    - For per-observation convenience over non-gridded pairs, see
      estimate_linkage_probabilities.

    Examples
    --------
    >>> gd = np.array([0, 1, 2])
    >>> td = np.array([0, 5, 10])
    >>> M = pairwise_linkage_probability_matrix(gd, td, intermediate_generations=(0,1), rng_seed=11)
    >>> M.shape
    (3, 3)
    """
    # ... existing implementation ...

    gd = np.asarray(genetic_distances, dtype=float)
    td = np.asarray(temporal_distances, dtype=float)

    g_grid, t_grid = np.meshgrid(gd, td, indexing="ij")  # each shape (G, T)
    flat_g = g_grid.ravel()
    flat_t = t_grid.ravel()

    flat_p = estimate_linkage_probability(
        genetic_distance=flat_g,
        temporal_distance=flat_t,
        intermediate_generations=intermediate_generations,
        no_intermediates=no_intermediates,
        **kwargs,
    )
    flat_p = np.asarray(flat_p, dtype=float)
    return flat_p.reshape(g_grid.shape)


def temporal_linkage_probability(
    temporal_distance: ArrayLike,
    toit: TOIT,
    num_simulations: int = 10000,
) -> np.ndarray:
    """
    Monte Carlo estimate of the temporal evidence P_t for one or more sampling-intervals.

    For each temporal distance t_i (days between case sampling dates), this returns
    the probability that the (absolute) adjusted interval falls within a directly
    simulated generation interval:
        P_t(i) = mean_j [ |t_i + (inc_A_j - inc_B_j)| <= GI_j ]

    where inc_A/inc_B are incubation periods for cases A and B, and GI is the
    generation interval, all sampled from the TOIT model.

    Parameters
    ----------
    temporal_distance : ArrayLike
        One or more temporal distances (days). Can be scalar or 1D array-like.
        The sign convention is not enforced; negative values are handled.
    toit : TOIT
        A configured infectiousness/clock model instance used to draw incubation
        and generation-time samples (see infectiousness_profile.TOIT).
    num_simulations : int, default=10000
        Number of Monte Carlo draws.

    Returns
    -------
    np.ndarray
        A 1D array of shape (K,), where K = len(np.atleast_1d(temporal_distance)),
        containing the temporal evidence P_t for each input temporal distance.

    Notes
    -----
    - Reproducibility is controlled by the rng_seed in the provided TOIT instance.
    - This function provides only the temporal component; combine with the genetic
      component (or use estimate_linkage_probability) to obtain P(link | g, t).

    Examples
    --------
    >>> toit = TOIT(rng_seed=123)
    >>> temporal_linkage_probability([0, 5, 10], toit=toit, num_simulations=5000)
    array([0.42..., 0.33..., 0.21...])
    """
    td = np.atleast_1d(np.asarray(temporal_distance, dtype=float))  # ensure 1D array

    inc_period = toit.sample_incubation(size=(num_simulations, 2))  # (N, 2)
    diff_inc = inc_period[:, 0] - inc_period[:, 1]  # (N,)
    gen_interval = toit.generation_time(size=num_simulations)  # (N,)

    return _temporal_kernel(
        temporal_distance=td,
        diff_inc=diff_inc,
        generation_interval=gen_interval,
    )


def genetic_linkage_probability(
    genetic_distance: ArrayLike,
    toit: TOIT,
    num_simulations: int = 10000,
    no_intermediates: int = 10,
    intermediate_generations: tuple[int, ...] | None = None,  # e.g. (0, 1)
    kind: str = "relative",  # "raw" | "relative" | "normalized"
) -> np.ndarray:
    """
    Monte Carlo estimate of the genetic evidence across intermediate-generation scenarios.

    For each genetic distance g_k (SNPs), this computes the vector p_m for m=0..M where:
      - p_m (kind="raw") is the scenario-wise probability of observing g_k under the model
        with exactly m intermediates on the transmission path (m=0 is direct).
      - p_m (kind="relative") rescales the raw p_m by the probability of at least one
        link: total = 1 - prod_m(1 - p_m). That is, p_m / total. The sum across m can
        exceed 1 (union bound).
      - p_m (kind="normalized") further normalizes the relative values to sum to 1 across m.

    If `intermediate_generations` is provided (e.g., (0, 1)), the function aggregates
    across those columns:
      - kind="raw" or "relative": returns the mean across selected m.
      - kind="normalized": returns the sum across selected m (i.e., the share of the
        normalized mass assigned to those m).

    Parameters
    ----------
    genetic_distance : ArrayLike
        One or more genetic distances (SNPs). Can be scalar or 1D array-like.
    toit : TOIT
        A configured infectiousness/clock model instance from which the epidemiological
        and clock-rate quantities are drawn.
    num_simulations : int, default=10000
        Number of Monte Carlo draws.
    no_intermediates : int, default=10
        M, the maximum number of intermediate generations simulated (m=0..M).
    intermediate_generations : tuple[int, ...] | None, default=None
        If provided, restricts/aggregates output to the specified m-values.
        Values must lie within [0, M].
        - If None: return a matrix of shape (K, M+1) over all m.
        - If provided: return a 1D vector of shape (K,) aggregated as described above.
    kind : {"raw", "relative", "normalized"}, default="relative"
        Selects which genetic evidence to return:
        - "raw": scenario-wise probabilities p_m as returned by the kernel
        - "relative": p_m divided by total = 1 - prod(1 - p_m)
        - "normalized": "relative" values renormalized to sum to 1 across m

    Returns
    -------
    np.ndarray
        - If intermediate_generations is None: array of shape (K, M+1)
        - Else: array of shape (K,)
        Here K = len(np.atleast_1d(genetic_distance)).

    Raises
    ------
    ValueError
        If any requested intermediate generation is outside [0, M], or if kind is invalid.

    Notes
    -----
    - This function provides only the genetic component; combine with the temporal
      component (or use estimate_linkage_probability) to obtain P(link | g, t).
    - Reproducibility is controlled by the rng_seed in the provided TOIT instance.
    - Interpretation:
        * "raw" reflects scenario-wise evidence magnitudes.
        * "relative" conditions on at least one transmission link being present.
          Sums across m may exceed 1.
        * "normalized" provides a proper distribution across m that sums to 1.

    Examples
    --------
    >>> toit = TOIT(rng_seed=123)
    >>> # Full matrix over m=0..M for g in {0,1,2}
    >>> genetic_linkage_probability([0, 1, 2], toit=toit, no_intermediates=5, kind="normalized").shape
    (3, 6)

    >>> # Aggregate over direct and one intermediate, returning a 1D vector per g
    >>> genetic_linkage_probability([0, 1, 2], toit=toit, no_intermediates=5,
    ...                             intermediate_generations=(0, 1), kind="normalized").shape
    (3,)
    """
    g = np.atleast_1d(np.asarray(genetic_distance, dtype=float))  # ensure 1D array
    M = int(no_intermediates)

    sim = _run_simulations(toit, int(num_simulations), M)

    # Genetic evidence: p_m for each m=0..M; shape (K, M+1)
    p_m = _genetic_kernel(
        dists=g,
        clock_rates=sim["clock_rates"],
        psi_sA=sim["psi_sA"],
        psi_sB=sim["psi_sB"],
        generation_interval=sim["generation_interval"],
        toit_difference=sim["toit_difference"],
        incubation_period=sim["incubation_period"],
        caseX_to_caseA=sim["caseX_to_caseA"],
        intermediates=M,
    )

    # total probability of at least one link per row
    total = 1.0 - np.prod(1.0 - p_m, axis=1)  # shape (K,)

    # "relative" values: avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        p_rel = np.where(total[:, None] > 0.0, p_m / total[:, None], 0.0)  # (K, M+1)

    # "normalized" values: renormalize to sum to 1 across m
    row_sums = p_rel.sum(axis=1)  # (K,)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_norm = np.where(row_sums[:, None] > 0.0, p_rel / row_sums[:, None], 0.0)

    if intermediate_generations is not None:
        cols = np.array(intermediate_generations, dtype=np.int64)
        if cols.min() < 0 or cols.max() > M:
            raise ValueError(
                f"intermediate_generations must be within [0, {M}], got {intermediate_generations}."
            )

        if kind == "relative":
            selected = p_rel[:, cols].mean(axis=1)  # (K,)
        elif kind == "raw":
            selected = p_m[:, cols].mean(axis=1)  # (K,)
        elif kind == "normalized":
            selected = p_norm[:, cols].sum(axis=1)  # (K,)
        else:
            raise ValueError(f"kind must be 'relative', 'raw', or 'normalized', got '{kind}'.")
        return selected
    else:
        if kind == "relative":
            return p_rel  # (K, M+1)
        elif kind == "raw":
            return p_m  # (K, M+1)
        elif kind == "normalized":
            return p_norm  # (K, M+1)
        else:
            raise ValueError(f"kind must be 'relative', 'raw', or 'normalized', got '{kind}'.")
