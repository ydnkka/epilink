"""
Comprehensive tests for the transmission_linkage_model module.
"""

from __future__ import annotations

import numpy as np
import pytest

from epilink import (
    TOIT,
    Epilink,
    MolecularClock,
    genetic_linkage_probability,
    linkage_probability,
    linkage_probability_matrix,
    temporal_linkage_probability,
)


def test_epilink_run_simulations_shapes():
    toit = TOIT(rng_seed=11)
    clock = MolecularClock(relax_rate=False, rng_seed=13)

    sim = Epilink.run_simulations(toit, clock, num_simulations=50, intermediate_hosts=3)

    assert sim.incubation_periods.shape == (50, 2)
    assert sim.generation_interval.shape == (50, 4)
    assert sim.sampling_delay_i.shape == (50,)
    assert sim.sampling_delay_j.shape == (50,)
    assert sim.diff_incubation_ij.shape == (50,)
    assert sim.generation_time_xi.shape == (50,)
    assert sim.diff_infection_ij.shape == (50,)
    assert sim.clock_rates.shape == (50,)
    assert np.all(sim.sampling_delay_i >= 0.0)
    assert np.all(sim.sampling_delay_j >= 0.0)
    assert np.all(sim.diff_infection_ij >= 0.0)
    assert np.all(sim.clock_rates > 0.0)


def test_temporal_kernel_exact():
    temporal_distance = np.array([0.0, 1.0, 2.0])
    diff_incubation = np.array([0.0, 1.0])
    generation_interval = np.array([1.0, 2.0])

    out = Epilink.temporal_kernel(temporal_distance, diff_incubation, generation_interval)
    np.testing.assert_allclose(out, np.array([1.0, 1.0, 0.0]), rtol=0.0, atol=0.0)


def test_temporal_kernel_pyfunc_exact():
    temporal_distance = np.array([0.0, 1.0, 2.0])
    diff_incubation = np.array([0.0, 1.0])
    generation_interval = np.array([1.0, 2.0])

    out = Epilink.temporal_kernel.py_func(temporal_distance, diff_incubation, generation_interval)
    np.testing.assert_allclose(out, np.array([1.0, 1.0, 0.0]), rtol=0.0, atol=0.0)


def test_genetic_kernel_zero_distance_and_negative():
    genetic_distance = np.array([0.0, -1.0])
    clock_rates = np.array([1.0])
    sampling_delay_i = np.array([0.0])
    sampling_delay_j = np.array([0.0])
    intermediate_generations = np.zeros((1, 3))
    diff_infection_ij = np.array([0.0])
    incubation_periods = np.zeros((1, 2))
    generation_time_xi = np.array([0.0])

    out = Epilink.genetic_kernel(
        genetic_distance_ij=genetic_distance,
        clock_rates=clock_rates,
        sampling_delay_i=sampling_delay_i,
        sampling_delay_j=sampling_delay_j,
        intermediate_generations=intermediate_generations,
        intermediates=2,
        diff_infection_ij=diff_infection_ij,
        incubation_periods=incubation_periods,
        generation_time_xi=generation_time_xi,
    )

    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[0], np.ones(3), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(out[1], np.zeros(3), rtol=0.0, atol=0.0)


def test_genetic_kernel_pyfunc_simple_case():
    genetic_distance = np.array([0.0])
    clock_rates = np.array([1.0])
    sampling_delay_i = np.array([0.0])
    sampling_delay_j = np.array([0.0])
    intermediate_generations = np.zeros((1, 2))
    diff_infection_ij = np.array([0.0])
    incubation_periods = np.zeros((1, 2))
    generation_time_xi = np.array([0.0])

    out = Epilink.genetic_kernel.py_func(
        genetic_distance_ij=genetic_distance,
        clock_rates=clock_rates,
        sampling_delay_i=sampling_delay_i,
        sampling_delay_j=sampling_delay_j,
        intermediate_generations=intermediate_generations,
        intermediates=1,
        diff_infection_ij=diff_infection_ij,
        incubation_periods=incubation_periods,
        generation_time_xi=generation_time_xi,
    )

    np.testing.assert_allclose(out, np.array([[1.0, 1.0]]), rtol=0.0, atol=0.0)


def test_genetic_kernel_with_intermediates():
    """Test genetic kernel with multiple intermediate hosts."""
    genetic_distance = np.array([0.0, 5.0, 10.0])
    clock_rates = np.array([1e-3, 2e-3])
    sampling_delay_i = np.array([1.0, 1.5])
    sampling_delay_j = np.array([1.0, 1.5])
    intermediate_generations = np.array([[2.0, 3.0, 4.0], [2.5, 3.5, 4.5]])
    diff_infection_ij = np.array([1.0, 1.2])
    incubation_periods = np.array([[5.0, 5.0], [5.5, 5.5]])
    generation_time_xi = np.array([3.0, 3.5])

    out = Epilink.genetic_kernel(
        genetic_distance_ij=genetic_distance,
        clock_rates=clock_rates,
        sampling_delay_i=sampling_delay_i,
        sampling_delay_j=sampling_delay_j,
        intermediate_generations=intermediate_generations,
        intermediates=2,
        diff_infection_ij=diff_infection_ij,
        incubation_periods=incubation_periods,
        generation_time_xi=generation_time_xi,
    )

    assert out.shape == (3, 3)
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_linkage_probability_scalar_and_array():
    toit = TOIT(rng_seed=21)
    clock = MolecularClock(relax_rate=False, rng_seed=22)

    scalar = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=0,
        temporal_distance=0,
        num_simulations=50,
        intermediate_hosts=2,
    )
    assert isinstance(scalar, float)
    assert 0.0 <= scalar <= 1.0

    arr = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=np.array([0, 1, 2]),
        temporal_distance=np.array([0, 1, 2]),
        num_simulations=50,
        intermediate_hosts=2,
        cache_unique_distances=False,
    )
    assert arr.shape == (3,)
    assert np.all((arr >= 0.0) & (arr <= 1.0))

    with pytest.raises(ValueError, match="same length"):
        linkage_probability(
            toit=toit,
            clock=clock,
            genetic_distance=[0, 1],
            temporal_distance=[1],
            num_simulations=10,
            intermediate_hosts=2,
            cache_unique_distances=False,
        )

    empty = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=[],
        temporal_distance=[],
        num_simulations=10,
        intermediate_hosts=2,
        cache_unique_distances=False,
    )
    assert np.isnan(empty)


def test_pairwise_linkage_probability_matrix_singleton():
    toit = TOIT(rng_seed=31)
    clock = MolecularClock(relax_rate=False, rng_seed=32)

    mat = linkage_probability_matrix(
        toit=toit,
        clock=clock,
        genetic_distances=np.array([0]),
        temporal_distances=np.array([0]),
        num_simulations=50,
        intermediate_hosts=2,
    )

    assert mat.shape == (1, 1)
    assert 0.0 <= mat[0, 0] <= 1.0


@pytest.mark.parametrize(
    "temporal_dist, seed",
    [
        (np.array([0, 1, 2]), 41),
        (np.array([0, 2, 5, 10, 20]), 140),
        (np.arange(0, 30, 2), 2050),
        (np.arange(0, 100), 6010),
    ],
)
def test_temporal_linkage_probability_arrays(temporal_dist, seed):
    toit = TOIT(rng_seed=seed)
    out = temporal_linkage_probability(temporal_dist, toit=toit, num_simulations=50)
    assert out.shape == (temporal_dist.size,)
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_genetic_linkage_probability_kinds_and_errors():
    toit = TOIT(rng_seed=51)
    clock = MolecularClock(relax_rate=False, rng_seed=52)

    out_raw = genetic_linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=[0, 1],
        num_simulations=50,
        intermediate_hosts=2,
        intermediate_generations=(0, 1),
        kind="raw",
    )
    assert out_raw.shape == (2,)
    assert np.all((out_raw >= 0.0) & (out_raw <= 1.0))

    out_relative = genetic_linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=[0, 1],
        num_simulations=50,
        intermediate_hosts=2,
        intermediate_generations=(0, 1),
        kind="relative",
    )
    assert out_relative.shape == (2,)

    out_normalized = genetic_linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=[0, 1],
        num_simulations=50,
        intermediate_hosts=2,
        intermediate_generations=(0, 1),
        kind="normalized",
    )
    assert out_normalized.shape == (2,)

    out_all = genetic_linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=[0, 1],
        num_simulations=50,
        intermediate_hosts=2,
        intermediate_generations=None,
        kind="raw",
    )
    assert out_all.shape == (2, 3)

    with pytest.raises(ValueError, match="intermediate_generations"):
        genetic_linkage_probability(
            toit=toit,
            clock=clock,
            genetic_distance=[0],
            num_simulations=10,
            intermediate_hosts=2,
            intermediate_generations=(3,),
            kind="relative",
        )

    with pytest.raises(ValueError, match="kind must be"):
        genetic_linkage_probability(
            toit=toit,
            clock=clock,
            genetic_distance=[0],
            num_simulations=10,
            intermediate_hosts=2,
            intermediate_generations=(0,),
            kind="unknown",
        )


def test_temporal_kernel_edge_cases():
    """Test temporal kernel with various edge cases."""
    # Test with larger temporal distances that exceed generation interval
    temporal_distance = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    diff_incubation = np.array([0.0, 1.0, -1.0, 2.0])
    generation_interval = np.array([3.0, 4.0, 5.0, 3.5])

    out = Epilink.temporal_kernel(temporal_distance, diff_incubation, generation_interval)

    assert out.shape == (5,)
    assert np.all((out >= 0.0) & (out <= 1.0))
    # First element (t=0) should have high probability
    assert out[0] > 0.0
    # Very large distances should have lower probabilities
    assert out[-1] <= out[0]


def test_temporal_kernel_single_simulation():
    """Test temporal kernel with single MC simulation."""
    temporal_distance = np.array([2.0])
    diff_incubation = np.array([1.0])
    generation_interval = np.array([5.0])

    out = Epilink.temporal_kernel(temporal_distance, diff_incubation, generation_interval)

    # abs(2.0 + 1.0) = 3.0 <= 5.0, so should be 1.0
    assert out[0] == 1.0


def test_genetic_kernel_multiple_intermediates():
    """Test genetic kernel with multiple intermediate hosts and various scenarios."""
    genetic_distance = np.array([0.0, 5.0, 10.0, 15.0])
    clock_rates = np.array([1e-3, 2e-3, 1.5e-3])
    sampling_delay_i = np.array([2.0, 3.0, 2.5])
    sampling_delay_j = np.array([2.0, 3.0, 2.5])
    intermediate_generations = np.array([[3.0, 4.0, 5.0], [3.5, 4.5, 5.5], [3.2, 4.2, 5.2]])
    diff_infection_ij = np.array([1.0, 1.5, 1.2])
    incubation_periods = np.array([[5.0, 5.0], [6.0, 6.0], [5.5, 5.5]])
    generation_time_xi = np.array([4.0, 4.5, 4.2])

    out = Epilink.genetic_kernel(
        genetic_distance_ij=genetic_distance,
        clock_rates=clock_rates,
        sampling_delay_i=sampling_delay_i,
        sampling_delay_j=sampling_delay_j,
        intermediate_generations=intermediate_generations,
        intermediates=2,
        diff_infection_ij=diff_infection_ij,
        incubation_periods=incubation_periods,
        generation_time_xi=generation_time_xi,
    )

    # Shape should be (num_distances, num_intermediates + 1)
    assert out.shape == (4, 3)
    # All probabilities should be valid
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_genetic_kernel_m_greater_than_zero():
    """Test genetic kernel scenarios with m > 0 (intermediate hosts)."""
    genetic_distance = np.array([10.0])
    clock_rates = np.array([1e-3, 2e-3])
    sampling_delay_i = np.array([2.0, 2.0])
    sampling_delay_j = np.array([2.0, 2.0])
    # Multiple columns for different intermediate scenarios
    intermediate_generations = np.array([[3.0, 4.0, 5.0, 6.0], [3.5, 4.5, 5.5, 6.5]])
    diff_infection_ij = np.array([2.0, 2.5])
    incubation_periods = np.array([[5.0, 5.0], [5.5, 5.5]])
    generation_time_xi = np.array([4.0, 4.5])

    out = Epilink.genetic_kernel(
        genetic_distance_ij=genetic_distance,
        clock_rates=clock_rates,
        sampling_delay_i=sampling_delay_i,
        sampling_delay_j=sampling_delay_j,
        intermediate_generations=intermediate_generations,
        intermediates=3,  # M=3, so we test m=1, 2, 3
        diff_infection_ij=diff_infection_ij,
        incubation_periods=incubation_periods,
        generation_time_xi=generation_time_xi,
    )

    # Shape: (1 distance, 4 scenarios: m=0,1,2,3)
    assert out.shape == (1, 4)
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_linkage_probability_various_intermediate_generations():
    """Test linkage_probability with different intermediate_generations tuples."""
    toit = TOIT(rng_seed=100)
    clock = MolecularClock(relax_rate=False, rng_seed=101)

    # Test with only direct transmission (m=0)
    result_direct = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=5,
        temporal_distance=3,
        intermediate_generations=(0,),
        intermediate_hosts=3,
        num_simulations=50,
        cache_unique_distances=False,
    )
    assert isinstance(result_direct, float)
    assert 0.0 <= result_direct <= 1.0

    # Test with multiple intermediate scenarios
    result_multi = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=5,
        temporal_distance=3,
        intermediate_generations=(0, 1, 2),
        intermediate_hosts=3,
        num_simulations=50,
        cache_unique_distances=False,
    )
    assert isinstance(result_multi, float)
    assert 0.0 <= result_multi <= 1.0


def test_temporal_linkage_probability_scalar_input():
    """Test temporal_linkage_probability with scalar input."""
    toit = TOIT(rng_seed=130)

    # Scalar input
    result_scalar = temporal_linkage_probability(
        temporal_distance=5.0,
        toit=toit,
        num_simulations=50,
    )

    assert result_scalar.shape == (1,)
    assert 0.0 <= result_scalar[0] <= 1.0


def test_genetic_linkage_probability_all_kinds_with_none():
    """Test genetic_linkage_probability with intermediate_generations=None for all kinds."""
    toit = TOIT(rng_seed=150)
    clock = MolecularClock(relax_rate=False, rng_seed=151)

    genetic_dist = np.array([0, 5, 10])

    # Test 'raw' with None
    result_raw = genetic_linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=genetic_dist,
        num_simulations=50,
        intermediate_hosts=2,
        intermediate_generations=None,
        kind="raw",
    )
    assert result_raw.shape == (3, 3)  # 3 distances, 3 scenarios (m=0,1,2)
    assert np.all((result_raw >= 0.0) & (result_raw <= 1.0))

    # Test 'relative' with None
    result_relative = genetic_linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=genetic_dist,
        num_simulations=50,
        intermediate_hosts=2,
        intermediate_generations=None,
        kind="relative",
    )
    assert result_relative.shape == (3, 3)
    assert np.all((result_relative >= 0.0) & (result_relative <= 1.0))

    # Test 'normalized' with None
    result_normalized = genetic_linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=genetic_dist,
        num_simulations=50,
        intermediate_hosts=2,
        intermediate_generations=None,
        kind="normalized",
    )
    assert result_normalized.shape == (3, 3)
    assert np.all((result_normalized >= 0.0) & (result_normalized <= 1.0))
    # Each row should sum to ~1.0 for normalized (or 0.0 if all zeros)
    row_sums = result_normalized.sum(axis=1)
    # Check that each row either sums to ~1.0 or is all zeros
    for row_sum in row_sums:
        assert np.abs(row_sum - 1.0) < 0.01 or np.abs(row_sum) < 1e-10


def test_linkage_probability_single_element_no_cache():
    """Test linkage_probability with single element and cache disabled."""
    toit = TOIT(rng_seed=170)
    clock = MolecularClock(relax_rate=False, rng_seed=171)

    # Single element array with cache disabled
    result = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=np.array([5]),
        temporal_distance=np.array([3]),
        num_simulations=30,
        intermediate_hosts=2,
        cache_unique_distances=False,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert 0.0 <= result[0] <= 1.0


def test_epilink_run_simulations_reproducibility():
    """Test that Epilink.run_simulations produces reproducible results with same seed."""
    toit1 = TOIT(rng_seed=200)
    clock1 = MolecularClock(relax_rate=False, rng_seed=201)

    toit2 = TOIT(rng_seed=200)
    clock2 = MolecularClock(relax_rate=False, rng_seed=201)

    sim1 = Epilink.run_simulations(toit1, clock1, num_simulations=10, intermediate_hosts=2)
    sim2 = Epilink.run_simulations(toit2, clock2, num_simulations=10, intermediate_hosts=2)

    np.testing.assert_array_equal(sim1.incubation_periods, sim2.incubation_periods)
    np.testing.assert_array_equal(sim1.generation_interval, sim2.generation_interval)
    np.testing.assert_array_equal(sim1.clock_rates, sim2.clock_rates)


def test_linkage_probability_with_larger_no_intermediates():
    """Test linkage_probability with larger number of intermediate hosts."""
    toit = TOIT(rng_seed=210)
    clock = MolecularClock(relax_rate=False, rng_seed=211)

    result = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=10,
        temporal_distance=5,
        intermediate_generations=(0, 1, 2, 3, 4, 5),
        intermediate_hosts=10,
        num_simulations=50,
        cache_unique_distances=False,
    )

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_genetic_linkage_probability_scalar_distance():
    """Test genetic_linkage_probability with scalar distance input."""
    toit = TOIT(rng_seed=220)
    clock = MolecularClock(relax_rate=False, rng_seed=221)

    # Scalar input should be converted to array
    result = genetic_linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=5,  # scalar
        num_simulations=30,
        intermediate_hosts=2,
        intermediate_generations=(0, 1),
        kind="relative",
    )

    assert result.shape == (1,)
    assert 0.0 <= result[0] <= 1.0


def test_linkage_probability_no_cache_direct_path():
    """Test linkage_probability direct computation path (no caching)."""
    toit = TOIT(rng_seed=230)
    clock = MolecularClock(relax_rate=False, rng_seed=231)

    # Test with cache disabled (default behavior for non-duplicate data)
    result = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=np.array([0, 5, 10]),
        temporal_distance=np.array([0, 2, 5]),
        intermediate_generations=(0,),
        intermediate_hosts=2,
        num_simulations=50,
        cache_unique_distances=False,
    )

    assert result.shape == (3,)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_linkage_probability_multiple_intermediate_generations():
    """Test linkage_probability with multiple intermediate generation scenarios."""
    toit = TOIT(rng_seed=240)
    clock = MolecularClock(relax_rate=False, rng_seed=241)

    # Test with m=0, 1, 2
    result = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=np.array([5, 10, 15]),
        temporal_distance=np.array([3, 5, 7]),
        intermediate_generations=(0, 1, 2),
        intermediate_hosts=5,
        num_simulations=50,
        cache_unique_distances=False,
    )

    assert result.shape == (3,)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_linkage_probability_high_genetic_distance():
    """Test linkage_probability with high genetic distances."""
    toit = TOIT(rng_seed=250)
    clock = MolecularClock(relax_rate=False, rng_seed=251)

    # High genetic distances should typically give lower probabilities
    result = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=np.array([0, 50, 100, 200]),
        temporal_distance=np.array([0, 10, 20, 30]),
        intermediate_generations=(0, 1),
        intermediate_hosts=3,
        num_simulations=50,
        cache_unique_distances=False,
    )

    assert result.shape == (4,)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_linkage_probability_high_temporal_distance():
    """Test linkage_probability with high temporal distances."""
    toit = TOIT(rng_seed=260)
    clock = MolecularClock(relax_rate=False, rng_seed=261)

    # High temporal distances should typically give lower probabilities
    result = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=np.array([5, 5, 5, 5]),
        temporal_distance=np.array([0, 20, 50, 100]),
        intermediate_generations=(0,),
        intermediate_hosts=2,
        num_simulations=50,
        cache_unique_distances=False,
    )

    assert result.shape == (4,)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_linkage_probability_matrix_single_distance():
    """Test linkage_probability_matrix with single genetic and temporal distance."""
    toit = TOIT(rng_seed=270)
    clock = MolecularClock(relax_rate=False, rng_seed=271)

    result = linkage_probability_matrix(
        toit=toit,
        clock=clock,
        genetic_distances=np.array([5]),
        temporal_distances=np.array([3]),
        intermediate_generations=(0,),
        intermediate_hosts=2,
        num_simulations=50,
    )

    assert result.shape == (1, 1)
    assert 0.0 <= result[0, 0] <= 1.0


def test_linkage_probability_zero_distances():
    """Test linkage_probability with zero genetic and temporal distances."""
    toit = TOIT(rng_seed=310)
    clock = MolecularClock(relax_rate=False, rng_seed=311)

    # Zero distances should give high probability (same individual or very close)
    result = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=0,
        temporal_distance=0,
        intermediate_generations=(0,),
        intermediate_hosts=2,
        num_simulations=100,
        cache_unique_distances=False,
    )

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
    # Should be relatively high probability
    assert result > 0.0


def test_linkage_probability_mixed_distances():
    """Test linkage_probability with mixed high/low distances."""
    toit = TOIT(rng_seed=320)
    clock = MolecularClock(relax_rate=False, rng_seed=321)

    # Mix of close and far pairs
    result = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=np.array([0, 100, 5, 50]),
        temporal_distance=np.array([0, 50, 2, 10]),
        intermediate_generations=(0, 1),
        intermediate_hosts=3,
        num_simulations=50,
        cache_unique_distances=False,
    )

    assert result.shape == (4,)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_linkage_probability_all_intermediate_scenarios():
    """Test linkage_probability selecting all possible intermediate scenarios."""
    toit = TOIT(rng_seed=330)
    clock = MolecularClock(relax_rate=False, rng_seed=331)

    # Select all scenarios from m=0 to m=5
    result = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=np.array([10]),
        temporal_distance=np.array([5]),
        intermediate_generations=(0, 1, 2, 3, 4, 5),
        intermediate_hosts=5,
        num_simulations=50,
        cache_unique_distances=False,
    )

    assert result.shape == (1,)
    assert 0.0 <= result[0] <= 1.0


def test_linkage_probability_single_intermediate_scenario():
    """Test linkage_probability with only direct transmission (m=0)."""
    toit = TOIT(rng_seed=340)
    clock = MolecularClock(relax_rate=False, rng_seed=341)

    result_m0 = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=5,
        temporal_distance=3,
        intermediate_generations=(0,),  # Only direct
        intermediate_hosts=3,
        num_simulations=50,
        cache_unique_distances=False,
    )

    assert isinstance(result_m0, float)
    assert 0.0 <= result_m0 <= 1.0


def test_linkage_probability_consistency():
    """Test that linkage_probability gives consistent results for same inputs."""
    toit1 = TOIT(rng_seed=360)
    clock1 = MolecularClock(relax_rate=False, rng_seed=361)

    toit2 = TOIT(rng_seed=360)  # Same seed
    clock2 = MolecularClock(relax_rate=False, rng_seed=361)

    result1 = linkage_probability(
        toit=toit1,
        clock=clock1,
        genetic_distance=10,
        temporal_distance=5,
        intermediate_generations=(0, 1),
        intermediate_hosts=2,
        num_simulations=50,
        cache_unique_distances=False,
    )

    result2 = linkage_probability(
        toit=toit2,
        clock=clock2,
        genetic_distance=10,
        temporal_distance=5,
        intermediate_generations=(0, 1),
        intermediate_hosts=2,
        num_simulations=50,
        cache_unique_distances=False,
    )

    assert result1 == result2


def test_linkage_probability_with_different_num_simulations():
    """Test linkage_probability with different numbers of simulations."""
    toit = TOIT(rng_seed=370)
    clock = MolecularClock(relax_rate=False, rng_seed=371)

    # Fewer simulations
    result_few = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=10,
        temporal_distance=5,
        intermediate_generations=(0,),
        intermediate_hosts=2,
        num_simulations=20,
        cache_unique_distances=False,
    )

    # More simulations (reset RNG for comparison)
    toit2 = TOIT(rng_seed=370)
    clock2 = MolecularClock(relax_rate=False, rng_seed=371)

    result_many = linkage_probability(
        toit=toit2,
        clock=clock2,
        genetic_distance=10,
        temporal_distance=5,
        intermediate_generations=(0,),
        intermediate_hosts=2,
        num_simulations=100,
        cache_unique_distances=False,
    )

    # Both should be valid probabilities
    assert isinstance(result_few, float)
    assert isinstance(result_many, float)
    assert 0.0 <= result_few <= 1.0
    assert 0.0 <= result_many <= 1.0


class TestCachingPath:
    """Test the caching path in linkage_probability with duplicate distance pairs."""

    @pytest.mark.parametrize(
        "genetic_dist, temporal_dist, intermediate_generations, intermediate_hosts",
        [
            (np.array([5, 10, 5, 15, 10, 5]), np.array([3, 7, 3, 9, 7, 3]), (0, 1), 3),
            (np.array([5, 10, 5, 15]), np.array([3, 7, 3, 9]), (0, 1, 2, 3), 5),
            (np.array([5, 10, 5]), np.array([3, 7, 3]), (0, 5), 5),
            (
                np.array([5, 10, 5]),
                np.array([3, 7, 3]),
                (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                10,
            ),
        ],
    )
    def test_linkage_probability_cache_with_duplicates(
        self,
        genetic_dist,
        temporal_dist,
        intermediate_generations,
        intermediate_hosts,
    ):
        """Test caching path when there are duplicate distance pairs."""
        toit = TOIT(rng_seed=1000)
        clock = MolecularClock(relax_rate=False, rng_seed=1001)

        result = linkage_probability(
            toit=toit,
            clock=clock,
            genetic_distance=genetic_dist,
            temporal_distance=temporal_dist,
            intermediate_generations=intermediate_generations,
            intermediate_hosts=intermediate_hosts,
            num_simulations=50,
            cache_unique_distances=True,
        )

        assert result.shape == (genetic_dist.size,)
        assert np.all((result >= 0.0) & (result <= 1.0))
        pairs = list(zip(genetic_dist, temporal_dist))
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                if pairs[i] == pairs[j]:
                    assert result[i] == result[j]

    def test_linkage_probability_all_same_pairs_cache_enabled(self):
        """Test caching when all pairs are identical."""
        toit = TOIT(rng_seed=1010)
        clock = MolecularClock(relax_rate=False, rng_seed=1011)

        # All identical pairs
        genetic_dist = np.array([10, 10, 10, 10])
        temporal_dist = np.array([5, 5, 5, 5])

        result = linkage_probability(
            toit=toit,
            clock=clock,
            genetic_distance=genetic_dist,
            temporal_distance=temporal_dist,
            intermediate_generations=(0,),
            intermediate_hosts=2,
            num_simulations=50,
            cache_unique_distances=True,
        )

        assert result.shape == (4,)
        # All results should be identical
        np.testing.assert_array_equal(result, np.full(4, result[0]))

    def test_linkage_probability_cache_with_validation_error(self):
        """Test that validation errors are raised even with caching enabled."""
        toit = TOIT(rng_seed=1020)
        clock = MolecularClock(relax_rate=False, rng_seed=1021)

        genetic_dist = np.array([5, 10, 5])
        temporal_dist = np.array([3, 7, 3])

        # Test with invalid intermediate_generations in caching path
        with pytest.raises(ValueError, match="intermediate_generations must be within"):
            linkage_probability(
                toit=toit,
                clock=clock,
                genetic_distance=genetic_dist,
                temporal_distance=temporal_dist,
                intermediate_generations=(10,),  # Exceeds intermediate_hosts=3
                intermediate_hosts=3,
                num_simulations=50,
                cache_unique_distances=True,
            )

        # Test with negative intermediate_generations in caching path
        with pytest.raises(ValueError, match="intermediate_generations must be within"):
            linkage_probability(
                toit=toit,
                clock=clock,
                genetic_distance=genetic_dist,
                temporal_distance=temporal_dist,
                intermediate_generations=(-1, 0),
                intermediate_hosts=3,
                num_simulations=50,
                cache_unique_distances=True,
            )


class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions."""

    def test_linkage_probability_single_element_with_cache(self):
        """Test that single element doesn't use cache path (size > 1 requirement)."""
        toit = TOIT(rng_seed=2000)
        clock = MolecularClock(relax_rate=False, rng_seed=2001)

        # Single element - should skip caching even if enabled
        result = linkage_probability(
            toit=toit,
            clock=clock,
            genetic_distance=np.array([5]),
            temporal_distance=np.array([3]),
            intermediate_generations=(0,),
            intermediate_hosts=2,
            num_simulations=50,
            cache_unique_distances=True,  # Enabled but size=1 so skips caching
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_linkage_probability_two_elements_unique_cache(self):
        """Test caching with exactly 2 unique elements."""
        toit = TOIT(rng_seed=2010)
        clock = MolecularClock(relax_rate=False, rng_seed=2011)

        genetic_dist = np.array([5, 10])
        temporal_dist = np.array([3, 7])

        result = linkage_probability(
            toit=toit,
            clock=clock,
            genetic_distance=genetic_dist,
            temporal_distance=temporal_dist,
            intermediate_generations=(0, 1),
            intermediate_hosts=2,
            num_simulations=50,
            cache_unique_distances=True,
        )

        assert result.shape == (2,)
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_linkage_probability_cache_min_intermediate_zero(self):
        """Test caching path with intermediate_generations starting at 0."""
        toit = TOIT(rng_seed=2020)
        clock = MolecularClock(relax_rate=False, rng_seed=2021)

        genetic_dist = np.array([5, 10, 5])
        temporal_dist = np.array([3, 7, 3])

        result = linkage_probability(
            toit=toit,
            clock=clock,
            genetic_distance=genetic_dist,
            temporal_distance=temporal_dist,
            intermediate_generations=(0,),  # Only minimum
            intermediate_hosts=5,
            num_simulations=50,
            cache_unique_distances=True,
        )

        assert result.shape == (3,)
        assert result[0] == result[2]

    def test_genetic_linkage_probability_boundary_values(self):
        """Test genetic_linkage_probability with boundary intermediate_generations."""
        toit = TOIT(rng_seed=2030)
        clock = MolecularClock(relax_rate=False, rng_seed=2031)

        # Test at exact boundaries
        result_min = genetic_linkage_probability(
            toit=toit,
            clock=clock,
            genetic_distance=[5],
            num_simulations=30,
            intermediate_hosts=10,
            intermediate_generations=(0,),  # Minimum
            kind="normalized",
        )
        assert result_min.shape == (1,)

        toit2 = TOIT(rng_seed=2030)
        clock2 = MolecularClock(relax_rate=False, rng_seed=2031)

        result_max = genetic_linkage_probability(
            toit=toit2,
            clock=clock2,
            genetic_distance=[5],
            num_simulations=30,
            intermediate_hosts=10,
            intermediate_generations=(10,),  # Maximum
            kind="normalized",
        )
        assert result_max.shape == (1,)

    def test_linkage_probability_matrix_multiple_distances(self):
        """Test linkage_probability_matrix with multiple genetic and temporal distances."""
        toit = TOIT(rng_seed=2040)
        clock = MolecularClock(relax_rate=False, rng_seed=2041)

        genetic_distances = np.array([0, 5, 10, 15, 20])
        temporal_distances = np.array([0, 2, 5, 10, 15])

        result = linkage_probability_matrix(
            toit=toit,
            clock=clock,
            genetic_distances=genetic_distances,
            temporal_distances=temporal_distances,
            intermediate_generations=(0, 1, 2),
            intermediate_hosts=5,
            num_simulations=50,
        )

        assert result.shape == (5, 5)
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_genetic_linkage_probability_large_distance_array(self):
        """Test genetic_linkage_probability with larger distance array."""
        toit = TOIT(rng_seed=2060)
        clock = MolecularClock(relax_rate=False, rng_seed=2061)

        genetic_dist = np.arange(0, 50, 5)  # [0, 5, 10, ..., 45]

        result = genetic_linkage_probability(
            toit=toit,
            clock=clock,
            genetic_distance=genetic_dist,
            num_simulations=50,
            intermediate_hosts=3,
            intermediate_generations=(0, 1, 2),
            kind="normalized",
        )

        assert result.shape == (10,)
        assert np.all((result >= 0.0) & (result <= 1.0))


class TestZeroProbabilityEdgeCases:
    """Test cases that should produce zero or near-zero probabilities."""

    def test_linkage_probability_extreme_distances(self):
        """Test with extremely large distances that should give very low probabilities."""
        toit = TOIT(rng_seed=3000)
        clock = MolecularClock(relax_rate=False, rng_seed=3001)

        # Very large distances
        result = linkage_probability(
            toit=toit,
            clock=clock,
            genetic_distance=1000,  # Very large
            temporal_distance=365,  # One year
            intermediate_generations=(0,),
            intermediate_hosts=2,
            num_simulations=100,
            cache_unique_distances=False,
        )

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_linkage_probability_cache_extreme_distances(self):
        """Test caching path with extreme distances."""
        toit = TOIT(rng_seed=3010)
        clock = MolecularClock(relax_rate=False, rng_seed=3011)

        genetic_dist = np.array([1000, 500, 1000])
        temporal_dist = np.array([365, 180, 365])

        result = linkage_probability(
            toit=toit,
            clock=clock,
            genetic_distance=genetic_dist,
            temporal_distance=temporal_dist,
            intermediate_generations=(0, 1),
            intermediate_hosts=3,
            num_simulations=50,
            cache_unique_distances=True,
        )

        assert result.shape == (3,)
        assert result[0] == result[2]  # Duplicates should match


class TestComplexScenarios:
    """Test complex scenarios combining multiple features."""

    def test_genetic_linkage_probability_all_kinds_with_selection(self):
        """Test all kind values with specific intermediate_generations."""
        toit = TOIT(rng_seed=4020)
        clock = MolecularClock(relax_rate=False, rng_seed=4021)

        genetic_dist = np.array([5, 10, 15])

        # Test each kind with selected intermediates
        for kind in ["raw", "relative", "normalized"]:
            result = genetic_linkage_probability(
                toit=toit,
                clock=clock,
                genetic_distance=genetic_dist,
                num_simulations=50,
                intermediate_hosts=3,
                intermediate_generations=(0, 1, 2),
                kind=kind,
            )
            assert result.shape == (3,)
            assert np.all((result >= 0.0) & (result <= 1.0))

    def test_linkage_probability_matrix_with_zeros(self):
        """Test matrix computation with zero distances included."""
        toit = TOIT(rng_seed=4030)
        clock = MolecularClock(relax_rate=False, rng_seed=4031)

        genetic_distances = np.array([0, 1, 2, 3])
        temporal_distances = np.array([0, 1, 2, 3])

        result = linkage_probability_matrix(
            toit=toit,
            clock=clock,
            genetic_distances=genetic_distances,
            temporal_distances=temporal_distances,
            intermediate_generations=(0,),
            intermediate_hosts=2,
            num_simulations=50,
        )

        assert result.shape == (4, 4)
        assert np.all((result >= 0.0) & (result <= 1.0))
        # (0, 0) should have high probability
        assert result[0, 0] > 0.0


class TestIntermediateGenerationsSelection:
    """Test various intermediate_generations selections."""

    def test_linkage_probability_single_high_intermediate(self):
        """Test with only a high intermediate value selected."""
        toit = TOIT(rng_seed=5000)
        clock = MolecularClock(relax_rate=False, rng_seed=5001)

        result = linkage_probability(
            toit=toit,
            clock=clock,
            genetic_distance=10,
            temporal_distance=5,
            intermediate_generations=(5,),  # Only high value
            intermediate_hosts=5,
            num_simulations=50,
            cache_unique_distances=False,
        )

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_linkage_probability_cache_single_high_intermediate(self):
        """Test caching path with only high intermediate value."""
        toit = TOIT(rng_seed=5010)
        clock = MolecularClock(relax_rate=False, rng_seed=5011)

        genetic_dist = np.array([10, 15, 10])
        temporal_dist = np.array([5, 8, 5])

        result = linkage_probability(
            toit=toit,
            clock=clock,
            genetic_distance=genetic_dist,
            temporal_distance=temporal_dist,
            intermediate_generations=(4,),
            intermediate_hosts=5,
            num_simulations=50,
            cache_unique_distances=True,
        )

        assert result.shape == (3,)
        assert result[0] == result[2]

    def test_linkage_probability_non_consecutive_intermediates(self):
        """Test with non-consecutive intermediate values."""
        toit = TOIT(rng_seed=5020)
        clock = MolecularClock(relax_rate=False, rng_seed=5021)

        # Select non-consecutive values: 0, 2, 4
        result = linkage_probability(
            toit=toit,
            clock=clock,
            genetic_distance=10,
            temporal_distance=5,
            intermediate_generations=(0, 2, 4),
            intermediate_hosts=5,
            num_simulations=50,
            cache_unique_distances=False,
        )

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_linkage_probability_cache_non_consecutive_intermediates(self):
        """Test caching with non-consecutive intermediate values."""
        toit = TOIT(rng_seed=5030)
        clock = MolecularClock(relax_rate=False, rng_seed=5031)

        genetic_dist = np.array([10, 15, 10])
        temporal_dist = np.array([5, 8, 5])

        result = linkage_probability(
            toit=toit,
            clock=clock,
            genetic_distance=genetic_dist,
            temporal_distance=temporal_dist,
            intermediate_generations=(0, 3, 5),
            intermediate_hosts=5,
            num_simulations=50,
            cache_unique_distances=True,
        )

        assert result.shape == (3,)
        assert result[0] == result[2]


class TestVeryLargeArrays:
    """Test with larger arrays to ensure scalability."""

    def test_linkage_probability_large_array_with_cache(self):
        """Test with larger arrays and caching enabled."""
        toit = TOIT(rng_seed=6000)
        clock = MolecularClock(relax_rate=False, rng_seed=6001)

        # Create large array with many duplicates
        np.random.seed(6000)
        genetic_dist = np.random.choice([5, 10, 15, 20], size=50)
        temporal_dist = np.random.choice([3, 7, 10, 15], size=50)

        result = linkage_probability(
            toit=toit,
            clock=clock,
            genetic_distance=genetic_dist,
            temporal_distance=temporal_dist,
            intermediate_generations=(0, 1),
            intermediate_hosts=3,
            num_simulations=50,
            cache_unique_distances=True,
        )

        assert result.shape == (50,)
        assert np.all((result >= 0.0) & (result <= 1.0))


class TestErrorConditions:
    """Test error conditions and validation paths."""

    def test_linkage_probability_mismatched_sizes_no_cache(self):
        """Test ValueError for mismatched sizes without caching."""
        toit = TOIT(rng_seed=7000)
        clock = MolecularClock(relax_rate=False, rng_seed=7001)

        with pytest.raises(ValueError, match="must have the same length"):
            linkage_probability(
                toit=toit,
                clock=clock,
                genetic_distance=np.array([5, 10]),
                temporal_distance=np.array([3]),  # Different size
                intermediate_generations=(0,),
                intermediate_hosts=2,
                num_simulations=50,
                cache_unique_distances=False,
            )

    def test_linkage_probability_empty_input_no_cache(self):
        """Test empty input returns np.nan without caching."""
        toit = TOIT(rng_seed=7010)
        clock = MolecularClock(relax_rate=False, rng_seed=7011)

        result = linkage_probability(
            toit=toit,
            clock=clock,
            genetic_distance=np.array([]),
            temporal_distance=np.array([]),
            intermediate_generations=(0,),
            intermediate_hosts=2,
            num_simulations=50,
            cache_unique_distances=False,
        )

        assert np.isnan(result)

    def test_linkage_probability_invalid_intermediate_no_cache(self):
        """Test ValueError for invalid intermediate_generations without cache."""
        toit = TOIT(rng_seed=7020)
        clock = MolecularClock(relax_rate=False, rng_seed=7021)

        # Test exceeding intermediate_hosts
        with pytest.raises(ValueError, match="intermediate_generations must be within"):
            linkage_probability(
                toit=toit,
                clock=clock,
                genetic_distance=np.array([5]),
                temporal_distance=np.array([3]),
                intermediate_generations=(10,),  # Exceeds intermediate_hosts=3
                intermediate_hosts=3,
                num_simulations=50,
                cache_unique_distances=False,
            )

        # Test negative value
        with pytest.raises(ValueError, match="intermediate_generations must be within"):
            linkage_probability(
                toit=toit,
                clock=clock,
                genetic_distance=np.array([5]),
                temporal_distance=np.array([3]),
                intermediate_generations=(-1,),
                intermediate_hosts=3,
                num_simulations=50,
                cache_unique_distances=False,
            )

    def test_genetic_linkage_probability_invalid_intermediate_with_selection(self):
        """Test ValueError for invalid intermediate_generations in genetic_linkage_probability."""
        toit = TOIT(rng_seed=7030)
        clock = MolecularClock(relax_rate=False, rng_seed=7031)

        # Test exceeding intermediate_hosts
        with pytest.raises(ValueError, match="intermediate_generations must be within"):
            genetic_linkage_probability(
                toit=toit,
                clock=clock,
                genetic_distance=[5],
                num_simulations=30,
                intermediate_hosts=3,
                intermediate_generations=(5,),  # Exceeds intermediate_hosts=3
                kind="relative",
            )

        # Test negative value
        with pytest.raises(ValueError, match="intermediate_generations must be within"):
            genetic_linkage_probability(
                toit=toit,
                clock=clock,
                genetic_distance=[5],
                num_simulations=30,
                intermediate_hosts=3,
                intermediate_generations=(-2,),
                kind="relative",
            )

    def test_genetic_linkage_probability_invalid_kind_with_selection(self):
        """Test ValueError for invalid kind with intermediate_generations specified."""
        toit = TOIT(rng_seed=7040)
        clock = MolecularClock(relax_rate=False, rng_seed=7041)

        with pytest.raises(ValueError, match="kind must be"):
            genetic_linkage_probability(
                toit=toit,
                clock=clock,
                genetic_distance=[5],
                num_simulations=30,
                intermediate_hosts=3,
                intermediate_generations=(0, 1),
                kind="invalid",
            )

    def test_genetic_linkage_probability_invalid_kind_without_selection(self):
        """Test ValueError for invalid kind with intermediate_generations=None."""
        toit = TOIT(rng_seed=7050)
        clock = MolecularClock(relax_rate=False, rng_seed=7051)

        with pytest.raises(ValueError, match="kind must be"):
            genetic_linkage_probability(
                toit=toit,
                clock=clock,
                genetic_distance=[5],
                num_simulations=30,
                intermediate_hosts=3,
                intermediate_generations=None,
                kind="wrong_kind",
            )

    def test_genetic_linkage_probability_all_kinds_return_correct_types(self):
        """Test that different kinds return appropriate shapes with None."""
        toit = TOIT(rng_seed=7060)
        clock = MolecularClock(relax_rate=False, rng_seed=7061)

        genetic_dist = [5, 10]

        # Test that each kind with None returns (K, M+1) shape
        for kind in ["raw", "relative", "normalized"]:
            result = genetic_linkage_probability(
                toit=toit,
                clock=clock,
                genetic_distance=genetic_dist,
                num_simulations=30,
                intermediate_hosts=2,
                intermediate_generations=None,
                kind=kind,
            )
            assert result.shape == (2, 3)  # 2 distances, 3 scenarios (m=0,1,2)
