"""Comprehensive tests for transmission linkage inference."""

from __future__ import annotations

import numpy as np
import pytest

from epilink import (
    InfectiousnessToTransmissionTime,
    LinkageMonteCarloSamples,
    MolecularClock,
    estimate_genetic_linkage_probability,
    estimate_linkage_probability,
    estimate_linkage_probability_grid,
    estimate_temporal_linkage_probability,
)


def test_epilink_run_simulations_shapes():
    toit = InfectiousnessToTransmissionTime(rng_seed=11)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=13)

    sim = LinkageMonteCarloSamples.run_simulations(
        toit, clock, num_simulations=50, max_intermediate_hosts=3
    )

    assert sim.incubation_periods.shape == (50, 2)
    assert sim.generation_intervals.shape == (50, 4)
    assert sim.sampling_delay_i.shape == (50,)
    assert sim.sampling_delay_j.shape == (50,)
    assert sim.diff_incubation_ij.shape == (50,)
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

    out = LinkageMonteCarloSamples.temporal_kernel(
        temporal_distance, diff_incubation, generation_interval
    )
    np.testing.assert_allclose(out, np.array([1.0, 1.0, 0.0]), rtol=0.0, atol=0.0)


def test_temporal_kernel_pyfunc_exact():
    temporal_distance = np.array([0.0, 1.0, 2.0])
    diff_incubation = np.array([0.0, 1.0])
    generation_interval = np.array([1.0, 2.0])

    out = LinkageMonteCarloSamples.temporal_kernel.py_func(
        temporal_distance, diff_incubation, generation_interval
    )
    np.testing.assert_allclose(out, np.array([1.0, 1.0, 0.0]), rtol=0.0, atol=0.0)


def test_genetic_kernel_zero_distance_and_negative():
    genetic_distance = np.array([0.0, -1.0])
    clock_rates = np.array([1.0])
    sampling_delay_i = np.array([0.0])
    sampling_delay_j = np.array([0.0])
    included_intermediate_counts = np.zeros((1, 3))
    diff_infection_ij = np.array([0.0])
    incubation_periods = np.zeros((1, 2))

    out = LinkageMonteCarloSamples.genetic_kernel(
        genetic_distance_ij=genetic_distance,
        clock_rates=clock_rates,
        sampling_delay_i=sampling_delay_i,
        sampling_delay_j=sampling_delay_j,
        generation_intervals=included_intermediate_counts,
        max_intermediate_hosts=2,
        diff_infection_ij=diff_infection_ij,
        incubation_periods=incubation_periods,
    )

    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[0], np.full(3, 2.0), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(out[1], np.zeros(3), rtol=0.0, atol=0.0)


def test_genetic_kernel_pyfunc_simple_case():
    genetic_distance = np.array([0.0])
    clock_rates = np.array([1.0])
    sampling_delay_i = np.array([0.0])
    sampling_delay_j = np.array([0.0])
    included_intermediate_counts = np.zeros((1, 2))
    diff_infection_ij = np.array([0.0])
    incubation_periods = np.zeros((1, 2))

    out = LinkageMonteCarloSamples.genetic_kernel.py_func(
        genetic_distance_ij=genetic_distance,
        clock_rates=clock_rates,
        sampling_delay_i=sampling_delay_i,
        sampling_delay_j=sampling_delay_j,
        generation_intervals=included_intermediate_counts,
        max_intermediate_hosts=1,
        diff_infection_ij=diff_infection_ij,
        incubation_periods=incubation_periods,
    )

    np.testing.assert_allclose(out, np.array([[2.0, 2.0]]), rtol=0.0, atol=0.0)


def test_genetic_kernel_with_intermediates():
    """Test genetic kernel with multiple intermediate hosts."""
    genetic_distance = np.array([0.0, 5.0, 10.0])
    clock_rates = np.array([1e-3, 2e-3])
    sampling_delay_i = np.array([1.0, 1.5])
    sampling_delay_j = np.array([1.0, 1.5])
    included_intermediate_counts = np.array([[2.0, 3.0, 4.0], [2.5, 3.5, 4.5]])
    diff_infection_ij = np.array([1.0, 1.2])
    incubation_periods = np.array([[5.0, 5.0], [5.5, 5.5]])

    out = LinkageMonteCarloSamples.genetic_kernel(
        genetic_distance_ij=genetic_distance,
        clock_rates=clock_rates,
        sampling_delay_i=sampling_delay_i,
        sampling_delay_j=sampling_delay_j,
        generation_intervals=included_intermediate_counts,
        max_intermediate_hosts=2,
        diff_infection_ij=diff_infection_ij,
        incubation_periods=incubation_periods,
    )

    assert out.shape == (3, 3)
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 2.0))


def test_linkage_probability_scalar_and_array():
    toit = InfectiousnessToTransmissionTime(rng_seed=21)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=22)

    scalar = estimate_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=0,
        temporal_distance=0,
        num_simulations=50,
        max_intermediate_hosts=2,
    )
    assert isinstance(scalar, float)
    assert 0.0 <= scalar <= 1.0

    arr = estimate_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=np.array([0, 1, 2]),
        temporal_distance=np.array([0, 1, 2]),
        num_simulations=50,
        max_intermediate_hosts=2,
        cache_unique_distances=False,
    )
    assert arr.shape == (3,)
    assert np.all((arr >= 0.0) & (arr <= 1.0))

    with pytest.raises(ValueError, match="same length"):
        estimate_linkage_probability(
            transmission_profile=toit,
            clock=clock,
            genetic_distance=[0, 1],
            temporal_distance=[1],
            num_simulations=10,
            max_intermediate_hosts=2,
            cache_unique_distances=False,
        )

    empty = estimate_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=[],
        temporal_distance=[],
        num_simulations=10,
        max_intermediate_hosts=2,
        cache_unique_distances=False,
    )
    assert np.isnan(empty)


def test_pairwise_linkage_probability_matrix_singleton():
    toit = InfectiousnessToTransmissionTime(rng_seed=31)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=32)

    mat = estimate_linkage_probability_grid(
        transmission_profile=toit,
        clock=clock,
        genetic_distances=np.array([0]),
        temporal_distances=np.array([0]),
        num_simulations=50,
        max_intermediate_hosts=2,
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
    toit = InfectiousnessToTransmissionTime(rng_seed=seed)
    out = estimate_temporal_linkage_probability(
        temporal_dist, transmission_profile=toit, num_simulations=50
    )
    assert out.shape == (temporal_dist.size,)
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_genetic_linkage_probability_modes_and_errors():
    toit = InfectiousnessToTransmissionTime(rng_seed=51)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=52)

    out_raw = estimate_genetic_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=[0, 1],
        num_simulations=50,
        max_intermediate_hosts=2,
        included_intermediate_counts=(0, 1),
        output_mode="raw",
    )
    assert out_raw.shape == (2,)
    assert np.all(np.isfinite(out_raw))
    assert np.all(out_raw >= 0.0)

    out_normalized = estimate_genetic_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=[0, 1],
        num_simulations=50,
        max_intermediate_hosts=2,
        included_intermediate_counts=(0, 1),
        output_mode="normalized",
    )
    assert out_normalized.shape == (2,)
    assert np.all((out_normalized >= 0.0) & (out_normalized <= 1.0))

    out_all = estimate_genetic_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=[0, 1],
        num_simulations=50,
        max_intermediate_hosts=2,
        included_intermediate_counts=None,
        output_mode="raw",
    )
    assert out_all.shape == (2, 3)
    assert np.all(np.isfinite(out_all))
    assert np.all((out_all >= 0.0) & (out_all <= 2.0))

    with pytest.raises(ValueError, match="included_intermediate_counts"):
        estimate_genetic_linkage_probability(
            transmission_profile=toit,
            clock=clock,
            genetic_distance=[0],
            num_simulations=10,
            max_intermediate_hosts=2,
            included_intermediate_counts=(3,),
            output_mode="normalized",
        )

    with pytest.raises(ValueError, match="output_mode must be"):
        estimate_genetic_linkage_probability(
            transmission_profile=toit,
            clock=clock,
            genetic_distance=[0],
            num_simulations=10,
            max_intermediate_hosts=2,
            included_intermediate_counts=(0,),
            output_mode="relative",
        )

    with pytest.raises(ValueError, match="output_mode must be"):
        estimate_genetic_linkage_probability(
            transmission_profile=toit,
            clock=clock,
            genetic_distance=[0],
            num_simulations=10,
            max_intermediate_hosts=2,
            included_intermediate_counts=(0,),
            output_mode="unknown",
        )


def test_temporal_kernel_edge_cases():
    """Test temporal kernel with various edge cases."""
    # Test with larger temporal distances that exceed generation interval
    temporal_distance = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    diff_incubation = np.array([0.0, 1.0, -1.0, 2.0])
    generation_interval = np.array([3.0, 4.0, 5.0, 3.5])

    out = LinkageMonteCarloSamples.temporal_kernel(
        temporal_distance, diff_incubation, generation_interval
    )

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

    out = LinkageMonteCarloSamples.temporal_kernel(
        temporal_distance, diff_incubation, generation_interval
    )

    # abs(2.0 + 1.0) = 3.0 <= 5.0, so should be 1.0
    assert out[0] == 1.0


def test_genetic_kernel_multiple_intermediates():
    """Test genetic kernel with multiple intermediate hosts and various scenarios."""
    genetic_distance = np.array([0.0, 5.0, 10.0, 15.0])
    clock_rates = np.array([1e-3, 2e-3, 1.5e-3])
    sampling_delay_i = np.array([2.0, 3.0, 2.5])
    sampling_delay_j = np.array([2.0, 3.0, 2.5])
    included_intermediate_counts = np.array([[3.0, 4.0, 5.0], [3.5, 4.5, 5.5], [3.2, 4.2, 5.2]])
    diff_infection_ij = np.array([1.0, 1.5, 1.2])
    incubation_periods = np.array([[5.0, 5.0], [6.0, 6.0], [5.5, 5.5]])

    out = LinkageMonteCarloSamples.genetic_kernel(
        genetic_distance_ij=genetic_distance,
        clock_rates=clock_rates,
        sampling_delay_i=sampling_delay_i,
        sampling_delay_j=sampling_delay_j,
        generation_intervals=included_intermediate_counts,
        max_intermediate_hosts=2,
        diff_infection_ij=diff_infection_ij,
        incubation_periods=incubation_periods,
    )

    # Shape should be (num_distances, num_intermediates + 1)
    assert out.shape == (4, 3)
    # All probabilities should be valid
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 2.0))


def test_genetic_kernel_m_greater_than_zero():
    """Test genetic kernel scenarios with m > 0 (intermediate hosts)."""
    genetic_distance = np.array([10.0])
    clock_rates = np.array([1e-3, 2e-3])
    sampling_delay_i = np.array([2.0, 2.0])
    sampling_delay_j = np.array([2.0, 2.0])
    # Multiple columns for different intermediate scenarios
    included_intermediate_counts = np.array([[3.0, 4.0, 5.0, 6.0], [3.5, 4.5, 5.5, 6.5]])
    diff_infection_ij = np.array([2.0, 2.5])
    incubation_periods = np.array([[5.0, 5.0], [5.5, 5.5]])

    out = LinkageMonteCarloSamples.genetic_kernel(
        genetic_distance_ij=genetic_distance,
        clock_rates=clock_rates,
        sampling_delay_i=sampling_delay_i,
        sampling_delay_j=sampling_delay_j,
        generation_intervals=included_intermediate_counts,
        max_intermediate_hosts=3,  # M=3, so we test m=1, 2, 3
        diff_infection_ij=diff_infection_ij,
        incubation_periods=incubation_periods,
    )

    # Shape: (1 distance, 4 scenarios: m=0,1,2,3)
    assert out.shape == (1, 4)
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 2.0))


def test_linkage_probability_various_intermediate_generations():
    """Test estimate_linkage_probability with different included_intermediate_counts tuples."""
    toit = InfectiousnessToTransmissionTime(rng_seed=100)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=101)

    # Test with only direct transmission (m=0)
    result_direct = estimate_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=5,
        temporal_distance=3,
        included_intermediate_counts=(0,),
        max_intermediate_hosts=3,
        num_simulations=50,
        cache_unique_distances=False,
    )
    assert isinstance(result_direct, float)
    assert 0.0 <= result_direct <= 1.0

    # Test with multiple intermediate scenarios
    result_multi = estimate_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=5,
        temporal_distance=3,
        included_intermediate_counts=(0, 1, 2),
        max_intermediate_hosts=3,
        num_simulations=50,
        cache_unique_distances=False,
    )
    assert isinstance(result_multi, float)
    assert 0.0 <= result_multi <= 1.0


def test_temporal_linkage_probability_scalar_input():
    """Test estimate_temporal_linkage_probability with scalar input."""
    toit = InfectiousnessToTransmissionTime(rng_seed=130)

    # Scalar input
    result_scalar = estimate_temporal_linkage_probability(
        temporal_distance=5.0,
        transmission_profile=toit,
        num_simulations=50,
    )

    assert result_scalar.shape == (1,)
    assert 0.0 <= result_scalar[0] <= 1.0


def test_genetic_linkage_probability_supported_modes_with_none():
    """Test estimate_genetic_linkage_probability with included_intermediate_counts=None."""
    genetic_dist = np.array([0, 5, 10])

    # Test 'raw' with None
    result_raw = estimate_genetic_linkage_probability(
        transmission_profile=InfectiousnessToTransmissionTime(rng_seed=150),
        clock=MolecularClock(use_relaxed_clock=False, rng_seed=151),
        genetic_distance=genetic_dist,
        num_simulations=50,
        max_intermediate_hosts=2,
        included_intermediate_counts=None,
        output_mode="raw",
    )
    assert result_raw.shape == (3, 3)  # 3 distances, 3 scenarios (m=0,1,2)
    assert np.all(np.isfinite(result_raw))
    assert np.all((result_raw >= 0.0) & (result_raw <= 2.0))

    # Test 'normalized' with None
    result_normalized = estimate_genetic_linkage_probability(
        transmission_profile=InfectiousnessToTransmissionTime(rng_seed=150),
        clock=MolecularClock(use_relaxed_clock=False, rng_seed=151),
        genetic_distance=genetic_dist,
        num_simulations=50,
        max_intermediate_hosts=2,
        included_intermediate_counts=None,
        output_mode="normalized",
    )
    assert result_normalized.shape == (3, 3)
    assert np.all((result_normalized >= 0.0) & (result_normalized <= 1.0))
    # Each row should sum to ~1.0 when there is support, otherwise remain all zeros.
    expected_row_sums = np.where(result_raw.sum(axis=1) > 0.0, 1.0, 0.0)
    row_sums = result_normalized.sum(axis=1)
    np.testing.assert_allclose(row_sums, expected_row_sums, atol=1e-10)


def test_scalar_intermediate_generation_selection():
    toit = InfectiousnessToTransmissionTime(rng_seed=152)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=153)

    p_link = estimate_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=2,
        temporal_distance=3,
        included_intermediate_counts=0,
        max_intermediate_hosts=2,
        num_simulations=50,
        cache_unique_distances=False,
    )
    assert isinstance(p_link, float)
    assert 0.0 <= p_link <= 1.0

    p_genetic = estimate_genetic_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=[2],
        num_simulations=50,
        max_intermediate_hosts=2,
        included_intermediate_counts=0,
        output_mode="normalized",
    )
    assert p_genetic.shape == (1,)
    assert 0.0 <= p_genetic[0] <= 1.0


def test_linkage_probability_single_element_no_cache():
    """Test estimate_linkage_probability with single element and cache disabled."""
    toit = InfectiousnessToTransmissionTime(rng_seed=170)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=171)

    # Single element array with cache disabled
    result = estimate_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=np.array([5]),
        temporal_distance=np.array([3]),
        num_simulations=30,
        max_intermediate_hosts=2,
        cache_unique_distances=False,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert 0.0 <= result[0] <= 1.0


def test_epilink_run_simulations_reproducibility():
    """Test that LinkageMonteCarloSamples.run_simulations produces reproducible results with same seed."""
    toit1 = InfectiousnessToTransmissionTime(rng_seed=200)
    clock1 = MolecularClock(use_relaxed_clock=False, rng_seed=201)

    toit2 = InfectiousnessToTransmissionTime(rng_seed=200)
    clock2 = MolecularClock(use_relaxed_clock=False, rng_seed=201)

    sim1 = LinkageMonteCarloSamples.run_simulations(
        toit1, clock1, num_simulations=10, max_intermediate_hosts=2
    )
    sim2 = LinkageMonteCarloSamples.run_simulations(
        toit2, clock2, num_simulations=10, max_intermediate_hosts=2
    )

    np.testing.assert_array_equal(sim1.incubation_periods, sim2.incubation_periods)
    np.testing.assert_array_equal(sim1.generation_intervals, sim2.generation_intervals)
    np.testing.assert_array_equal(sim1.clock_rates, sim2.clock_rates)


def test_linkage_probability_with_larger_no_intermediates():
    """Test estimate_linkage_probability with larger number of intermediate hosts."""
    toit = InfectiousnessToTransmissionTime(rng_seed=210)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=211)

    result = estimate_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=10,
        temporal_distance=5,
        included_intermediate_counts=(0, 1, 2, 3, 4, 5),
        max_intermediate_hosts=10,
        num_simulations=50,
        cache_unique_distances=False,
    )

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_genetic_linkage_probability_scalar_distance():
    """Test estimate_genetic_linkage_probability with scalar distance input."""
    toit = InfectiousnessToTransmissionTime(rng_seed=220)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=221)

    # Scalar input should be converted to array
    result = estimate_genetic_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=5,  # scalar
        num_simulations=30,
        max_intermediate_hosts=2,
        included_intermediate_counts=(0, 1),
        output_mode="normalized",
    )

    assert result.shape == (1,)
    assert 0.0 <= result[0] <= 1.0


def test_linkage_probability_no_cache_direct_path():
    """Test estimate_linkage_probability direct computation path (no caching)."""
    toit = InfectiousnessToTransmissionTime(rng_seed=230)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=231)

    # Test with cache disabled (default behavior for non-duplicate data)
    result = estimate_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=np.array([0, 5, 10]),
        temporal_distance=np.array([0, 2, 5]),
        included_intermediate_counts=(0,),
        max_intermediate_hosts=2,
        num_simulations=50,
        cache_unique_distances=False,
    )

    assert result.shape == (3,)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_linkage_probability_multiple_intermediate_generations():
    """Test estimate_linkage_probability with multiple intermediate generation scenarios."""
    toit = InfectiousnessToTransmissionTime(rng_seed=240)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=241)

    # Test with m=0, 1, 2
    result = estimate_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=np.array([5, 10, 15]),
        temporal_distance=np.array([3, 5, 7]),
        included_intermediate_counts=(0, 1, 2),
        max_intermediate_hosts=5,
        num_simulations=50,
        cache_unique_distances=False,
    )

    assert result.shape == (3,)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_linkage_probability_high_genetic_distance():
    """Test estimate_linkage_probability with high genetic distances."""
    toit = InfectiousnessToTransmissionTime(rng_seed=250)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=251)

    # High genetic distances should typically give lower probabilities
    result = estimate_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=np.array([0, 50, 100, 200]),
        temporal_distance=np.array([0, 10, 20, 30]),
        included_intermediate_counts=(0, 1),
        max_intermediate_hosts=3,
        num_simulations=50,
        cache_unique_distances=False,
    )

    assert result.shape == (4,)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_linkage_probability_high_temporal_distance():
    """Test estimate_linkage_probability with high temporal distances."""
    toit = InfectiousnessToTransmissionTime(rng_seed=260)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=261)

    # High temporal distances should typically give lower probabilities
    result = estimate_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=np.array([5, 5, 5, 5]),
        temporal_distance=np.array([0, 20, 50, 100]),
        included_intermediate_counts=(0,),
        max_intermediate_hosts=2,
        num_simulations=50,
        cache_unique_distances=False,
    )

    assert result.shape == (4,)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_linkage_probability_matrix_single_distance():
    """Test estimate_linkage_probability_grid with single genetic and temporal distance."""
    toit = InfectiousnessToTransmissionTime(rng_seed=270)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=271)

    result = estimate_linkage_probability_grid(
        transmission_profile=toit,
        clock=clock,
        genetic_distances=np.array([5]),
        temporal_distances=np.array([3]),
        included_intermediate_counts=(0,),
        max_intermediate_hosts=2,
        num_simulations=50,
    )

    assert result.shape == (1, 1)
    assert 0.0 <= result[0, 0] <= 1.0


def test_linkage_probability_zero_distances():
    """Test estimate_linkage_probability with zero genetic and temporal distances."""
    toit = InfectiousnessToTransmissionTime(rng_seed=310)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=311)

    # Zero distances should give high probability (same individual or very close)
    result = estimate_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=0,
        temporal_distance=0,
        included_intermediate_counts=(0,),
        max_intermediate_hosts=2,
        num_simulations=100,
        cache_unique_distances=False,
    )

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
    # Should be relatively high probability
    assert result > 0.0


def test_linkage_probability_mixed_distances():
    """Test estimate_linkage_probability with mixed high/low distances."""
    toit = InfectiousnessToTransmissionTime(rng_seed=320)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=321)

    # Mix of close and far pairs
    result = estimate_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=np.array([0, 100, 5, 50]),
        temporal_distance=np.array([0, 50, 2, 10]),
        included_intermediate_counts=(0, 1),
        max_intermediate_hosts=3,
        num_simulations=50,
        cache_unique_distances=False,
    )

    assert result.shape == (4,)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_linkage_probability_all_intermediate_scenarios():
    """Test estimate_linkage_probability selecting all possible intermediate scenarios."""
    toit = InfectiousnessToTransmissionTime(rng_seed=330)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=331)

    # Select all scenarios from m=0 to m=5
    result = estimate_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=np.array([10]),
        temporal_distance=np.array([5]),
        included_intermediate_counts=(0, 1, 2, 3, 4, 5),
        max_intermediate_hosts=5,
        num_simulations=50,
        cache_unique_distances=False,
    )

    assert result.shape == (1,)
    assert 0.0 <= result[0] <= 1.0


def test_linkage_probability_single_intermediate_scenario():
    """Test estimate_linkage_probability with only direct transmission (m=0)."""
    toit = InfectiousnessToTransmissionTime(rng_seed=340)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=341)

    result_m0 = estimate_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=5,
        temporal_distance=3,
        included_intermediate_counts=(0,),  # Only direct
        max_intermediate_hosts=3,
        num_simulations=50,
        cache_unique_distances=False,
    )

    assert isinstance(result_m0, float)
    assert 0.0 <= result_m0 <= 1.0


def test_linkage_probability_consistency():
    """Test that estimate_linkage_probability gives consistent results for same inputs."""
    toit1 = InfectiousnessToTransmissionTime(rng_seed=360)
    clock1 = MolecularClock(use_relaxed_clock=False, rng_seed=361)

    toit2 = InfectiousnessToTransmissionTime(rng_seed=360)  # Same seed
    clock2 = MolecularClock(use_relaxed_clock=False, rng_seed=361)

    result1 = estimate_linkage_probability(
        transmission_profile=toit1,
        clock=clock1,
        genetic_distance=10,
        temporal_distance=5,
        included_intermediate_counts=(0, 1),
        max_intermediate_hosts=2,
        num_simulations=50,
        cache_unique_distances=False,
    )

    result2 = estimate_linkage_probability(
        transmission_profile=toit2,
        clock=clock2,
        genetic_distance=10,
        temporal_distance=5,
        included_intermediate_counts=(0, 1),
        max_intermediate_hosts=2,
        num_simulations=50,
        cache_unique_distances=False,
    )

    assert result1 == result2


def test_linkage_probability_with_different_num_simulations():
    """Test estimate_linkage_probability with different numbers of simulations."""
    toit = InfectiousnessToTransmissionTime(rng_seed=370)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=371)

    # Fewer simulations
    result_few = estimate_linkage_probability(
        transmission_profile=toit,
        clock=clock,
        genetic_distance=10,
        temporal_distance=5,
        included_intermediate_counts=(0,),
        max_intermediate_hosts=2,
        num_simulations=20,
        cache_unique_distances=False,
    )

    # More simulations (reset RNG for comparison)
    toit2 = InfectiousnessToTransmissionTime(rng_seed=370)
    clock2 = MolecularClock(use_relaxed_clock=False, rng_seed=371)

    result_many = estimate_linkage_probability(
        transmission_profile=toit2,
        clock=clock2,
        genetic_distance=10,
        temporal_distance=5,
        included_intermediate_counts=(0,),
        max_intermediate_hosts=2,
        num_simulations=100,
        cache_unique_distances=False,
    )

    # Both should be valid probabilities
    assert isinstance(result_few, float)
    assert isinstance(result_many, float)
    assert 0.0 <= result_few <= 1.0
    assert 0.0 <= result_many <= 1.0


class TestCachingPath:
    """Test the caching path in estimate_linkage_probability with duplicate distance pairs."""

    @pytest.mark.parametrize(
        "genetic_dist, temporal_dist, included_intermediate_counts, max_intermediate_hosts",
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
        included_intermediate_counts,
        max_intermediate_hosts,
    ):
        """Test caching path when there are duplicate distance pairs."""
        toit = InfectiousnessToTransmissionTime(rng_seed=1000)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=1001)

        result = estimate_linkage_probability(
            transmission_profile=toit,
            clock=clock,
            genetic_distance=genetic_dist,
            temporal_distance=temporal_dist,
            included_intermediate_counts=included_intermediate_counts,
            max_intermediate_hosts=max_intermediate_hosts,
            num_simulations=50,
            cache_unique_distances=True,
        )

        assert result.shape == (genetic_dist.size,)
        assert np.all((result >= 0.0) & (result <= 1.0))
        pairs = list(zip(genetic_dist, temporal_dist, strict=False))
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                if pairs[i] == pairs[j]:
                    assert result[i] == result[j]

    def test_linkage_probability_all_same_pairs_cache_enabled(self):
        """Test caching when all pairs are identical."""
        toit = InfectiousnessToTransmissionTime(rng_seed=1010)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=1011)

        # All identical pairs
        genetic_dist = np.array([10, 10, 10, 10])
        temporal_dist = np.array([5, 5, 5, 5])

        result = estimate_linkage_probability(
            transmission_profile=toit,
            clock=clock,
            genetic_distance=genetic_dist,
            temporal_distance=temporal_dist,
            included_intermediate_counts=(0,),
            max_intermediate_hosts=2,
            num_simulations=50,
            cache_unique_distances=True,
        )

        assert result.shape == (4,)
        # All results should be identical
        np.testing.assert_array_equal(result, np.full(4, result[0]))

    def test_linkage_probability_cache_with_validation_error(self):
        """Test that validation errors are raised even with caching enabled."""
        toit = InfectiousnessToTransmissionTime(rng_seed=1020)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=1021)

        genetic_dist = np.array([5, 10, 5])
        temporal_dist = np.array([3, 7, 3])

        # Test with invalid included_intermediate_counts in caching path
        with pytest.raises(ValueError, match="included_intermediate_counts must be within"):
            estimate_linkage_probability(
                transmission_profile=toit,
                clock=clock,
                genetic_distance=genetic_dist,
                temporal_distance=temporal_dist,
                included_intermediate_counts=(10,),  # Exceeds max_intermediate_hosts=3
                max_intermediate_hosts=3,
                num_simulations=50,
                cache_unique_distances=True,
            )

        # Test with negative included_intermediate_counts in caching path
        with pytest.raises(ValueError, match="included_intermediate_counts must be within"):
            estimate_linkage_probability(
                transmission_profile=toit,
                clock=clock,
                genetic_distance=genetic_dist,
                temporal_distance=temporal_dist,
                included_intermediate_counts=(-1, 0),
                max_intermediate_hosts=3,
                num_simulations=50,
                cache_unique_distances=True,
            )


class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions."""

    def test_linkage_probability_single_element_with_cache(self):
        """Test that single element doesn't use cache path (size > 1 requirement)."""
        toit = InfectiousnessToTransmissionTime(rng_seed=2000)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=2001)

        # Single element - should skip caching even if enabled
        result = estimate_linkage_probability(
            transmission_profile=toit,
            clock=clock,
            genetic_distance=np.array([5]),
            temporal_distance=np.array([3]),
            included_intermediate_counts=(0,),
            max_intermediate_hosts=2,
            num_simulations=50,
            cache_unique_distances=True,  # Enabled but size=1 so skips caching
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_linkage_probability_two_elements_unique_cache(self):
        """Test caching with exactly 2 unique elements."""
        toit = InfectiousnessToTransmissionTime(rng_seed=2010)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=2011)

        genetic_dist = np.array([5, 10])
        temporal_dist = np.array([3, 7])

        result = estimate_linkage_probability(
            transmission_profile=toit,
            clock=clock,
            genetic_distance=genetic_dist,
            temporal_distance=temporal_dist,
            included_intermediate_counts=(0, 1),
            max_intermediate_hosts=2,
            num_simulations=50,
            cache_unique_distances=True,
        )

        assert result.shape == (2,)
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_linkage_probability_cache_min_intermediate_zero(self):
        """Test caching path with included_intermediate_counts starting at 0."""
        toit = InfectiousnessToTransmissionTime(rng_seed=2020)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=2021)

        genetic_dist = np.array([5, 10, 5])
        temporal_dist = np.array([3, 7, 3])

        result = estimate_linkage_probability(
            transmission_profile=toit,
            clock=clock,
            genetic_distance=genetic_dist,
            temporal_distance=temporal_dist,
            included_intermediate_counts=(0,),  # Only minimum
            max_intermediate_hosts=5,
            num_simulations=50,
            cache_unique_distances=True,
        )

        assert result.shape == (3,)
        assert result[0] == result[2]

    def test_genetic_linkage_probability_boundary_values(self):
        """Test estimate_genetic_linkage_probability with boundary included_intermediate_counts."""
        toit = InfectiousnessToTransmissionTime(rng_seed=2030)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=2031)

        # Test at exact boundaries
        result_min = estimate_genetic_linkage_probability(
            transmission_profile=toit,
            clock=clock,
            genetic_distance=[5],
            num_simulations=30,
            max_intermediate_hosts=10,
            included_intermediate_counts=(0,),  # Minimum
            output_mode="normalized",
        )
        assert result_min.shape == (1,)

        toit2 = InfectiousnessToTransmissionTime(rng_seed=2030)
        clock2 = MolecularClock(use_relaxed_clock=False, rng_seed=2031)

        result_max = estimate_genetic_linkage_probability(
            transmission_profile=toit2,
            clock=clock2,
            genetic_distance=[5],
            num_simulations=30,
            max_intermediate_hosts=10,
            included_intermediate_counts=(10,),  # Maximum
            output_mode="normalized",
        )
        assert result_max.shape == (1,)

    def test_linkage_probability_matrix_multiple_distances(self):
        """Test estimate_linkage_probability_grid with multiple genetic and temporal distances."""
        toit = InfectiousnessToTransmissionTime(rng_seed=2040)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=2041)

        genetic_distances = np.array([0, 5, 10, 15, 20])
        temporal_distances = np.array([0, 2, 5, 10, 15])

        result = estimate_linkage_probability_grid(
            transmission_profile=toit,
            clock=clock,
            genetic_distances=genetic_distances,
            temporal_distances=temporal_distances,
            included_intermediate_counts=(0, 1, 2),
            max_intermediate_hosts=5,
            num_simulations=50,
        )

        assert result.shape == (5, 5)
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_genetic_linkage_probability_large_distance_array(self):
        """Test estimate_genetic_linkage_probability with larger distance array."""
        toit = InfectiousnessToTransmissionTime(rng_seed=2060)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=2061)

        genetic_dist = np.arange(0, 50, 5)  # [0, 5, 10, ..., 45]

        result = estimate_genetic_linkage_probability(
            transmission_profile=toit,
            clock=clock,
            genetic_distance=genetic_dist,
            num_simulations=50,
            max_intermediate_hosts=3,
            included_intermediate_counts=(0, 1, 2),
            output_mode="normalized",
        )

        assert result.shape == (10,)
        assert np.all((result >= 0.0) & (result <= 1.0))


class TestZeroProbabilityEdgeCases:
    """Test cases that should produce zero or near-zero probabilities."""

    def test_linkage_probability_extreme_distances(self):
        """Test with extremely large distances that should give very low probabilities."""
        toit = InfectiousnessToTransmissionTime(rng_seed=3000)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=3001)

        # Very large distances
        result = estimate_linkage_probability(
            transmission_profile=toit,
            clock=clock,
            genetic_distance=1000,  # Very large
            temporal_distance=365,  # One year
            included_intermediate_counts=(0,),
            max_intermediate_hosts=2,
            num_simulations=100,
            cache_unique_distances=False,
        )

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_linkage_probability_cache_extreme_distances(self):
        """Test caching path with extreme distances."""
        toit = InfectiousnessToTransmissionTime(rng_seed=3010)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=3011)

        genetic_dist = np.array([1000, 500, 1000])
        temporal_dist = np.array([365, 180, 365])

        result = estimate_linkage_probability(
            transmission_profile=toit,
            clock=clock,
            genetic_distance=genetic_dist,
            temporal_distance=temporal_dist,
            included_intermediate_counts=(0, 1),
            max_intermediate_hosts=3,
            num_simulations=50,
            cache_unique_distances=True,
        )

        assert result.shape == (3,)
        assert result[0] == result[2]  # Duplicates should match


class TestComplexScenarios:
    """Test complex scenarios combining multiple features."""

    def test_genetic_linkage_probability_all_kinds_with_selection(self):
        """Test supported output modes with specific included_intermediate_counts."""
        toit = InfectiousnessToTransmissionTime(rng_seed=4020)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=4021)

        genetic_dist = np.array([5, 10, 15])

        # Test each supported mode with selected intermediates
        for kind in ["raw", "normalized"]:
            result = estimate_genetic_linkage_probability(
                transmission_profile=toit,
                clock=clock,
                genetic_distance=genetic_dist,
                num_simulations=50,
                max_intermediate_hosts=3,
                included_intermediate_counts=(0, 1, 2),
                output_mode=kind,
            )
            assert result.shape == (3,)
            assert np.all(np.isfinite(result))
            assert np.all(result >= 0.0)
            if kind == "normalized":
                assert np.all(result <= 1.0)

    def test_linkage_probability_matrix_with_zeros(self):
        """Test matrix computation with zero distances included."""
        toit = InfectiousnessToTransmissionTime(rng_seed=4030)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=4031)

        genetic_distances = np.array([0, 1, 2, 3])
        temporal_distances = np.array([0, 1, 2, 3])

        result = estimate_linkage_probability_grid(
            transmission_profile=toit,
            clock=clock,
            genetic_distances=genetic_distances,
            temporal_distances=temporal_distances,
            included_intermediate_counts=(0,),
            max_intermediate_hosts=2,
            num_simulations=50,
        )

        assert result.shape == (4, 4)
        assert np.all((result >= 0.0) & (result <= 1.0))
        # (0, 0) should have high probability
        assert result[0, 0] > 0.0


class TestIntermediateGenerationsSelection:
    """Test various included_intermediate_counts selections."""

    def test_linkage_probability_single_high_intermediate(self):
        """Test with only a high intermediate value selected."""
        toit = InfectiousnessToTransmissionTime(rng_seed=5000)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=5001)

        result = estimate_linkage_probability(
            transmission_profile=toit,
            clock=clock,
            genetic_distance=10,
            temporal_distance=5,
            included_intermediate_counts=(5,),  # Only high value
            max_intermediate_hosts=5,
            num_simulations=50,
            cache_unique_distances=False,
        )

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_linkage_probability_cache_single_high_intermediate(self):
        """Test caching path with only high intermediate value."""
        toit = InfectiousnessToTransmissionTime(rng_seed=5010)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=5011)

        genetic_dist = np.array([10, 15, 10])
        temporal_dist = np.array([5, 8, 5])

        result = estimate_linkage_probability(
            transmission_profile=toit,
            clock=clock,
            genetic_distance=genetic_dist,
            temporal_distance=temporal_dist,
            included_intermediate_counts=(4,),
            max_intermediate_hosts=5,
            num_simulations=50,
            cache_unique_distances=True,
        )

        assert result.shape == (3,)
        assert result[0] == result[2]

    def test_linkage_probability_non_consecutive_intermediates(self):
        """Test with non-consecutive intermediate values."""
        toit = InfectiousnessToTransmissionTime(rng_seed=5020)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=5021)

        # Select non-consecutive values: 0, 2, 4
        result = estimate_linkage_probability(
            transmission_profile=toit,
            clock=clock,
            genetic_distance=10,
            temporal_distance=5,
            included_intermediate_counts=(0, 2, 4),
            max_intermediate_hosts=5,
            num_simulations=50,
            cache_unique_distances=False,
        )

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_linkage_probability_cache_non_consecutive_intermediates(self):
        """Test caching with non-consecutive intermediate values."""
        toit = InfectiousnessToTransmissionTime(rng_seed=5030)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=5031)

        genetic_dist = np.array([10, 15, 10])
        temporal_dist = np.array([5, 8, 5])

        result = estimate_linkage_probability(
            transmission_profile=toit,
            clock=clock,
            genetic_distance=genetic_dist,
            temporal_distance=temporal_dist,
            included_intermediate_counts=(0, 3, 5),
            max_intermediate_hosts=5,
            num_simulations=50,
            cache_unique_distances=True,
        )

        assert result.shape == (3,)
        assert result[0] == result[2]


class TestVeryLargeArrays:
    """Test with larger arrays to ensure scalability."""

    def test_linkage_probability_large_array_with_cache(self):
        """Test with larger arrays and caching enabled."""
        toit = InfectiousnessToTransmissionTime(rng_seed=6000)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=6001)

        # Create large array with many duplicates
        np.random.seed(6000)
        genetic_dist = np.random.choice([5, 10, 15, 20], size=50)
        temporal_dist = np.random.choice([3, 7, 10, 15], size=50)

        result = estimate_linkage_probability(
            transmission_profile=toit,
            clock=clock,
            genetic_distance=genetic_dist,
            temporal_distance=temporal_dist,
            included_intermediate_counts=(0, 1),
            max_intermediate_hosts=3,
            num_simulations=50,
            cache_unique_distances=True,
        )

        assert result.shape == (50,)
        assert np.all((result >= 0.0) & (result <= 1.0))


class TestErrorConditions:
    """Test error conditions and validation paths."""

    def test_linkage_probability_mismatched_sizes_no_cache(self):
        """Test ValueError for mismatched sizes without caching."""
        toit = InfectiousnessToTransmissionTime(rng_seed=7000)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=7001)

        with pytest.raises(ValueError, match="must have the same length"):
            estimate_linkage_probability(
                transmission_profile=toit,
                clock=clock,
                genetic_distance=np.array([5, 10]),
                temporal_distance=np.array([3]),  # Different size
                included_intermediate_counts=(0,),
                max_intermediate_hosts=2,
                num_simulations=50,
                cache_unique_distances=False,
            )

    def test_linkage_probability_empty_input_no_cache(self):
        """Test empty input returns np.nan without caching."""
        toit = InfectiousnessToTransmissionTime(rng_seed=7010)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=7011)

        result = estimate_linkage_probability(
            transmission_profile=toit,
            clock=clock,
            genetic_distance=np.array([]),
            temporal_distance=np.array([]),
            included_intermediate_counts=(0,),
            max_intermediate_hosts=2,
            num_simulations=50,
            cache_unique_distances=False,
        )

        assert np.isnan(result)

    def test_linkage_probability_invalid_intermediate_no_cache(self):
        """Test ValueError for invalid included_intermediate_counts without cache."""
        toit = InfectiousnessToTransmissionTime(rng_seed=7020)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=7021)

        # Test exceeding max_intermediate_hosts
        with pytest.raises(ValueError, match="included_intermediate_counts must be within"):
            estimate_linkage_probability(
                transmission_profile=toit,
                clock=clock,
                genetic_distance=np.array([5]),
                temporal_distance=np.array([3]),
                included_intermediate_counts=(10,),  # Exceeds max_intermediate_hosts=3
                max_intermediate_hosts=3,
                num_simulations=50,
                cache_unique_distances=False,
            )

        # Test negative value
        with pytest.raises(ValueError, match="included_intermediate_counts must be within"):
            estimate_linkage_probability(
                transmission_profile=toit,
                clock=clock,
                genetic_distance=np.array([5]),
                temporal_distance=np.array([3]),
                included_intermediate_counts=(-1,),
                max_intermediate_hosts=3,
                num_simulations=50,
                cache_unique_distances=False,
            )

    def test_genetic_linkage_probability_invalid_intermediate_with_selection(self):
        """Test ValueError for invalid included_intermediate_counts in estimate_genetic_linkage_probability."""
        toit = InfectiousnessToTransmissionTime(rng_seed=7030)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=7031)

        # Test exceeding max_intermediate_hosts
        with pytest.raises(ValueError, match="included_intermediate_counts must be within"):
            estimate_genetic_linkage_probability(
                transmission_profile=toit,
                clock=clock,
                genetic_distance=[5],
                num_simulations=30,
                max_intermediate_hosts=3,
                included_intermediate_counts=(5,),  # Exceeds max_intermediate_hosts=3
                output_mode="normalized",
            )

        # Test negative value
        with pytest.raises(ValueError, match="included_intermediate_counts must be within"):
            estimate_genetic_linkage_probability(
                transmission_profile=toit,
                clock=clock,
                genetic_distance=[5],
                num_simulations=30,
                max_intermediate_hosts=3,
                included_intermediate_counts=(-2,),
                output_mode="normalized",
            )

    def test_genetic_linkage_probability_invalid_kind_with_selection(self):
        """Test ValueError for invalid kind with included_intermediate_counts specified."""
        toit = InfectiousnessToTransmissionTime(rng_seed=7040)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=7041)

        with pytest.raises(ValueError, match="output_mode must be"):
            estimate_genetic_linkage_probability(
                transmission_profile=toit,
                clock=clock,
                genetic_distance=[5],
                num_simulations=30,
                max_intermediate_hosts=3,
                included_intermediate_counts=(0, 1),
                output_mode="invalid",
            )

    def test_genetic_linkage_probability_invalid_kind_without_selection(self):
        """Test ValueError for invalid kind with included_intermediate_counts=None."""
        toit = InfectiousnessToTransmissionTime(rng_seed=7050)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=7051)

        with pytest.raises(ValueError, match="output_mode must be"):
            estimate_genetic_linkage_probability(
                transmission_profile=toit,
                clock=clock,
                genetic_distance=[5],
                num_simulations=30,
                max_intermediate_hosts=3,
                included_intermediate_counts=None,
                output_mode="wrong_kind",
            )

    def test_genetic_linkage_probability_supported_modes_return_correct_types(self):
        """Test that supported modes return appropriate shapes with None."""
        toit = InfectiousnessToTransmissionTime(rng_seed=7060)
        clock = MolecularClock(use_relaxed_clock=False, rng_seed=7061)

        genetic_dist = [5, 10]

        # Test that each supported mode with None returns (K, M+1) shape
        for kind in ["raw", "normalized"]:
            result = estimate_genetic_linkage_probability(
                transmission_profile=toit,
                clock=clock,
                genetic_distance=genetic_dist,
                num_simulations=30,
                max_intermediate_hosts=2,
                included_intermediate_counts=None,
                output_mode=kind,
            )
            assert result.shape == (2, 3)  # 2 distances, 3 scenarios (m=0,1,2)
