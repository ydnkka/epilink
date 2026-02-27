"""
Tests to improve code coverage for uncovered lines.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import pytest

from epilink import (
    TOIT,
    TOST,
    MolecularClock,
    Epilink,
    linkage_probability,
    linkage_probability_matrix,
    populate_epidemic_data,
    simulate_genomic_data,
    SequencePacker64,
    PackedGenomicData,
)


# ============================================================================
# Tests for simulate_epidemic_and_genomic.py uncovered lines
# ============================================================================

def test_populate_epidemic_data_stochastic_mode():
    """Test populate_epidemic_data with stochastic sampling (non-zero sampling_scale)."""
    tree = nx.DiGraph()
    tree.add_edges_from([("A", "B"), ("A", "C")])
    toit = TOIT(rng_seed=42)

    # Use non-zero sampling_scale to trigger stochastic mode (lines 99-102)
    out = populate_epidemic_data(
        toit=toit,
        tree=tree,
        prop_sampled=1.0,
        sampling_scale=1.0,  # Non-zero triggers stochastic mode
        sampling_shape=2.0,
        root_start_range=5,  # Should be int, not float
    )

    # Check that all nodes have valid dates
    for node in out.nodes:
        assert out.nodes[node]["exposure_date"] >= 0
        assert out.nodes[node]["date_infectious"] > out.nodes[node]["exposure_date"]
        assert out.nodes[node]["date_symptom_onset"] > out.nodes[node]["date_infectious"]
        assert out.nodes[node]["sample_date"] >= out.nodes[node]["date_symptom_onset"]


def test_simulate_genomic_data_with_mutations():
    """Test genomic simulation with actual mutations."""
    tree = nx.DiGraph()
    tree.add_edge("A", "B")
    tree.nodes["A"]["sample_date"] = 0.0
    tree.nodes["B"]["sample_date"] = 10.0  # Longer time = more mutations
    tree.nodes["B"]["exposure_date"] = 5.0

    clock = MolecularClock(subs_rate=1e-3, relax_rate=False, gen_len=1000, rng_seed=123)
    out = simulate_genomic_data(clock=clock, tree=tree, return_raw=True)

    # Check that sequences were generated
    assert "raw" in out
    assert "packed" in out
    assert "linear" in out["raw"]
    assert "poisson" in out["raw"]

    # Sequences should have the right length
    linear_raw = out["raw"]["linear"]
    assert linear_raw.shape[1] == 1000


def test_packed_genomic_data_hamming():
    """Test Hamming distance calculation on packed data."""
    int8_matrix = np.array(
        [
            [0, 1, 2, 3] * 20,  # 80 bases
            [0, 1, 2, 3] * 20,
            [3, 2, 1, 0] * 20,
        ],
        dtype=np.int8,
    )
    node_map = {"n1": 0, "n2": 1, "n3": 2}
    base_map = {0: "A", 1: "C", 2: "G", 3: "T"}

    packed_data = PackedGenomicData(int8_matrix, original_length=80, node_map=node_map, base_map=base_map)

    # Get Hamming distance matrix
    hamming = packed_data.compute_hamming_distances()

    assert hamming.shape == (3, 3)
    # n1 and n2 should be identical (distance 0)
    assert hamming[0, 1] == 0.0
    # n1 and n3 should differ at all positions (distance 80)
    assert hamming[0, 2] == 80.0


# ============================================================================
# Tests for transmission_linkage_model.py uncovered lines
# ============================================================================

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

    # Shape should be (num_distances, num_intermediates + 1)
    assert out.shape == (3, 3)
    # All probabilities should be between 0 and 1
    assert np.all((out >= 0.0) & (out <= 1.0))


# ============================================================================
# Tests for infectiousness_profile.py uncovered lines
# ============================================================================

def test_molecular_clock_invalid_parameters():
    """Test MolecularClock validation (lines 236-241)."""
    with pytest.raises(ValueError, match="subs_rate must be positive"):
        MolecularClock(subs_rate=-0.1)

    with pytest.raises(ValueError, match="subs_rate must be positive"):
        MolecularClock(subs_rate=0.0)

    with pytest.raises(ValueError, match="subs_rate_sigma must be non-negative"):
        MolecularClock(subs_rate=1e-3, subs_rate_sigma=-0.1)

    with pytest.raises(ValueError, match="gen_len must be positive"):
        MolecularClock(subs_rate=1e-3, gen_len=0)

    with pytest.raises(ValueError, match="gen_len must be positive"):
        MolecularClock(subs_rate=1e-3, gen_len=-10)


def test_toit_base_class_not_implemented_methods():
    """Test that base class abstract methods raise NotImplementedError (lines 387, 442, 463)."""
    from epilink.infectiousness_profile import InfectiousnessProfile

    # Create a minimal subclass without implementing abstract methods
    class MinimalProfile(InfectiousnessProfile):
        pass

    profile = MinimalProfile(a=0.0, b=10.0)

    with pytest.raises(NotImplementedError):
        profile.pdf(np.array([1.0]))

    with pytest.raises(NotImplementedError):
        profile.rvs(size=1)


def test_tost_full_workflow():
    """Test TOST distribution methods."""
    tost = TOST(a=-20.0, b=20.0, rng_seed=777)

    # Test PDF
    x_vals = np.linspace(-10, 10, 20)
    pdf_vals = tost.pdf(x_vals)
    assert pdf_vals.shape == (20,)
    assert np.all(pdf_vals >= 0.0)

    # Test CDF
    cdf_vals = tost.cdf(x_vals)
    assert cdf_vals.shape == (20,)
    assert np.all((cdf_vals >= 0.0) & (cdf_vals <= 1.0))
    # CDF should be monotonically increasing
    assert np.all(np.diff(cdf_vals) >= -1e-6)  # Allow small numerical errors

    # Test mean
    mean_val = tost.mean()
    assert isinstance(mean_val, float)
    assert tost.a <= mean_val <= tost.b

    # Test sampling
    samples = tost.rvs(size=100)
    assert samples.shape == (100,)
    assert np.all((samples >= tost.a) & (samples <= tost.b))


def test_toit_full_workflow():
    """Test TOIT distribution methods."""
    toit = TOIT(a=0.0, b=50.0, rng_seed=888)

    # Test PDF
    x_vals = np.linspace(0, 30, 20)
    pdf_vals = toit.pdf(x_vals)
    assert pdf_vals.shape == (20,)
    assert np.all(pdf_vals >= 0.0)

    # Test CDF
    cdf_vals = toit.cdf(x_vals)
    assert cdf_vals.shape == (20,)
    assert np.all((cdf_vals >= 0.0) & (cdf_vals <= 1.0))

    # Test mean
    mean_val = toit.mean()
    assert isinstance(mean_val, float)
    assert toit.a <= mean_val <= toit.b

    # Test generation_time sampling
    gen_times = toit.generation_time(size=50)
    assert gen_times.shape == (50,)
    assert np.all(gen_times >= 0.0)


def test_molecular_clock_with_rate_relaxation():
    """Test MolecularClock with relaxed clock rates."""
    clock = MolecularClock(
        subs_rate=1e-3,
        relax_rate=True,
        subs_rate_sigma=0.2,
        gen_len=1000,
        rng_seed=555
    )

    # Sample multiple rates
    rates = clock.sample_clock_rate_per_day(size=100)
    assert rates.shape == (100,)
    assert np.all(rates > 0.0)
    # With relaxed rates, there should be variation
    assert np.std(rates) > 0.0


def test_infectiousness_profile_sample_methods():
    """Test various sampling methods."""
    toit = TOIT(rng_seed=666)

    # Test sample_latent
    latent = toit.sample_latent(size=50)
    assert latent.shape == (50,)
    assert np.all(latent > 0.0)

    # Test sample_presymptomatic
    presymp = toit.sample_presymptomatic(size=50)
    assert presymp.shape == (50,)
    assert np.all(presymp > 0.0)

    # Test sample_incubation
    incub = toit.sample_incubation(size=50)
    assert incub.shape == (50,)
    assert np.all(incub > 0.0)

    # Test sample_symptomatic
    symp = toit.sample_symptomatic(size=50)
    assert symp.shape == (50,)
    assert np.all(symp > 0.0)






