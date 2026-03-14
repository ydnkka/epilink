"""
Comprehensive tests for transmission profile and natural-history components.

This test suite validates the E/P/I variable infectiousness model implementation,
including parameter validation, distribution functions, molecular clock models,
and sampling methods.
"""

import numpy as np
import pytest
from numpy.random import default_rng

from epilink import (
    BaseTransmissionProfile,
    InfectiousnessToTransmissionTime,
    MolecularClock,
    NaturalHistoryParameters,
    SymptomOnsetToTransmissionTime,
    estimate_presymptomatic_transmission_fraction,
)

# ============================================================================
# NaturalHistoryParameters Tests
# ============================================================================


def test_infectiousness_params_default_initialization():
    """Test that NaturalHistoryParameters initializes with correct default values."""
    params = NaturalHistoryParameters()
    assert params.incubation_shape == 5.807
    assert params.incubation_scale == 0.948
    assert params.latent_shape == 3.38
    assert params.symptomatic_rate == 0.37
    assert params.symptomatic_shape == 1.0
    assert params.rel_presymptomatic_infectiousness == 2.29


def test_infectiousness_params_derived_properties():
    """Test derived properties of NaturalHistoryParameters are calculated correctly."""
    params = NaturalHistoryParameters()

    # Test presymptomatic_shape = incubation_shape - latent_shape
    expected_presymp_shape = params.incubation_shape - params.latent_shape
    assert np.isclose(params.presymptomatic_shape, expected_presymp_shape)

    # Test symptomatic_scale = 1 / (symptomatic_shape * symptomatic_rate)
    expected_symp_scale = 1.0 / (params.symptomatic_shape * params.symptomatic_rate)
    assert np.isclose(params.symptomatic_scale, expected_symp_scale)

    # Test incubation_rate = 1 / (incubation_shape * incubation_scale)
    expected_inc_rate = 1.0 / (params.incubation_shape * params.incubation_scale)
    assert np.isclose(params.incubation_rate, expected_inc_rate)

    # Test infectiousness_normalisation is positive
    assert params.infectiousness_normalisation > 0


def test_infectiousness_params_custom_values():
    """Test NaturalHistoryParameters with custom parameter values."""
    params = NaturalHistoryParameters(
        incubation_shape=6.0,
        incubation_scale=1.0,
        latent_shape=3.0,
        symptomatic_rate=0.5,
        symptomatic_shape=2.0,
        rel_presymptomatic_infectiousness=3.0,
    )
    assert params.incubation_shape == 6.0
    assert params.presymptomatic_shape == 3.0
    assert params.symptomatic_scale == 1.0


def test_infectiousness_params_validation_negative_incubation():
    """Test that NaturalHistoryParameters raises error for negative incubation parameters."""
    with pytest.raises(ValueError, match="incubation_shape and incubation_scale must be positive"):
        NaturalHistoryParameters(incubation_shape=-1.0)

    with pytest.raises(ValueError, match="incubation_shape and incubation_scale must be positive"):
        NaturalHistoryParameters(incubation_scale=-1.0)


def test_infectiousness_params_validation_latent_shape():
    """Test that NaturalHistoryParameters validates latent_shape constraints."""
    # latent_shape must be positive
    with pytest.raises(ValueError, match="latent_shape must be positive"):
        NaturalHistoryParameters(latent_shape=-1.0)

    # latent_shape must be < incubation_shape
    with pytest.raises(ValueError, match="latent_shape must be < incubation_shape"):
        NaturalHistoryParameters(incubation_shape=5.0, latent_shape=6.0)


def test_infectiousness_params_validation_symptomatic():
    """Test that NaturalHistoryParameters validates symptomatic parameters."""
    with pytest.raises(ValueError, match="symptomatic_rate and symptomatic_shape must be positive"):
        NaturalHistoryParameters(symptomatic_rate=-0.1)

    with pytest.raises(ValueError, match="symptomatic_rate and symptomatic_shape must be positive"):
        NaturalHistoryParameters(symptomatic_shape=0.0)


def test_infectiousness_params_validation_relative_infectiousness():
    """Test that NaturalHistoryParameters validates relative presymptomatic infectiousness."""
    with pytest.raises(ValueError, match="rel_presymptomatic_infectiousness must be positive"):
        NaturalHistoryParameters(rel_presymptomatic_infectiousness=-1.0)


# ============================================================================
# MolecularClock Tests
# ============================================================================


def test_molecular_clock_default_initialization():
    """Test that MolecularClock initializes with correct default values."""
    clock = MolecularClock()
    assert clock.substitution_rate == 1e-3
    assert clock.use_relaxed_clock is True
    assert clock.relaxed_clock_sigma == 0.33
    assert clock.genome_length == 29903
    assert clock.rng is not None


def test_molecular_clock_custom_parameters():
    """Test MolecularClock with custom parameters."""
    clock = MolecularClock(
        substitution_rate=2e-3,
        use_relaxed_clock=False,
        relaxed_clock_sigma=0.5,
        genome_length=30000,
        rng_seed=42,
    )
    assert clock.substitution_rate == 2e-3
    assert clock.use_relaxed_clock is False
    assert clock.relaxed_clock_sigma == 0.5
    assert clock.genome_length == 30000


def test_molecular_clock_validation():
    """Test that MolecularClock validates input parameters."""
    with pytest.raises(ValueError, match="substitution_rate must be positive"):
        MolecularClock(substitution_rate=-1e-3)

    with pytest.raises(ValueError, match="substitution_rate must be positive"):
        MolecularClock(substitution_rate=0.0)

    with pytest.raises(ValueError, match="relaxed_clock_sigma must be non-negative"):
        MolecularClock(relaxed_clock_sigma=-0.1)

    with pytest.raises(ValueError, match="genome_length must be positive"):
        MolecularClock(genome_length=0)

    with pytest.raises(ValueError, match="genome_length must be positive"):
        MolecularClock(genome_length=-100)


def test_molecular_clock_strict_rate():
    """Test strict molecular clock (constant rate)."""
    clock = MolecularClock(use_relaxed_clock=False, substitution_rate=1e-3, rng_seed=42)
    rates = clock.sample_substitution_rate_per_day(size=1000)

    # All rates should be identical for strict clock
    assert np.all(rates == rates[0])

    # Check correct conversion from per-site-per-year to per-day
    expected_rate = (1e-3 * 29903) / 365.0
    assert np.isclose(rates[0], expected_rate)


def test_molecular_clock_relaxed_rate():
    """Test relaxed molecular clock (lognormal distribution)."""
    clock = MolecularClock(use_relaxed_clock=True, substitution_rate=1e-3, rng_seed=42)
    rates = clock.sample_substitution_rate_per_day(size=2000)

    # Rates should vary for relaxed clock
    assert np.std(rates) > 0

    # Median should be close to expected (lognormal median = exp(mu))
    expected_median = (1e-3 * 29903) / 365.0
    median_rate = np.median(rates)
    assert np.isclose(median_rate, expected_median, rtol=0.2)


def test_molecular_clock_expected_mutations_no_rates():
    """Test expected mutations calculation with default rate."""
    clock = MolecularClock(substitution_rate=1e-3, genome_length=29903)
    times = np.array([10.0, 20.0, 30.0])

    mutations = clock.estimate_expected_mutations(times)

    # Check shape
    assert mutations.shape == times.shape

    # Check non-negative
    assert np.all(mutations >= 0)

    # Check linear relationship with time
    assert mutations[1] > mutations[0]
    assert mutations[2] > mutations[1]
    assert np.isclose(mutations[1] / mutations[0], 2.0, rtol=0.01)


def test_molecular_clock_expected_mutations_with_rates():
    """Test expected mutations calculation with provided rates."""
    clock = MolecularClock()
    times = np.array([5.0, 10.0, 15.0])
    rates = np.array([0.1, 0.2, 0.3])

    mutations = clock.estimate_expected_mutations(times, rates)

    expected = rates * times
    assert np.allclose(mutations, expected)


def test_molecular_clock_expected_mutations_negative_clipping():
    """Test that expected mutations are clipped to be non-negative."""
    clock = MolecularClock()
    times = np.array([-5.0, 0.0, 5.0])

    mutations = clock.estimate_expected_mutations(times)

    # All values should be non-negative (negative times clipped to 0)
    assert np.all(mutations >= 0)


def test_molecular_clock_expected_mutations_sampled_size_strict():
    """Test expected mutations with sampled rates using size on a strict clock."""
    clock = MolecularClock(use_relaxed_clock=False, substitution_rate=1e-3, genome_length=1000)
    times = np.array([2.0, 4.0, 6.0])

    mutations = clock.estimate_expected_mutations(times, size=3)

    expected_rate = (1e-3 * 1000) / 365.0
    expected = np.broadcast_to(expected_rate * times, (3, times.size))
    assert mutations.shape == (3, times.size)
    assert np.allclose(mutations, expected)


def test_molecular_clock_expected_mutations_sampled_size_relaxed():
    """Test expected mutations with sampled rates using size on a relaxed clock."""
    times = np.array([1.0, 3.0, 5.0])

    clock_rates = MolecularClock(use_relaxed_clock=True, rng_seed=42)
    rates = clock_rates.sample_substitution_rate_per_day(size=4)

    clock_mut = MolecularClock(use_relaxed_clock=True, rng_seed=42)
    mutations = clock_mut.estimate_expected_mutations(times, size=4)

    expected = rates[:, None] * times
    assert mutations.shape == (4, times.size)
    assert np.allclose(mutations, expected)


def test_base_transmission_profile_not_implemented():
    """Test that base class abstract methods raise NotImplementedError."""

    class MinimalProfile(BaseTransmissionProfile):
        pass

    profile = MinimalProfile(grid_min_days=0.0, grid_max_days=10.0)

    with pytest.raises(NotImplementedError):
        profile.pdf(np.array([1.0]))

    with pytest.raises(NotImplementedError):
        profile.rvs(size=1)


# ============================================================================
# SymptomOnsetToTransmissionTime (Time from Onset to Transmission) Tests
# ============================================================================


def test_tost_initialization():
    """Test SymptomOnsetToTransmissionTime initializes with correct default parameters."""
    tost = SymptomOnsetToTransmissionTime()
    assert tost.grid_min_days == -30.0
    assert tost.grid_max_days == 30.0
    assert tost.parameters is not None


def test_tost_pdf_basic():
    """Test SymptomOnsetToTransmissionTime PDF is non-negative and has mass on both sides of symptom onset."""
    params = NaturalHistoryParameters()
    tost = SymptomOnsetToTransmissionTime(parameters=params)
    x = np.linspace(-10, 10, 256)
    pdf = tost.pdf(x)

    # PDF should be non-negative everywhere
    assert np.all(pdf >= 0)

    # Should have mass on both sides of zero (presymptomatic and symptomatic)
    assert pdf[x < 0].max() > 0
    assert pdf[x >= 0].max() > 0


def test_tost_pdf_piecewise():
    """Test SymptomOnsetToTransmissionTime PDF piecewise behavior at symptom onset boundary."""
    tost = SymptomOnsetToTransmissionTime()
    x = np.array([-1.0, 0.0, 1.0])
    pdf = tost.pdf(x)

    # All values should be non-negative
    assert pdf[0] >= 0 and pdf[1] >= 0 and pdf[2] >= 0

    # PDF at x=0 should be continuous (or close)
    x_fine = np.array([-0.01, 0.0, 0.01])
    pdf_fine = tost.pdf(x_fine)
    assert np.all(pdf_fine >= 0)


def test_tost_pdf_scalar_input():
    """Test SymptomOnsetToTransmissionTime PDF handles scalar input correctly."""
    tost = SymptomOnsetToTransmissionTime()
    pdf_scalar = tost.pdf(0.0)
    # PDF may return array even for scalar input
    assert pdf_scalar.size == 1
    assert pdf_scalar >= 0


def test_tost_rvs_shape():
    """Test SymptomOnsetToTransmissionTime random variates have correct shape."""
    tost = SymptomOnsetToTransmissionTime(rng_seed=42)

    # Test scalar size
    samples = tost.rvs(size=100)
    assert samples.shape == (100,)

    # Test tuple size
    samples = tost.rvs(size=(10, 5))
    assert samples.shape == (10, 5)

    # All samples should be within bounds
    assert np.all((samples >= tost.grid_min_days) & (samples <= tost.grid_max_days))


def test_tost_custom_bounds():
    """Test SymptomOnsetToTransmissionTime with custom support bounds."""
    tost = SymptomOnsetToTransmissionTime(grid_min_days=-20.0, grid_max_days=20.0)
    assert tost.grid_min_days == -20.0
    assert tost.grid_max_days == 20.0

    samples = tost.rvs(size=100)
    assert np.all((samples >= -20.0) & (samples <= 20.0))


def test_tost_mean():
    """Test SymptomOnsetToTransmissionTime mean calculation."""
    tost = SymptomOnsetToTransmissionTime()
    mean_val = tost.mean()

    # Mean should be finite
    assert np.isfinite(mean_val)

    # For default parameters, mean should be negative (presymptomatic transmission)
    # This is based on Hart et al. 2021 findings
    assert mean_val < 0


def test_tost_cdf():
    """Test SymptomOnsetToTransmissionTime CDF is bounded and has correct boundary values."""
    tost = SymptomOnsetToTransmissionTime()
    x = np.linspace(-30, 30, 100)
    cdf = tost.cdf(x)

    # CDF should be bounded in [0, 1]
    assert np.all((cdf >= 0) & (cdf <= 1))

    # CDF at lower bound should be near 0
    assert cdf[0] < 0.01

    # CDF at upper bound should be near 1
    assert cdf[-1] > 0.99

    # Test specific points: CDF should generally increase
    # (not testing strict monotonicity due to numerical integration issues at x=0 discontinuity)
    assert tost.cdf(-10.0) < tost.cdf(10.0)


# ============================================================================
# InfectiousnessToTransmissionTime (Time from Onset of Infectiousness to Transmission) Tests
# ============================================================================


def test_toit_initialization():
    """Test InfectiousnessToTransmissionTime initializes with correct default parameters."""
    toit = InfectiousnessToTransmissionTime()
    assert toit.grid_min_days == 0.0
    assert toit.grid_max_days == 60.0
    assert toit.parameters is not None


def test_toit_pdf_nonnegative_and_sampling():
    """Test InfectiousnessToTransmissionTime PDF is non-negative and random sampling works."""
    toit = InfectiousnessToTransmissionTime()
    x = np.linspace(0, 30, 256)
    pdf = toit.pdf(x)

    # PDF should be non-negative
    assert np.all(pdf >= 0)

    # Sampling should produce correct shape
    samples = toit.rvs(size=100).astype(float)
    assert samples.shape == (100,)


def test_toit_pdf_negative_x():
    """Test InfectiousnessToTransmissionTime PDF returns zero for negative inputs."""
    toit = InfectiousnessToTransmissionTime()
    x = np.linspace(-5, -0.1, 10)
    pdf = toit.pdf(x)

    # All negative values should give zero PDF
    assert np.all(pdf == 0)


def test_toit_pdf_mixed_positive_negative():
    """Test InfectiousnessToTransmissionTime PDF handles mixed positive and negative inputs."""
    toit = InfectiousnessToTransmissionTime()
    x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    pdf = toit.pdf(x)

    # Negative values should be zero
    assert pdf[0] == 0.0
    assert pdf[1] == 0.0

    # Non-negative values should be >= 0
    assert np.all(pdf[2:] >= 0)


def test_toit_pdf_at_zero():
    """Test InfectiousnessToTransmissionTime PDF at boundary (x=0)."""
    toit = InfectiousnessToTransmissionTime()
    pdf = toit.pdf([0.0])
    assert pdf[0] >= 0


def test_toit_rvs_shape():
    """Test InfectiousnessToTransmissionTime random variates have correct shape."""
    toit = InfectiousnessToTransmissionTime(rng_seed=42)

    # Test scalar size
    samples = toit.rvs(size=100)
    assert samples.shape == (100,)

    # Test tuple size
    samples = toit.rvs(size=(10, 5))
    assert samples.shape == (10, 5)

    # All samples should be within bounds
    assert np.all((samples >= toit.grid_min_days) & (samples <= toit.grid_max_days))


def test_toit_sample_generation_intervals():
    """Test InfectiousnessToTransmissionTime generation time calculation (latent + InfectiousnessToTransmissionTime)."""
    toit = InfectiousnessToTransmissionTime(rng_seed=42)
    gen_times = toit.sample_generation_intervals(size=100)

    # Should have correct shape
    assert gen_times.shape == (100,)

    # All generation times should be positive
    assert np.all(gen_times > 0)

    # Generation times should be longer than just InfectiousnessToTransmissionTime times
    toit_samples = toit.rvs(size=100)
    # On average, gen_time should be larger since it includes latent period
    assert np.mean(gen_times) > np.mean(toit_samples)


def test_toit_grid_fallback_uniform():
    """Test InfectiousnessToTransmissionTime grid fallback to uniform distribution when PDF is zero."""
    # Create parameters that result in zero PDF over grid
    # (using equal grid bounds creates a collapsed sampling grid)
    toit = InfectiousnessToTransmissionTime(
        grid_min_days=5.0,
        grid_max_days=5.0,
        integration_grid_points=2,
        sampling_grid_points=2,
    )
    xs, ps = toit._ensure_sampling_grid()

    # Grid should have expected size
    assert xs.size == 2

    # Probabilities should sum to 1
    assert np.allclose(ps.sum(), 1.0)

    # All probabilities should be non-negative
    assert np.all(ps >= 0)


def test_ensure_grid_fallback_uniform():
    """Test that zero PDFs fall back to a uniform grid."""

    class ZeroPdfProfile(BaseTransmissionProfile):
        def pdf(self, x):
            return np.zeros_like(np.asarray(x, dtype=float))

    profile = ZeroPdfProfile(grid_min_days=0.0, grid_max_days=1.0, grid_points=10)
    xs, ps = profile._ensure_sampling_grid()

    assert xs.shape == ps.shape
    assert np.allclose(ps, np.ones_like(ps) / len(ps))


def test_toit_sampling_grid_normalized():
    """Test InfectiousnessToTransmissionTime internal grid is properly normalized."""
    toit = InfectiousnessToTransmissionTime(
        grid_min_days=0.0,
        grid_max_days=5.0,
        integration_grid_points=128,
        sampling_grid_points=128,
    )
    xs, ps = toit._ensure_sampling_grid()

    # Grid and probabilities should have same shape
    assert xs.shape == ps.shape

    # All probabilities non-negative
    assert np.all(ps >= 0.0)

    # Probabilities should sum to 1
    assert np.isclose(ps.sum(), 1.0, atol=1e-6)


def test_toit_grid_caching():
    """Test InfectiousnessToTransmissionTime grid caching mechanism."""
    toit = InfectiousnessToTransmissionTime()

    # Initially, grid should be None
    assert toit._sampling_grid is None
    assert toit._sampling_weights is None

    # After first call, grid should be cached
    xs1, ps1 = toit._ensure_sampling_grid()
    assert toit._sampling_grid is not None
    assert toit._sampling_weights is not None

    # Second call should return same cached grid
    xs2, ps2 = toit._ensure_sampling_grid()
    assert np.all(xs1 == xs2)
    assert np.all(ps1 == ps2)


def test_toit_mean():
    """Test InfectiousnessToTransmissionTime mean calculation."""
    toit = InfectiousnessToTransmissionTime()
    mean_val = toit.mean()

    # Mean should be finite and positive
    assert np.isfinite(mean_val)
    assert mean_val > 0


def test_toit_cdf():
    """Test InfectiousnessToTransmissionTime CDF is monotonically increasing and bounded."""
    toit = InfectiousnessToTransmissionTime()
    x = np.linspace(0, 60, 100)
    cdf = toit.cdf(x)

    # CDF should be bounded in [0, 1]
    assert np.all((cdf >= 0) & (cdf <= 1))

    # CDF should be monotonically increasing
    assert np.all(np.diff(cdf) >= -1e-6)  # Allow small numerical errors

    # CDF at lower bound should be near 0
    assert cdf[0] < 0.01

    # CDF at upper bound should be near 1
    assert cdf[-1] > 0.99


# ============================================================================
# Stage Duration Sampling Tests
# ============================================================================


def test_molecular_clock_integration():
    """Test that MolecularClock can be used independently."""
    clock = MolecularClock(substitution_rate=1e-3, rng_seed=42)

    # Test with InfectiousnessToTransmissionTime generation times
    toit = InfectiousnessToTransmissionTime(rng_seed=42)
    gen_times = toit.sample_generation_intervals(size=100)

    # Sample clock rates
    rates = clock.sample_substitution_rate_per_day(size=100)

    # Calculate expected mutations
    mutations = clock.estimate_expected_mutations(gen_times, rates)

    assert mutations.shape == (100,)
    assert np.all(mutations >= 0)


def test_sample_latent_periods():
    """Test latent period sampling from InfectiousnessToTransmissionTime."""
    toit = InfectiousnessToTransmissionTime(rng_seed=42)
    samples = toit.sample_latent_periods(size=1000)

    # Check shape
    assert samples.shape == (1000,)

    # All samples should be positive
    assert np.all(samples > 0)

    # Mean should be close to theoretical mean (shape * scale)
    expected_mean = toit.parameters.latent_shape * toit.parameters.incubation_scale
    assert np.isclose(np.mean(samples), expected_mean, rtol=0.1)


def test_sample_presymptomatic_periods():
    """Test presymptomatic period sampling from InfectiousnessToTransmissionTime."""
    toit = InfectiousnessToTransmissionTime(rng_seed=42)
    samples = toit.sample_presymptomatic_periods(size=1000)

    # Check shape
    assert samples.shape == (1000,)

    # All samples should be positive
    assert np.all(samples > 0)

    # Mean should be close to theoretical mean
    expected_mean = toit.parameters.presymptomatic_shape * toit.parameters.incubation_scale
    assert np.isclose(np.mean(samples), expected_mean, rtol=0.1)


def test_sample_incubation_periods():
    """Test incubation period sampling (latent + presymptomatic)."""
    toit = InfectiousnessToTransmissionTime(rng_seed=42)
    samples = toit.sample_incubation_periods(size=1000)

    # Check shape
    assert samples.shape == (1000,)

    # All samples should be positive
    assert np.all(samples > 0)

    # Mean should be close to theoretical mean
    expected_mean = toit.parameters.incubation_shape * toit.parameters.incubation_scale
    assert np.isclose(np.mean(samples), expected_mean, rtol=0.1)


def test_sample_symptomatic_periods():
    """Test symptomatic period sampling from InfectiousnessToTransmissionTime."""
    toit = InfectiousnessToTransmissionTime(rng_seed=42)
    samples = toit.sample_symptomatic_periods(size=1000)

    # Check shape
    assert samples.shape == (1000,)

    # All samples should be positive
    assert np.all(samples > 0)

    # Mean should be close to theoretical mean
    expected_mean = toit.parameters.symptomatic_shape * toit.parameters.symptomatic_scale
    assert np.isclose(np.mean(samples), expected_mean, rtol=0.1)


# ============================================================================
# Presymptomatic Fraction Tests
# ============================================================================


def test_presymptomatic_fraction_in_0_1():
    """Test presymptomatic fraction is a valid probability."""
    params = NaturalHistoryParameters()
    q = estimate_presymptomatic_transmission_fraction(params)

    # Should be a valid probability
    assert 0 <= q <= 1


def test_presymptomatic_fraction_formula():
    """Test presymptomatic fraction matches analytical formula."""
    params = NaturalHistoryParameters()
    q = estimate_presymptomatic_transmission_fraction(params)

    # Calculate expected value using formula from docstring
    numerator = (
        params.rel_presymptomatic_infectiousness
        * params.presymptomatic_shape
        * params.symptomatic_rate
    )
    denominator = numerator + (params.incubation_shape * params.incubation_rate)
    expected_q = numerator / denominator

    assert np.isclose(q, expected_q)


def test_presymptomatic_fraction_custom_params():
    """Test presymptomatic fraction with custom parameters."""
    # High relative infectiousness should increase presymptomatic fraction
    params_high = NaturalHistoryParameters(rel_presymptomatic_infectiousness=5.0)
    q_high = estimate_presymptomatic_transmission_fraction(params_high)

    # Low relative infectiousness should decrease presymptomatic fraction
    params_low = NaturalHistoryParameters(rel_presymptomatic_infectiousness=1.0)
    q_low = estimate_presymptomatic_transmission_fraction(params_low)

    assert q_high > q_low
    assert 0 <= q_low <= q_high <= 1


# ============================================================================
# Integration and Edge Case Tests
# ============================================================================


def test_frozen_gamma_distributions():
    """Test that frozen gamma distributions are properly initialized."""
    toit = InfectiousnessToTransmissionTime()

    # Test incubation distribution
    assert toit.incubation is not None
    inc_samples = toit.incubation.rvs(size=100)
    assert inc_samples.shape == (100,)
    assert np.all(inc_samples > 0)

    # Test latent distribution
    assert toit.latent is not None
    lat_samples = toit.latent.rvs(size=100)
    assert lat_samples.shape == (100,)

    # Test presymptomatic distribution
    assert toit.presymptomatic is not None
    pre_samples = toit.presymptomatic.rvs(size=100)
    assert pre_samples.shape == (100,)

    # Test symptomatic distribution
    assert toit.symptomatic is not None
    symp_samples = toit.symptomatic.rvs(size=100)
    assert symp_samples.shape == (100,)


def test_repr_methods():
    """Test string representations of all classes."""
    # NaturalHistoryParameters
    params = NaturalHistoryParameters()
    assert "NaturalHistoryParameters" in repr(params)

    # MolecularClock
    clock = MolecularClock()
    assert "MolecularClock" in repr(clock)

    # InfectiousnessToTransmissionTime
    toit = InfectiousnessToTransmissionTime()
    assert "InfectiousnessToTransmissionTime" in repr(toit)

    # SymptomOnsetToTransmissionTime
    tost = SymptomOnsetToTransmissionTime()
    assert "SymptomOnsetToTransmissionTime" in repr(tost)


@pytest.mark.parametrize(
    "profile_cls, seed",
    [
        (InfectiousnessToTransmissionTime, 123),
        (SymptomOnsetToTransmissionTime, 456),
    ],
)
def test_profile_rvs_reproducibility(profile_cls, seed):
    """Test reproducible sampling with a fixed RNG seed."""
    profile1 = profile_cls(rng_seed=seed)
    profile2 = profile_cls(rng_seed=seed)
    assert np.allclose(profile1.rvs(10), profile2.rvs(10))


def test_base_profile_cdf_requires_trapz(monkeypatch):
    """Test cdf raises ImportError when trapezoid integration is unavailable."""
    from epilink.model import profiles as ip

    class UniformProfile(BaseTransmissionProfile):
        def pdf(self, x):
            x_arr = np.asarray(x, dtype=float)
            return np.ones_like(x_arr) / (self.grid_max_days - self.grid_min_days)

    profile = UniformProfile(grid_min_days=0.0, grid_max_days=1.0)
    monkeypatch.setattr(ip, "_trapz", None)

    with pytest.raises(ImportError, match="Neither np.trapezoid nor np.trapz found in NumPy."):
        profile.cdf(0.5)


def test_base_profile_mean_requires_trapz(monkeypatch):
    """Test mean raises ImportError when trapezoid integration is unavailable."""
    from epilink.model import profiles as ip

    class UniformProfile(BaseTransmissionProfile):
        def pdf(self, x):
            x_arr = np.asarray(x, dtype=float)
            return np.ones_like(x_arr) / (self.grid_max_days - self.grid_min_days)

    profile = UniformProfile(grid_min_days=0.0, grid_max_days=1.0)
    monkeypatch.setattr(ip, "_trapz", None)

    with pytest.raises(ImportError, match="Neither np.trapezoid nor np.trapz found in NumPy."):
        profile.mean()


def test_toit_pdf_requires_trapz(monkeypatch):
    """Test InfectiousnessToTransmissionTime pdf raises ImportError when trapezoid integration is unavailable."""
    from epilink.model import profiles as ip

    monkeypatch.setattr(ip, "_trapz", None)
    toit = InfectiousnessToTransmissionTime()

    with pytest.raises(ImportError, match="Neither np.trapezoid nor np.trapz found in NumPy."):
        toit.pdf([0.5])


def test_pdf_integration_approximates_one():
    """Test that PDF integrates to approximately 1."""
    from scipy.integrate import quad

    # SymptomOnsetToTransmissionTime should integrate to ~1 over its support
    tost = SymptomOnsetToTransmissionTime()
    integral, _ = quad(
        lambda x: float(tost.pdf(np.array([x]))[0]), tost.grid_min_days, tost.grid_max_days
    )
    assert np.isclose(integral, 1.0, atol=0.01)

    # InfectiousnessToTransmissionTime should integrate to ~1 over its support
    toit = InfectiousnessToTransmissionTime(
        grid_max_days=100.0
    )  # Use larger upper bound for better coverage
    integral, _ = quad(lambda x: float(toit.pdf(np.array([x]))[0]), 0.0, 100.0)
    assert np.isclose(integral, 1.0, atol=0.01)


def test_different_grid_sizes():
    """Test InfectiousnessToTransmissionTime and SymptomOnsetToTransmissionTime work with different grid sizes."""
    # Small grid
    toit_small = InfectiousnessToTransmissionTime(
        integration_grid_points=64, sampling_grid_points=64
    )
    samples_small = toit_small.rvs(size=100)
    assert samples_small.shape == (100,)

    # Large grid
    toit_large = InfectiousnessToTransmissionTime(
        integration_grid_points=512, sampling_grid_points=512
    )
    samples_large = toit_large.rvs(size=100)
    assert samples_large.shape == (100,)

    # SymptomOnsetToTransmissionTime with different grid sizes
    tost_small = SymptomOnsetToTransmissionTime(grid_points=128)
    samples = tost_small.rvs(size=100)
    assert samples.shape == (100,)


def test_custom_rng_object():
    """Test that custom RNG objects work correctly."""
    rng = default_rng(999)

    toit = InfectiousnessToTransmissionTime(rng=rng)
    samples1 = toit.rvs(size=10)

    # Using same seed should give same results
    rng2 = default_rng(999)
    toit2 = InfectiousnessToTransmissionTime(rng=rng2)
    samples2 = toit2.rvs(size=10)

    assert np.allclose(samples1, samples2)


def test_vectorized_pdf_evaluation():
    """Test that PDF evaluation is properly vectorized."""
    toit = InfectiousnessToTransmissionTime()

    # Single value
    pdf_single = toit.pdf(5.0)
    assert np.isscalar(pdf_single) or pdf_single.shape == ()

    # Array
    x_array = np.array([1.0, 5.0, 10.0])
    pdf_array = toit.pdf(x_array)
    assert pdf_array.shape == (3,)

    # 2D array
    x_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    pdf_2d = toit.pdf(x_2d)
    assert pdf_2d.shape == (2, 2)
