import numpy as np
import pytest

from epilink import (
    TOIT,
    estimate_linkage_probability,
    genetic_linkage_probability,
)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        estimate_linkage_probability(genetic_distance=[0, 1], temporal_distance=[0])


def test_intermediate_generations_bounds():
    with pytest.raises(ValueError):
        estimate_linkage_probability(
            genetic_distance=1,
            temporal_distance=1,
            intermediate_generations=(11,),
            no_intermediates=10,
        )


def test_numba_disabled_coverage(monkeypatch):
    # Disable JIT to execute Python path of kernels for coverage
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    out = estimate_linkage_probability(
        genetic_distance=[0, 1, 2], temporal_distance=[0, 1, 2], num_simulations=200
    )
    assert out.shape == (3,)


def test_numba_disabled_kernel_paths(monkeypatch):
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    out = estimate_linkage_probability(
        genetic_distance=np.array([0, 1, 2], dtype=float),
        temporal_distance=np.array([0, 1, 2], dtype=float),
        num_simulations=200,
    )
    assert out.shape == (3,)
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_genetic_linkage_probability_errors():
    toit = TOIT(rng_seed=123)
    with pytest.raises(ValueError):
        genetic_linkage_probability(
            [0], toit=toit, no_intermediates=2, intermediate_generations=(3,), kind="raw"
        )
    with pytest.raises(ValueError):
        genetic_linkage_probability([0], toit=toit, no_intermediates=2, kind="bad-kind")


def test_estimate_linkage_probability_errors():
    with pytest.raises(ValueError):
        estimate_linkage_probability([0, 1], [2], num_simulations=50)
    with pytest.raises(ValueError):
        estimate_linkage_probability(
            [0], [2], intermediate_generations=(999,), no_intermediates=3, num_simulations=50
        )


def test_estimate_linkage_probability_basic():
    p = estimate_linkage_probability(
        1, 5, intermediate_generations=(0,), num_simulations=200, rng_seed=1
    )
    assert isinstance(p, float)
    arr = estimate_linkage_probability(
        [0, 1], [2, 3], intermediate_generations=(0,), num_simulations=200, rng_seed=1
    )
    assert isinstance(arr, np.ndarray) and arr.shape == (2,)
