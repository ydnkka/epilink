import numpy as np
import pytest
from epilink import estimate_linkage_probability

def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        estimate_linkage_probability(genetic_distance=[0,1], sampling_interval=[0])

def test_intermediate_generations_bounds():
    with pytest.raises(ValueError):
        estimate_linkage_probability(genetic_distance=1, sampling_interval=1, intermediate_generations=(11,), no_intermediates=10)

def test_numba_disabled_coverage(monkeypatch):
    # Disable JIT to execute Python path of kernels for coverage
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    out = estimate_linkage_probability(genetic_distance=[0,1,2], sampling_interval=[0,1,2], num_simulations=200)
    assert out.shape == (3,)

def test_numba_disabled_kernel_paths(monkeypatch):
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    out = estimate_linkage_probability(
        genetic_distance=np.array([0, 1, 2], dtype=float),
        sampling_interval=np.array([0, 1, 2], dtype=float),
        num_simulations=200,
    )
    assert out.shape == (3,)
    assert np.all((out >= 0.0) & (out <= 1.0))