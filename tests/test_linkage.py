import numpy as np

from epilink import estimate_linkage_probability, pairwise_linkage_probability_matrix


def test_estimate_linkage_scalar():
    p = estimate_linkage_probability(genetic_distance=2, sampling_interval=3, num_simulations=500)
    assert 0.0 <= p <= 1.0

def test_estimate_linkage_vectorized():
    gd = np.array([0, 1, 2, 3], dtype=float)
    td = np.array([0, 2, 5, 10], dtype=float)
    out = estimate_linkage_probability(gd, td, num_simulations=500)
    assert out.shape == (4,)
    assert np.all((0.0 <= out) & (out <= 1.0))

def test_pairwise_matrix():
    gd = np.arange(0, 4)
    td = np.array([0, 2, 5])
    mat = pairwise_linkage_probability_matrix(gd, td, num_simulations=500)
    assert mat.shape == (gd.size, td.size)
    assert np.all((0.0 <= mat) & (mat <= 1.0))
