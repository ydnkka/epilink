import numpy as np

from epilink import (
    TOIT,
    estimate_linkage_probability,
    genetic_linkage_probability,
    pairwise_linkage_probability_matrix,
    temporal_linkage_probability,
)


def test_estimate_linkage_scalar():
    p = estimate_linkage_probability(genetic_distance=2, temporal_distance=3, num_simulations=500)
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


def test_temporal_linkage_probability_shapes():
    toit = TOIT(rng_seed=123)
    out_scalar = temporal_linkage_probability(5, toit=toit, num_simulations=200)
    out_array = temporal_linkage_probability([0, -3, 7], toit=toit, num_simulations=200)

    assert isinstance(out_scalar, np.ndarray) and out_scalar.shape == (1,)
    assert out_array.shape == (3,)
    assert np.all((out_array >= 0) & (out_array <= 1))


def test_genetic_linkage_probability_kinds_and_selection():

    toit = TOIT(rng_seed=123)
    g = [0, 1]

    pm_raw = genetic_linkage_probability(
        g, toit=toit, no_intermediates=3, num_simulations=200, kind="raw"
    )
    pm_rel = genetic_linkage_probability(
        g, toit=toit, no_intermediates=3, num_simulations=200, kind="relative"
    )
    pm_norm = genetic_linkage_probability(
        g, toit=toit, no_intermediates=3, num_simulations=200, kind="normalized"
    )

    assert pm_raw.shape == pm_rel.shape == pm_norm.shape == (2, 4)  # m=0..3
