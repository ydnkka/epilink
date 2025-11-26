import numpy as np

from epilink import TOIT, estimate_linkage_probability, genetic_linkage_probability


def test_estimate_linkage_poisson_vs_deterministic_small_distance():
    """For small expected mutation counts, Poisson should favour low SNP distances."""

    gd = np.array([0.0, 3.0])
    td = np.array([5.0, 5.0])

    # Use a very small substitution rate to keep expected mutation counts low
    kwargs = dict(
        intermediate_generations=(0,),
        no_intermediates=3,
        subs_rate=1e-4,
        relax_rate=False,
        num_simulations=2000,
        rng_seed=123,
    )

    p_det = estimate_linkage_probability(
        genetic_distance=gd,
        temporal_distance=td,
        mutation_model="deterministic",
        **kwargs,
    )
    p_pois = estimate_linkage_probability(
        genetic_distance=gd,
        temporal_distance=td,
        mutation_model="poisson",
        **kwargs,
    )

    assert p_det.shape == p_pois.shape == (2,)
    # Under Poisson with low rate, probability at 0 mutations should be
    # comparatively higher than at larger distances.
    assert p_pois[0] > p_pois[1]


def test_genetic_linkage_probability_mutation_models_shapes():
    """Both mutation models should return correctly shaped outputs."""

    toit = TOIT(rng_seed=42)
    g = [0, 1, 2]

    pm_det = genetic_linkage_probability(
        g,
        toit=toit,
        no_intermediates=4,
        num_simulations=500,
        mutation_model="deterministic",
        mutation_tolerance=1,
        kind="normalized",
    )
    pm_pois = genetic_linkage_probability(
        g,
        toit=toit,
        no_intermediates=4,
        num_simulations=500,
        mutation_model="poisson",
        kind="normalized",
    )

    assert pm_det.shape == pm_pois.shape == (3, 5)  # m=0..4
    # Normalized kind should sum to ~1 across m
    assert np.allclose(pm_det.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(pm_pois.sum(axis=1), 1.0, atol=1e-6)

