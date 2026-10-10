"""The manuscript surface must pass SNP and day axes to the matching EpiLink models."""

import numpy as np

from evaluation.results.fig03 import score_surfaces


def test_primary_compatibility_surface_keeps_processes_and_axes_separate():
    class Model:
        def __init__(self, offset):
            self.offset = offset

        def score_target(self, *, sample_time_difference, genetic_distance):
            return self.offset + sample_time_difference + genetic_distance / 10

    class Context:
        def epilink(self, process):
            return Model({"deterministic": 0, "stochastic": 10}[process])

    surfaces = score_surfaces(Context(), np.array([0, 1, 2]), np.array([0, 5]))
    np.testing.assert_allclose(
        surfaces["deterministic"], [[0, 0.1, 0.2], [5, 5.1, 5.2]]
    )
    np.testing.assert_allclose(surfaces["stochastic"], surfaces["deterministic"] + 10)
