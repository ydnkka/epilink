import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from epilink_evaluation.clusterers import components, leiden
from epilink_evaluation.graphs import build_graph, selected_pairs
from epilink_evaluation.scorers import SCORERS
from epilink_evaluation.scorers.logistic import fit_logistic, predict_logistic, training_cells


def test_compressed_logistic_matches_uncompressed_training():
    x = np.repeat([[0, 0], [1, 2], [2, 1], [3, 4]], [10, 8, 6, 12], axis=0)
    y = np.concatenate([np.r_[np.ones(p), np.zeros(n - p)] for p, n in [(7, 10), (3, 8), (2, 6), (1, 12)]])
    observations = pd.DataFrame({"pair_id": np.arange(len(x)), "GD_deterministic": x[:, 0], "TD": x[:, 1]})
    truth = pd.DataFrame({"pair_id": observations.pair_id, "M": np.where(y, 0, 1)})
    cells = training_cells(observations, truth, "deterministic")
    model = fit_logistic(cells, regularization=0.7)
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(C=0.7, solver="lbfgs", tol=1e-9, max_iter=2000).fit(scaler.transform(x), y)
    heldout = pd.DataFrame({"GD_deterministic": [0, 2, 4, 100], "TD": [1, 0, 5, 200]})
    expected = classifier.predict_proba(scaler.transform(heldout.to_numpy()))[:, 1]
    np.testing.assert_allclose(predict_logistic(heldout, "deterministic", model), expected, atol=1e-7)
    np.testing.assert_allclose(model["mean"], x.mean(axis=0))
    assert model["training_pairs"] == len(x)
    assert model["training_prevalence"] == pytest.approx(y.mean())
    with pytest.raises(ValueError, match="not aligned"):
        training_cells(observations, truth.iloc[::-1], "deterministic")


def test_inclusive_thresholds_native_weights_and_isolates():
    a, b = np.triu_indices(4, 1)
    observations = pd.DataFrame({"a": a, "b": b})
    values = np.array([2.0, 0.5, 0, 0, 0, 0])
    spec = SCORERS["EDD"].spec
    binary = build_graph(observations, 4, values, spec, 0)
    native = build_graph(observations, 4, values, spec, 0, policy="native")
    assert binary.ecount() == 6
    assert native.ecount() == 2
    assert native.es["weight"] == [2.0, 0.5]  # Compatibility is not clipped to one.
    labels, _ = components(native)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]
    np.testing.assert_array_equal(selected_pairs(values, spec, 0.5), [True, True, False, False, False, False])
    empty = build_graph(observations, 4, values, spec, None, empty=True)
    assert len(set(components(empty)[0])) == 4
    assert len(set(leiden(empty, 0.1, "CPM", 2, 123)[0])) == 4
    genetic = SCORERS["GD_D"].spec
    np.testing.assert_array_equal(selected_pairs([0, 1, 1, 2], genetic, 1), [True, True, True, False])
    with pytest.raises(ValueError, match="Native weights"):
        build_graph(observations, 4, values, genetic, 1, policy="native")


def test_leiden_restarts_are_reproducible_and_selected_by_objective():
    a, b = np.triu_indices(6, 1)
    observations = pd.DataFrame({"a": a, "b": b})
    values = ((a < 3) == (b < 3)).astype(float)
    graph = build_graph(observations, 6, values, SCORERS["EDD"].spec, 0.5)
    labels, metadata = leiden(graph, 0.1, "CPM", 4, 123)
    repeated, repeated_metadata = leiden(graph, 0.1, "CPM", 4, 123)
    np.testing.assert_array_equal(labels[:, None] == labels, repeated[:, None] == repeated)
    np.testing.assert_array_equal(labels[:, None] == labels, (np.arange(6) < 3)[:, None] == (np.arange(6) < 3))
    assert metadata == repeated_metadata
    assert metadata["quality"] == max(metadata["restart_qualities"])
