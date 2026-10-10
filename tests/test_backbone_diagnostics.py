"""Check backbone summaries against explicit transmission structures."""

import networkx as nx
import numpy as np
import pytest
from scipy import stats

from epilink_evaluation.diagnostics import backbone
from epilink_evaluation.truth import relationship_table


@pytest.mark.parametrize(
    "counts", [[], [[0, 1]], [0, -1], [0, 1.5], [np.nan], [np.inf]]
)
def test_invalid_offspring_counts(counts):
    with pytest.raises(ValueError, match="Offspring counts"):
        backbone.offspring_statistics(counts)


def test_inclusive_poisson_boundary_and_zero_offspring_denominator():
    counts = np.array([0] * 8 + [4, 5])
    summary, flags, curve, _ = backbone.offspring_statistics(counts)
    assert summary["mean_offspring"] == 0.9
    assert summary["poisson_percentile"] == stats.poisson.ppf(0.99, 0.9) == 4
    assert summary["superspreading_operator"] == ">="
    assert flags.tolist() == [False] * 8 + [True, True]
    assert summary["superspreader_fraction"] == 0.2
    assert summary["superspreader_transmission_fraction"] == 1
    assert summary["fraction_for_80_percent"] == 0.2
    assert summary["zero_offspring_fraction"] == 0.8
    np.testing.assert_allclose(curve.transmission_fraction[:3], [0, 5 / 9, 1])


@pytest.mark.parametrize(
    "tree",
    [
        nx.path_graph(7, create_using=nx.DiGraph),
        nx.balanced_tree(2, 3, create_using=nx.DiGraph),
        nx.DiGraph([(0, i) for i in range(1, 8)]),
        nx.DiGraph([(0, 1), (0, 2), (3, 4)]),
    ],
)
def test_tree_summaries_match_full_relationship_truth(tree):
    summary, tables = backbone.backbone_diagnostics(tree)
    truth = relationship_table(tree)
    assert summary["n_M0_pairs"] == truth.M.eq(0).sum()
    assert summary["n_shared_infector_pairs"] == (truth.CA.eq(1) & truth.M.eq(0)).sum()
    assert summary["n_transmissions"] == len(tree) - summary["n_roots"]
    assert summary["mean_offspring"] == pytest.approx(
        tree.number_of_edges() / len(tree)
    )
    nodes = tables["nodes.parquet"]
    for row in nodes.to_dict("records"):
        node = list(tree)[row["node_index"]]
        assert row["offspring"] == tree.out_degree(node)
        assert row["descendant_count"] == len(nx.descendants(tree, node))
        assert row["depth"] == nx.shortest_path_length(
            tree, int(row["root_case_id"]), node
        )
    assert tables["generations.csv"].n_cases.sum() == len(tree)
    assert tables["components.csv"].n_cases.sum() == len(tree)
    assert tables["offspring.csv"].n_cases.sum() == len(tree)


def test_no_transmissions_and_poisson_limit_are_explicit():
    tree = nx.DiGraph()
    tree.add_nodes_from(["a", "b", "c"])
    summary, tables = backbone.backbone_diagnostics(tree)
    assert summary["n_roots"] == 3
    assert summary["fit_method"] == "degenerate"
    assert summary["n_superspreaders"] == 0
    assert not tables["nodes.parquet"].is_superspreader.any()
    assert np.isnan(summary["fraction_for_80_percent"])
    assert np.isnan(summary["superspreader_transmission_fraction"])
    summary, _, _, _ = backbone.offspring_statistics([0, 1, 1, 1])
    assert summary["fit_method"] == "poisson_limit"
    assert np.isnan(summary["dispersion_k"])


def test_negative_binomial_fit_recovers_known_dispersion():
    counts = stats.nbinom.rvs(0.4, 0.4 / 2.4, size=8000, random_state=123)
    summary, _, _, _ = backbone.offspring_statistics(counts)
    assert summary["fit_method"] == "mle"
    assert summary["mean_offspring"] == np.mean(counts)
    assert summary["dispersion_k"] == pytest.approx(0.4, abs=0.04)


def test_moments_fallback_and_reproducible_resampling(monkeypatch):
    def failed_fit(*args, **kwargs):
        raise RuntimeError("deliberate optimiser failure")

    monkeypatch.setattr(backbone.optimize, "minimize_scalar", failed_fit)
    counts = [0] * 8 + [4, 5]
    settings = {"bootstrap_replicates": 20, "bootstrap_seed": 12}
    first = backbone.offspring_statistics(counts, settings)
    second = backbone.offspring_statistics(counts, settings)
    summary = first[0]
    assert summary["fit_method"] == "moments_fallback"
    assert summary["dispersion_k"] == pytest.approx(
        0.9**2 / (np.var(counts, ddof=1) - 0.9)
    )
    assert summary["bootstrap"] == second[0]["bootstrap"]
    assert first[3].equals(second[3])
    assert summary["bootstrap"]["completed"] == 20
    assert summary["bootstrap"]["mean_offspring_kept"] == 20


@pytest.mark.parametrize(
    "settings",
    [
        {"superspreading_quantile": 1},
        {"superspreading_quantile": np.nan},
        {"bootstrap_replicates": -1},
        {"bootstrap_seed": True},
        {"unknown": 1},
    ],
)
def test_settings_validation(settings):
    with pytest.raises(ValueError):
        backbone.backbone_settings(settings)
