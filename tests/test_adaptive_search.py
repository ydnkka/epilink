from copy import deepcopy

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from epilink_evaluation.clusterers import components
from epilink_evaluation.graphs import build_graph
from epilink_evaluation.metrics.component_sweep import ComponentSweep
from epilink_evaluation.metrics.partitions import PartitionEvaluator
from epilink_evaluation.provenance import read_json
from epilink_evaluation.scorers import SCORERS
from epilink_evaluation.selection.operating import select_operating_points
from epilink_evaluation.selection.search import initial_resolutions, next_resolutions, resolution_settings
from epilink_evaluation.truth.relationships import TreeIndex
from epilink_evaluation.workflows import baseline as workflow
from epilink_evaluation.workflows.baseline import Baseline
from epilink_evaluation.workflows.settings import settings_registry


@pytest.mark.parametrize("name", ["LOGIT_D", "GD_D"])
def test_incremental_components_match_brute_force_with_ties_and_foreign_cutoffs(name):
    tree = nx.DiGraph([(0, 1), (1, 2), (1, 3), (4, 5)])
    a, b = np.triu_indices(6, 1)
    truth = TreeIndex(tree).classify(a, b)
    truth.insert(0, "pair_id", np.arange(len(a)))
    observations = pd.DataFrame({"pair_id": truth.pair_id, "a": a, "b": b})
    cases = pd.DataFrame({"case_id": [str(i) for i in range(6)]})
    evaluator = PartitionEvaluator(observations, cases, truth)
    values = np.array([0.8, 0.1, 0.8, 0, 0, 0.8, 0.4, 0.1, 0, 0.4, 0.1, 0, 0.4, 0.4, 0.8])
    spec = SCORERS[name].spec
    sweep = ComponentSweep(evaluator, values, spec.higher_is_better)
    queries = [None, *sorted([0, 0.1, 0.25, 0.4, 0.6, 0.8, 1], reverse=spec.higher_is_better)]
    for cutoff in queries:
        sweep.advance(cutoff)
        labels, summary, clusters = sweep.evaluate()
        graph = build_graph(observations, 6, values, spec, cutoff, cutoff is None)
        expected_labels, _ = components(graph)
        expected, expected_clusters = evaluator.evaluate(expected_labels)
        np.testing.assert_array_equal(labels[:, None] == labels, expected_labels[:, None] == expected_labels)
        pd.testing.assert_series_equal(pd.Series(summary), pd.Series(expected), check_names=False)
        pd.testing.assert_frame_equal(clusters, expected_clusters)
        assert sweep.position == graph.ecount()
    assert sweep.revision == 5


def test_pairwise_and_components_select_independent_values_on_shared_candidates(small_config):
    small_config["scorers"] = ["LOGIT_D"]
    definitions = settings_registry(small_config, {"LOGIT_D": [0.2, 0.8]})
    rows = [{"split": "development", "seed": seed, "pipeline": d["pipeline"], "setting_id": key,
             "M0_f1": 0 if d["empty"] else float(d["threshold"] == (0.8 if d["kind"] == "pairwise" else 0.2))}
            for key, d in definitions.items() for seed in [11, 12]]
    points = select_operating_points(pd.DataFrame(rows), definitions, [{"name": "f1", "objective": "M0_f1"}], [11, 12])
    assert {p["pipeline"]: p["definition"]["threshold"] for p in points} == {"pairwise/LOGIT_D": 0.8, "components/LOGIT_D": 0.2}


def test_resolution_refinement_finds_between_initial_points_and_enforces_budget():
    settings = {"min": 1, "max": 16, "initial_points": 3, "budget": 5}
    values = initial_resolutions(settings)
    assert values == [1, 4, 16]
    def evidence(values):
        definitions = {str(x): {"kind": "leiden", "pipeline": "leiden/test", "resolution": x} for x in values}
        frame = pd.DataFrame([{"split": "development", "seed": seed, "setting_id": str(x),
                               "M0_f1": 1 - abs(np.log2(x) - 3) / 5}
                              for x in values for seed in [11, 12]])
        return definitions, frame
    criteria = [{"name": "f1", "objective": "M0_f1"}]
    definitions, frame = evidence(values)
    extra = next_resolutions(settings, frame, definitions, criteria, [11, 12])
    np.testing.assert_allclose(extra, [2, 8])
    definitions, frame = evidence(values + extra)
    assert next_resolutions(settings, frame, definitions, criteria, [11, 12]) == []
    assert frame.groupby("setting_id").M0_f1.mean().idxmax() == str(extra[-1])
    frame["split"] = "evaluation"
    with pytest.raises(ValueError, match="development"):
        next_resolutions({**settings, "budget": 6}, frame, definitions, criteria, [11, 12])


@pytest.mark.parametrize("settings", [
    {"min": 0, "max": 1}, {"min": 1, "max": 1}, {"min": 1, "max": np.inf},
    {"min": 0.1, "max": 1, "initial_points": 4, "budget": 3},
    {"min": 0.1, "max": 1, "scale": "invalid"},
])
def test_invalid_resolution_search_settings(settings):
    with pytest.raises(ValueError):
        resolution_settings(settings)


def test_refinement_uses_constraints_in_every_seed_and_requires_complete_evidence():
    settings = {"min": 1, "max": 17, "initial_points": 5, "budget": 8, "scale": "linear"}
    definitions = {str(x): {"kind": "leiden", "pipeline": "leiden/test", "resolution": x}
                   for x in initial_resolutions(settings)}
    objectives = {1: 0.99, 5: 0.1, 9: 0.2, 13: 0.95, 17: 0.9}
    frame = pd.DataFrame([
        {"split": "development", "seed": seed, "setting_id": key,
         "M0_f1": objectives[d["resolution"]],
         "M0_precision": 0.1 if seed == 12 and d["resolution"] in {1, 13} else 0.9}
        for key, d in definitions.items() for seed in [11, 12]
    ])
    criteria = [{"name": "precision", "objective": "M0_f1", "constraints": {"M0_precision": {"min": 0.5}}}]
    assert next_resolutions(settings, frame, definitions, criteria, [11, 12]) == [3, 7, 15]
    with pytest.raises(ValueError, match="Incomplete"):
        next_resolutions(settings, frame.iloc[1:], definitions, criteria, [11, 12])


def test_adaptive_development_restores_candidates_and_freezes_replay(
    small_config, prepare_diagnostics, monkeypatch
):
    small_config["scorers"] = ["GD_D"]
    small_config["clustering"]["algorithms"] = ["components", "leiden"]
    small_config["splits"]["development"] = [72001, 72002]
    small_config["clustering"]["leiden"]["resolutions"] = {"min": 0.01, "max": 1, "initial_points": 3, "budget": 5}
    prepare_diagnostics(small_config)
    baseline = Baseline(small_config)
    frozen = baseline.select()
    resolutions = {d["resolution"] for d in baseline.definitions.values() if d["kind"] == "leiden"}
    assert len(resolutions) == 5
    assert {0.01, 1} <= resolutions
    for seed in small_config["splits"]["development"]:
        source = baseline.directory / "development" / f"seed_{seed}" / "clusters/component_sweeps/GD_D"
        details = read_json(source / "algorithm.json")
        assert details["n_partitions"] <= len(baseline.tree)
        assert details["n_partitions"] <= details["n_candidates"]
        assert read_json(source / "manifest.json")["status"] == "complete"
    snapshot = deepcopy(baseline.definitions)
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed searches must reuse their checkpoints")
    with monkeypatch.context() as patch:
        patch.setattr(workflow, "ComponentSweep", forbidden)
        patch.setattr(workflow, "leiden", forbidden)
        resumed = Baseline(deepcopy(small_config))
        assert resumed.definitions == snapshot
        assert resumed.select() == frozen
    assert resumed.evaluate()
    assert resumed.definitions == snapshot
    assert read_json(resumed.directory / "evaluation/selection_used.json") == frozen
    assert read_json(resumed.directory / "development/resolution_search/search.json")["status"] == "complete"


def test_adaptive_search_failure_resumes_successful_initial_trials(
    small_config, prepare_diagnostics, monkeypatch
):
    small_config["scorers"] = ["GD_D"]
    small_config["clustering"]["algorithms"] = ["leiden"]
    settings = {"min": 0.01, "max": 1, "initial_points": 3, "budget": 5}
    small_config["clustering"]["leiden"]["resolutions"] = settings
    prepare_diagnostics(small_config)
    original = workflow.leiden
    fail, calls = True, []
    def cluster(graph, resolution, objective, restarts, seed):
        calls.append(resolution)
        if fail and not any(np.isclose(resolution, x) for x in initial_resolutions(settings)):
            raise RuntimeError("refinement failure")
        return original(graph, resolution, objective, restarts, seed)
    monkeypatch.setattr(workflow, "leiden", cluster)
    baseline = Baseline(small_config)
    assert not baseline.clusters()
    initial_calls = [x for x in calls if any(np.isclose(x, y) for y in initial_resolutions(settings))]
    fail = False
    resumed = Baseline(deepcopy(small_config))
    assert resumed.clusters()
    assert [x for x in calls if any(np.isclose(x, y) for y in initial_resolutions(settings))] == initial_calls
