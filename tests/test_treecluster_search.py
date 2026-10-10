from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from Bio.Phylo._io import write as write_phylo
from Bio.Phylo.BaseTree import Clade, Tree

from epilink_evaluation.config import validate
from epilink_evaluation.provenance import fingerprint, read_json
from epilink_evaluation.selection.search import (
    cutoff_settings,
    initial_cutoffs,
    next_cutoffs,
)
from epilink_evaluation.workflows import baseline as workflow
from epilink_evaluation.workflows.baseline import Baseline
from epilink_evaluation.workflows.boston_scoring import operating_settings
from epilink_evaluation.workflows.settings import settings_registry


def test_integer_snp_search_rounds_initial_trials_and_exhausts_small_domains():
    settings = {"min": 0, "max": 10, "initial_points": 3, "budget": 5}
    values = initial_cutoffs(settings, integer=True)
    assert values == [0, 5, 10]
    definitions = {
        f"{method}/{value}": {
            "kind": "treecluster",
            "tree_kind": "raw",
            "pipeline": "treecluster/deterministic/raw",
            "method": method,
            "threshold_input": value,
        }
        for method in ["max_clade", "avg_clade"]
        for value in values
    }
    frame = pd.DataFrame(
        [
            {
                "split": "development",
                "seed": seed,
                "setting_id": key,
                "M0_f1": d["threshold_input"] / 10,
            }
            for key, d in definitions.items()
            for seed in [11, 12]
        ]
    )
    extra = next_cutoffs(
        settings, frame, definitions, [{"objective": "M0_f1"}], [11, 12], integer=True
    )
    assert extra == [2, 8]
    assert all(type(value) is int for value in extra)
    small = {"min": 0, "max": 2, "initial_points": 8, "budget": 28}
    assert initial_cutoffs(small, integer=True) == [0, 1, 2]
    exhausted = {
        str(x): {**next(iter(definitions.values())), "threshold_input": x}
        for x in [0, 1, 2]
    }
    assert (
        next_cutoffs(
            small, frame, exhausted, [{"objective": "M0_f1"}], [11, 12], integer=True
        )
        == []
    )
    complete = {**settings, "initial_points": 4, "budget": 11}
    values = initial_cutoffs(complete, integer=True)
    while True:
        trials = {
            str(value): {**next(iter(definitions.values())), "threshold_input": value}
            for value in values
        }
        evidence = pd.DataFrame(
            [
                {
                    "split": "development",
                    "seed": 11,
                    "setting_id": key,
                    "M0_f1": d["threshold_input"] / 10,
                }
                for key, d in trials.items()
            ]
        )
        extra = next_cutoffs(
            complete, evidence, trials, [{"objective": "M0_f1"}], [11], integer=True
        )
        if not extra:
            break
        assert not set(extra).intersection(values)
        values.extend(extra)
    assert sorted(values) == list(range(11))


def test_day_search_supports_zero_and_fractional_days_without_scaling(small_config):
    small_config["treecluster"].update(
        enabled=True,
        methods=["max_clade"],
        genetic_threshold_snps={"min": 0, "max": 10, "initial_points": 3, "budget": 5},
        temporal_threshold_days={"min": 0, "max": 15, "initial_points": 3, "budget": 5},
    )
    validate(small_config)
    definitions = {
        key: d
        for key, d in settings_registry(small_config).items()
        if d["kind"] == "treecluster"
    }
    assert {
        d["threshold_input"] for d in definitions.values() if d["tree_kind"] == "dated"
    } == {0, 7.5, 15}
    for d in definitions.values():
        if d["tree_kind"] == "dated":
            assert d["threshold"] == d["threshold_input"]
        else:
            assert (
                d["threshold"]
                == d["threshold_input"] / small_config["simulation"]["alignment_length"]
            )
    dated = {key: d for key, d in definitions.items() if d["tree_kind"] == "dated"}
    frame = pd.DataFrame(
        [
            {
                "split": "development",
                "seed": 11,
                "setting_id": key,
                "M0_f1": d["threshold_input"] / 15,
            }
            for key, d in dated.items()
        ]
    )
    settings = small_config["treecluster"]["temporal_threshold_days"]
    assert next_cutoffs(settings, frame, dated, [{"objective": "M0_f1"}], [11]) == [
        3.75,
        11.25,
    ]
    frame["split"] = "evaluation"
    with pytest.raises(ValueError, match="development"):
        next_cutoffs(settings, frame, dated, [{"objective": "M0_f1"}], [11])


def test_tree_refinement_considers_each_method_curve():
    settings = {"min": 0, "max": 20, "initial_points": 5, "budget": 8}
    objectives = {
        "max_clade": [0.1, 0.2, 0.4, 0.6, 0.9],
        "avg_clade": [0.1, 0.2, 0.9, 0.5, 0.1],
    }
    definitions, rows = {}, []
    for method, scores in objectives.items():
        for value, objective in zip(initial_cutoffs(settings, integer=True), scores):
            key = f"{method}/{value}"
            definitions[key] = {
                "kind": "treecluster",
                "tree_kind": "raw",
                "pipeline": "treecluster/deterministic/raw",
                "method": method,
                "threshold_input": value,
            }
            rows.extend(
                {
                    "split": "development",
                    "seed": seed,
                    "setting_id": key,
                    "M0_f1": objective,
                }
                for seed in [11, 12]
            )
    assert next_cutoffs(
        settings,
        pd.DataFrame(rows),
        definitions,
        [{"objective": "M0_f1"}],
        [11, 12],
        integer=True,
    ) == [2, 8, 18]


@pytest.mark.parametrize(
    "value, integer",
    [
        ({"min": -1, "max": 10}, True),
        ({"min": 0.5, "max": 10}, True),
        ({"min": 0, "max": 10.5}, True),
        ([0, 1.5], True),
        ([], False),
        ({"min": 0, "max": 0}, False),
        ({"min": 0, "max": 10, "scale": "log"}, False),
        ({"min": 0, "max": 10, "initial_points": 4, "budget": 3}, True),
    ],
)
def test_invalid_tree_cutoff_searches(value, integer):
    with pytest.raises(ValueError):
        cutoff_settings(value, integer=integer)


def prepared_test_trees(tmp_path, monkeypatch):
    """Persist synthetic input trees while exercising the real TreeCluster adapter."""
    created = []

    def prepare(config, dataset, process, dataset_id, implementation):
        source = tmp_path / "inferred" / dataset_id / process
        if not (source / "raw.nwk").exists():
            source.mkdir(parents=True, exist_ok=True)
            names = pd.read_parquet(dataset / "cases.parquet").case_id.tolist()
            lengths = {
                "raw": 0.0005 if process == "deterministic" else 0.0006,
                "dated": 4.0 if process == "deterministic" else 5.0,
            }
            for kind, length in lengths.items():
                tree = Tree(
                    root=Clade(
                        branch_length=0,
                        clades=[
                            Clade(name=name, branch_length=length) for name in names
                        ],
                    )
                )
                write_phylo(tree, source / f"{kind}.nwk", "newick")
            created.append((dataset_id, process))
        return source

    monkeypatch.setattr(workflow, "prepare_phylogeny", prepare)
    return created


def test_adaptive_treecluster_real_adapter_resume_freeze_and_boston_units(
    small_config, prepare_diagnostics, tmp_path, monkeypatch
):
    small_config["scorers"] = ["GD_D", "GD_S"]
    small_config["splits"]["development"] = [72001, 72002]
    small_config["treecluster"].update(
        enabled=True,
        methods=["max_clade", "avg_clade"],
        genetic_threshold_snps={"min": 0, "max": 10, "initial_points": 3, "budget": 5},
        temporal_threshold_days={"min": 0, "max": 15, "initial_points": 3, "budget": 5},
    )
    prepare_diagnostics(small_config)
    created = prepared_test_trees(tmp_path, monkeypatch)
    original, calls = workflow.treecluster, []

    def cluster(tree_path, cases, method, threshold, config, directory):
        calls.append(
            (
                "evaluation" if "evaluation" in directory.parts else "development",
                tree_path.name,
                method,
                threshold,
            )
        )
        return original(tree_path, cases, method, threshold, config, directory)

    monkeypatch.setattr(workflow, "treecluster", cluster)
    baseline = Baseline(small_config)
    frozen = baseline.select()
    for kind, units in (("raw", "snps"), ("dated", "days")):
        trials = {
            key: d
            for key, d in baseline.definitions.items()
            if d["kind"] == "treecluster" and d["tree_kind"] == kind
        }
        assert len(trials) == 5 * 2 * 2  # Cutoffs × processes × methods.
        assert len({d["threshold_input"] for d in trials.values()}) == 5
        saved = read_json(
            baseline.directory / f"development/treecluster_search/{kind}/search.json"
        )
        assert saved["status"] == "complete" and saved["units"] == units
        assert len(saved["evaluated_cutoffs"]) == 5
        for d in trials.values():
            if kind == "raw":
                assert type(d["threshold_input"]) is int
                assert (
                    d["threshold"]
                    == d["threshold_input"]
                    / small_config["simulation"]["alignment_length"]
                )
            else:
                assert d["threshold"] == d["threshold_input"]
    assert (
        len(created) == 4
    )  # One inference per development dataset/process, shared by kinds and trials.
    expected_calls = Counter(
        (f"{d['tree_kind']}.nwk", d["method"], d["threshold"])
        for d in baseline.definitions.values()
        if d["kind"] == "treecluster"
        for _ in small_config["splits"]["development"]
    )
    assert (
        Counter(call[1:] for call in calls if call[0] == "development")
        == expected_calls
    )
    snapshot = deepcopy(baseline.definitions)

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed TreeCluster trials must be reused")

    with monkeypatch.context() as patch:
        patch.setattr(workflow, "treecluster", forbidden)
        resumed = Baseline(deepcopy(small_config))
        assert resumed.definitions == snapshot
        assert resumed.select() == frozen
    selected = {
        p["setting_id"]: p["definition"]
        for p in frozen["operating_points"]
        if p["status"] == "selected"
    }
    reference = SimpleNamespace(
        config=baseline.config,
        selected=selected,
        frozen=frozen,
        identity={"selection_fingerprint": fingerprint(frozen)},
    )
    transferred, _ = operating_settings(
        reference, ["GD_D", "GD_S"], include_trees=True, target_alignment_length=29903
    )
    for d in transferred.values():
        if d["kind"] == "treecluster":
            source = selected[d["baseline_setting_id"]]
            if d["tree_kind"] == "raw":
                assert d["threshold_snps"] == source["threshold_input"]
                assert d["threshold"] == source["threshold_input"] / 29903
            else:
                assert d["threshold"] == source["threshold_input"]
    assert resumed.evaluate()
    assert resumed.definitions == snapshot
    assert (
        len(created) == 6
    )  # Only the new evaluation dataset requires additional inference.
    evaluation_calls = [call for call in calls if call[0] == "evaluation"]
    assert len(evaluation_calls) == sum(
        d["kind"] == "treecluster" for d in selected.values()
    )
    assert Counter(call[1:] for call in evaluation_calls) == Counter(
        (f"{d['tree_kind']}.nwk", d["method"], d["threshold"])
        for d in selected.values()
        if d["kind"] == "treecluster"
    )
    assert read_json(resumed.directory / "evaluation/selection_used.json") == frozen


def test_tree_refinement_failure_is_visible_and_resumes_successful_trials(
    small_config, prepare_diagnostics, tmp_path, monkeypatch
):
    small_config["scorers"] = ["GD_D"]
    small_config["treecluster"].update(
        enabled=True,
        methods=["max_clade"],
        genetic_threshold_snps={"min": 0, "max": 4, "initial_points": 2, "budget": 3},
        temporal_threshold_days={"min": 0, "max": 4, "initial_points": 2, "budget": 3},
    )
    prepare_diagnostics(small_config)
    prepared_test_trees(tmp_path, monkeypatch)
    fail, calls = True, []

    def cluster(tree_path, cases, method, threshold, config, directory):
        calls.append((tree_path.name, threshold))
        if (
            fail
            and tree_path.name == "raw.nwk"
            and round(threshold * small_config["simulation"]["alignment_length"]) == 2
        ):
            raise ValueError("synthetic refinement failure")
        return np.arange(len(cases)), {"synthetic_adapter": True}

    monkeypatch.setattr(workflow, "treecluster", cluster)
    baseline = Baseline(small_config)
    with pytest.raises(RuntimeError, match="partial"):
        baseline.select()
    assert not baseline._evaluation_released
    assert not (baseline.directory / "selection/operating_points.json").exists()
    assert not (baseline.directory / "evaluation").exists()
    assert (
        read_json(
            baseline.directory / "development/treecluster_search/raw/search.json"
        )["status"]
        == "running"
    )
    initial_calls = [
        x for x in calls if x[0] == "raw.nwk" and round(x[1] * 5000) in {0, 4}
    ]
    fail = False
    resumed = Baseline(deepcopy(small_config))
    resumed.select()
    assert [
        x for x in calls if x[0] == "raw.nwk" and round(x[1] * 5000) in {0, 4}
    ] == initial_calls
    assert (
        read_json(
            resumed.directory / "development/treecluster_search/dated/search.json"
        )["status"]
        == "complete"
    )
