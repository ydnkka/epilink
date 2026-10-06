from copy import deepcopy
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from Bio import Phylo

from epilink_evaluation.provenance import (
    digest_file,
    fingerprint,
    read_json,
    valid_artifact,
)
from epilink_evaluation.reporting.diagnostics import render_report
from epilink_evaluation.schemas import ENDPOINTS
from epilink_evaluation.workflows import diagnostics as workflow
from epilink_evaluation.workflows.diagnostics import Diagnostics


@pytest.fixture
def diagnostics_config(small_config, tmp_path):
    config = deepcopy(small_config)
    config["name"] = "synthetic_diagnostics"
    config["experiment_root"] = str(tmp_path / "shared_experiment")
    config["output_directory"] = str(tmp_path / "diagnostics")
    config["splits"]["development"] = [62001, 62002]
    config["simulation"]["fraction_sampled"] = 1.0
    config["diagnostics"] = {
        "leiden": {
            "objective": "CPM",
            "resolutions": [0.2, 0.8],
            "restarts": 2,
            "seed": 65001,
        },
        "treecluster": {
            "enabled": False,
            "methods": ["max_clade"],
            "threshold_hops": [0, 2],
            "command_timeout_seconds": 30,
            "executables": {"treecluster": "TreeCluster.py"},
        },
    }
    # This fixture originates in a baseline config; diagnostics needs no scoring policy.
    for key in (
        "scorers",
        "scorer",
        "inference",
        "pairwise",
        "clustering",
        "treecluster",
        "selection",
    ):
        config.pop(key, None)
    return config


def _completion_signature(diagnostics, marker):
    return {
        "experiment": diagnostics.exp.identity,
        "diagnostics": marker["fingerprint"],
        "datasets": marker["datasets"],
    }


def _timestamps(root):
    return {str(p): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


def test_development_only_preparation_and_complete_marker(diagnostics_config):
    diagnostics = Diagnostics(diagnostics_config)
    assert diagnostics.run("prepare")
    seeds = diagnostics_config["splits"]["development"]
    assert set(diagnostics.datasets) == set(seeds)
    assert not (diagnostics.exp.directory / "diagnostics.json").exists()
    assert not (diagnostics.directory / "completion/manifest.json").exists()
    for seed in diagnostics_config["splits"]["evaluation"]:
        with pytest.raises(ValueError):
            diagnostics.dataset(seed)
        with pytest.raises((ValueError, RuntimeError)):
            diagnostics.exp.dataset(seed)
    assert diagnostics.run("all")
    marker = read_json(diagnostics.exp.directory / "diagnostics.json")
    assert marker == {
        "run_directory": str(diagnostics.directory),
        "fingerprint": fingerprint(diagnostics.signature),
        "experiment": diagnostics.exp.identity,
        "datasets": {str(s): diagnostics.dataset(s).name for s in seeds},
        "status": "complete",
    }
    assert valid_artifact(
        diagnostics.directory / "completion", _completion_signature(diagnostics, marker)
    )
    coverage = read_json(diagnostics.directory / "completion/coverage.json")
    assert coverage["stages"] == ["backbone", "observations", "graphs"]
    for directory, saved in coverage["artifacts"].items():
        assert (
            digest_file(Path(directory) / "manifest.json") == saved["manifest_sha256"]
        )
        for name, checksum in saved["files"].items():
            assert digest_file(Path(directory) / name) == checksum
    assert diagnostics.signature["experiment"] == diagnostics.exp.identity
    for root in (diagnostics.root, diagnostics.exp.root):
        assert not list(root.rglob("models.json"))
        assert not list(root.rglob("scores.parquet"))
        assert not list(root.rglob("raw.nwk"))
        assert not list(root.rglob("dated.nwk"))
        assert not list(root.rglob("heldout_access.json"))
    observed = {p.parent for p in diagnostics.exp.root.rglob("pairs.parquet")}
    assert observed == set(diagnostics.datasets.values())
    report = (diagnostics.directory / "report.md").read_text()
    assert "not a population ceiling" in report
    assert "disabled by configuration" in report
    assert "development seed 62001" in report


def test_oracles_deduplicate_and_stages_reuse_checkpoints(
    diagnostics_config, monkeypatch
):
    diagnostics = Diagnostics(diagnostics_config)
    assert diagnostics.run("all")
    index = read_json(diagnostics.directory / "graphs/index.json")
    assert len(index["records"]) == 2 * len(ENDPOINTS) * 3
    assert len({r["source"] for r in index["records"]}) == len(ENDPOINTS)
    assert len({r["artifact"] for r in index["records"]}) == len(ENDPOINTS) * 3
    assert len({r["control_id"] for r in index["records"]}) == 1
    for record in index["records"]:
        assert "process" not in record
        artifact = Path(record["artifact"])
        assert len(pd.read_parquet(artifact / "memberships.parquet")) == len(
            diagnostics.tree
        )
        metrics = read_json(artifact / "metrics.json")
        assert all(f"{endpoint}_precision" in metrics for endpoint in ENDPOINTS)
        if record["algorithm"] == "components" and record["endpoint"] == "M0":
            graph = read_json(Path(record["source"]) / "summary.json")
            assert metrics["within_pairs"] > graph["n_target_edges"]
        if record["algorithm"] == "leiden":
            details = read_json(artifact / "algorithm.json")
            assert details["restart_selection"] == "maximum objective"
            assert details["quality"] == max(details["restart_qualities"])
    frame = pd.read_csv(diagnostics.directory / "graphs/metrics.csv")
    assert (frame.loc[frame.algorithm.eq("components"), "within_pairs"] == 105).all()
    summary = pd.read_csv(diagnostics.directory / "observations/summary.csv")
    aggregate = pd.read_csv(
        diagnostics.directory / "observations/summary_aggregate.csv"
    )
    keys = ["process", "feature_set", "endpoint"]
    expected = summary.groupby(keys).minimum_feature_only_misclassification_rate.mean()
    actual = aggregate.set_index(keys).minimum_feature_only_misclassification_rate_mean
    np.testing.assert_allclose(actual.sort_index(), expected.sort_index())
    assert (aggregate.n_seeds == 2).all()
    assert not any(c.endswith(("_ci_low", "_ci_high")) for c in aggregate)
    before = _timestamps(diagnostics.root / "artifacts")
    stage_before = {
        stage: _timestamps(diagnostics.directory / stage)
        for stage in ("backbone", "observations", "graphs", "trees", "completion")
    }

    def unexpected(*args, **kwargs):
        raise AssertionError("completed diagnostics must be reused")

    monkeypatch.setattr(workflow, "observation_diagnostics", unexpected)
    monkeypatch.setattr(workflow, "oracle_graph", unexpected)
    monkeypatch.setattr(workflow, "backbone_diagnostics", unexpected)
    resumed = Diagnostics(deepcopy(diagnostics_config))
    assert resumed.directory == diagnostics.directory
    assert resumed.run("all")
    assert _timestamps(diagnostics.root / "artifacts") == before
    assert {
        stage: _timestamps(resumed.directory / stage) for stage in stage_before
    } == stage_before
    monkeypatch.setattr(resumed.exp, "prepare_development", unexpected)
    monkeypatch.setattr(resumed.exp, "dataset", unexpected)
    assert resumed.run("report")
    render_report(resumed.directory)


def test_report_changes_do_not_invalidate_computation(diagnostics_config, monkeypatch):
    diagnostics = Diagnostics(diagnostics_config)
    assert diagnostics.run("observations")
    full = workflow.implementation_signature()
    full["evaluation"]["reporting/diagnostics.py"] = "report-only-edit"
    full["evaluation"]["workflows/baseline.py"] = "unrelated-baseline-edit"
    monkeypatch.setattr(workflow, "implementation_signature", lambda: full)
    resumed = Diagnostics(deepcopy(diagnostics_config))
    assert resumed.directory == diagnostics.directory
    assert resumed.signature == diagnostics.signature
    # Altering a Leiden grid creates a study run, but keeps feature artifacts reusable.
    changed = deepcopy(diagnostics_config)
    changed["diagnostics"]["leiden"]["resolutions"] = [0.3]
    new_grid = Diagnostics(changed)
    assert new_grid.directory != diagnostics.directory
    before = _timestamps(diagnostics.root / "artifacts/observations")
    assert new_grid.run("observations")
    assert _timestamps(diagnostics.root / "artifacts/observations") == before


def test_tree_failure_is_visible_and_successful_settings_resume(
    diagnostics_config, monkeypatch
):
    diagnostics_config["diagnostics"]["treecluster"]["enabled"] = True
    calls = []
    fail = True

    def fake_treecluster(tree_path, cases, method, threshold, config, directory):
        calls.append(threshold)
        tree = Phylo.read(tree_path, "newick")
        assert {c.name for c in tree.get_terminals()} == set(cases.case_id)
        assert all(c.branch_length in (0, 1) for c in tree.find_clades())
        if threshold == 2 and fail:
            raise RuntimeError("visible external failure")
        return np.arange(len(cases)), {"test_adapter": True}

    monkeypatch.setattr(workflow, "treecluster", fake_treecluster)
    diagnostics = Diagnostics(diagnostics_config)
    assert not diagnostics.run("all")
    assert calls == [0, 2]  # One sampled-case set, independent of seed/process.
    assert not (diagnostics.exp.directory / "diagnostics.json").exists()
    assert not (diagnostics.directory / "completion/manifest.json").exists()
    assert read_json(diagnostics.directory / "manifest.json")["status"] == "partial"
    assert (
        "visible external failure" in (diagnostics.directory / "report.md").read_text()
    )
    records = read_json(diagnostics.directory / "trees/index.json")["records"]
    completed = Path(records[0]["artifact"])
    before = _timestamps(completed)
    fail = False
    assert diagnostics.run("trees")
    assert calls == [0, 2, 2]
    assert _timestamps(completed) == before
    marker = read_json(diagnostics.exp.directory / "diagnostics.json")
    assert valid_artifact(
        diagnostics.directory / "completion", _completion_signature(diagnostics, marker)
    )
    assert len(list((diagnostics.root / "artifacts/hop_trees").iterdir())) == 1
    assert (diagnostics.directory / "figures/transmission_hop_thresholds.png").exists()
    # A later broken checkpoint must revoke this run's existing completion marker.
    records = read_json(diagnostics.directory / "trees/index.json")["records"]
    failed = next(Path(r["artifact"]) for r in records if r["threshold_hops"] == 2)
    (failed / "metrics.json").write_text("interrupted write")
    fail = True
    assert not diagnostics.run("trees")
    assert not (diagnostics.exp.directory / "diagnostics.json").exists()
    assert not valid_artifact(
        diagnostics.directory / "completion", _completion_signature(diagnostics, marker)
    )
    assert _timestamps(completed) == before


def test_forest_control_is_explicitly_partial(diagnostics_config):
    diagnostics_config["diagnostics"]["treecluster"]["enabled"] = True
    diagnostics = Diagnostics(diagnostics_config)
    diagnostics.prepare()
    # The shared experiment may itself reject forests; exercise the control boundary.
    diagnostics.tree = nx.DiGraph(diagnostics.tree)
    diagnostics.tree.remove_edge(*next(iter(diagnostics.tree.edges)))
    assert not diagnostics.run("trees")
    index = read_json(diagnostics.directory / "trees/index.json")
    assert index["status"] == "partial"
    assert "single rooted transmission tree" in index["errors"][0]["error"]
    assert not (diagnostics.exp.directory / "diagnostics.json").exists()


def test_backbone_stage_uses_all_cases_without_observation_generation(diagnostics_config, monkeypatch, tmp_path):
    from evaluation.results.fig24 import create_figure
    from epilink_evaluation.reporting.backbone import load_backbone_evidence

    diagnostics_config["simulation"]["fraction_sampled"] = 0.4
    diagnostics_config["diagnostics"]["backbone"] = {
        "bootstrap_replicates": 12, "bootstrap_seed": 31,
    }
    diagnostics = Diagnostics(diagnostics_config)

    def unexpected(*args, **kwargs):
        raise AssertionError("Backbone characterisation must not generate observations")

    monkeypatch.setattr(diagnostics.exp, "prepare_development", unexpected)
    monkeypatch.setattr(diagnostics.exp, "dataset", unexpected)
    assert diagnostics.run("backbone")
    assert not list(diagnostics.exp.root.rglob("pairs.parquet"))
    assert not (diagnostics.exp.directory / "diagnostics.json").exists()
    index = read_json(diagnostics.directory / "backbone/index.json")
    assert index["datasets"] == {}
    assert len(index["records"]) == 1
    _, summary, _ = load_backbone_evidence(diagnostics.directory)
    assert summary["n_cases"] == 15
    assert summary["n_transmissions"] == 14
    assert summary["bootstrap"]["completed"] == 12
    assert summary["superspreading_operator"] == ">="
    artifact = Path(index["records"][0]["artifact"])
    nodes = pd.read_parquet(artifact / "nodes.parquet")
    truth_nodes = pd.read_parquet(diagnostics.truth_directory / "nodes.parquet")
    pd.testing.assert_frame_equal(nodes[["node_index", "case_id"]], truth_nodes, check_dtype=False)
    assert "offspring >=" in (diagnostics.directory / "report.md").read_text()
    assert (diagnostics.directory / "figures/backbone_characterisation.png").exists()
    monkeypatch.setattr(workflow, "backbone_diagnostics", unexpected)
    assert diagnostics.run("backbone")
    paths = create_figure(diagnostics.directory, tmp_path / "displays", fmt="both")
    assert paths["png"].is_file() and paths["pdf"].is_file()
    caption = (tmp_path / "displays/fig24_backbone_characterisation.md").read_text()
    assert "offspring >=" in caption
    assert "15 cases" in caption


def test_backbone_artifact_is_independent_of_seeds_and_sampling(diagnostics_config, monkeypatch):
    original = Diagnostics(diagnostics_config)
    assert original.run("backbone")
    record = read_json(original.directory / "backbone/index.json")["records"][0]
    before = _timestamps(Path(record["artifact"]))
    changed = deepcopy(diagnostics_config)
    changed["simulation"]["fraction_sampled"] = 0.5
    changed["splits"]["development"] = [62003]
    revised = Diagnostics(changed)

    def unexpected(*args, **kwargs):
        raise AssertionError("The same backbone must reuse its descriptive artefact")

    monkeypatch.setattr(workflow, "backbone_diagnostics", unexpected)
    assert revised.run("backbone")
    assert revised.directory != original.directory
    assert read_json(revised.directory / "backbone/index.json")["records"] == [record]
    assert _timestamps(Path(record["artifact"])) == before


def test_corrupt_backbone_revokes_completion_and_can_be_repaired(diagnostics_config, monkeypatch):
    from epilink_evaluation.reporting.backbone import load_backbone_evidence

    diagnostics = Diagnostics(diagnostics_config)
    assert diagnostics.run("all")
    record = read_json(diagnostics.directory / "backbone/index.json")["records"][0]
    artifact = Path(record["artifact"])
    (artifact / "summary.json").write_text("interrupted write")
    with pytest.raises(ValueError, match="changed backbone artefact"):
        load_backbone_evidence(diagnostics.directory)
    with pytest.raises(ValueError):
        diagnostics.exp.require_diagnostics()
    producer = workflow.backbone_diagnostics

    def failed(*args, **kwargs):
        raise RuntimeError("visible backbone failure")

    monkeypatch.setattr(workflow, "backbone_diagnostics", failed)
    assert not diagnostics.run("backbone")
    assert not (diagnostics.exp.directory / "diagnostics.json").exists()
    assert "visible backbone failure" in (diagnostics.directory / "report.md").read_text()
    monkeypatch.setattr(workflow, "backbone_diagnostics", producer)
    resumed = Diagnostics(deepcopy(diagnostics_config))
    assert resumed.run("backbone")
    resumed.exp.require_diagnostics()
    assert len(read_json(resumed.directory / "backbone/index.json")["records"]) == 1
