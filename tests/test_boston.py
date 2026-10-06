"""Tests for Boston empirical clustering workflow."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from epilink_evaluation.provenance import fingerprint, read_json
from epilink_evaluation.workflows.boston import (
    BostonEmpirical,
    build_observations,
    load_boston_inputs,
)
from epilink_evaluation.workflows.boston_assessment import (
    assess_partitions,
    load_treecluster,
)
from epilink_evaluation.workflows.boston_config import load_study_config
from epilink_evaluation.workflows.boston_scoring import (
    BASELINE_SCORES,
    operating_settings,
    score_observations,
)


def test_load_boston_inputs_valid(tmp_path):
    cases = pd.DataFrame(
        {
            "case_id": ["A", "B", "C"],
            "sample_date": pd.date_range("2020-03-01", periods=3),
            "Exposure": ["Conference", "SNF", "BHCHP"],
        }
    )
    cases_path = tmp_path / "cases.parquet"
    cases.to_parquet(cases_path, index=False)

    pairs = pd.DataFrame(
        {
            "CaseID1": ["A", "B"],
            "CaseID2": ["B", "C"],
            "TD": [1.0, 2.0],
            "GD": [2.0, 5.0],
            "TN93_distance": [2.0 / 29903, 5.0 / 29903],
        }
    )
    pairs_path = tmp_path / "pairs.parquet"
    pairs.to_parquet(pairs_path, index=False)

    cases_out, pairs_out = load_boston_inputs(cases_path, pairs_path)
    assert len(cases_out) == 3
    assert len(pairs_out) == 2


def test_load_boston_inputs_missing_metadata(tmp_path):
    cases = pd.DataFrame(
        {
            "case_id": ["A", "B"],
            "sample_date": pd.date_range("2020-03-01", periods=2),
        }
    )
    cases_path = tmp_path / "cases.parquet"
    cases.to_parquet(cases_path, index=False)

    pairs = pd.DataFrame(
        {
            "CaseID1": ["A", "C"],
            "CaseID2": ["B", "D"],
            "TD": [1.0, 2.0],
            "GD": [2.0, 5.0],
            "TN93_distance": [2.0 / 29903, 5.0 / 29903],
        }
    )
    pairs_path = tmp_path / "pairs.parquet"
    pairs.to_parquet(pairs_path, index=False)

    with pytest.raises(ValueError, match="lack metadata"):
        load_boston_inputs(cases_path, pairs_path)


def test_build_observations(tmp_path):
    cases = pd.DataFrame(
        {
            "case_id": ["A", "B", "C"],
            "sample_date": pd.date_range("2020-03-01", periods=3),
        }
    )
    cases_path = tmp_path / "cases.parquet"
    cases.to_parquet(cases_path, index=False)

    pairs = pd.DataFrame(
        {
            "CaseID1": ["A", "B"],
            "CaseID2": ["B", "C"],
            "TD": [1.0, 2.0],
            "GD": [2.0, 5.0],
            "TN93_distance": [2.0 / 29903, 5.0 / 29903],
        }
    )
    pairs_path = tmp_path / "pairs.parquet"
    pairs.to_parquet(pairs_path, index=False)

    cases_df, pairs_df = load_boston_inputs(cases_path, pairs_path)
    observations, case_index = build_observations(cases_df, pairs_df)
    assert len(observations) == 2
    assert list(observations.columns) == ["a", "b", "TD", "GD", "tn93"]
    assert case_index["A"] == 0
    assert case_index["B"] == 1
    assert case_index["C"] == 2


@pytest.fixture
def evaluated_baseline(small_config, prepare_diagnostics):
    from epilink_evaluation.workflows.baseline import Baseline

    small_config["scorers"] = [
        "EDD",
        "EDS",
        "ESD",
        "ESS",
        "GD_S",
        "GD_D",
        "LOGIT_S",
        "LOGIT_D",
    ]
    small_config["clustering"]["algorithms"] = ["components", "leiden"]
    small_config["treecluster"]["enabled"] = False
    prepare_diagnostics(small_config)
    baseline = Baseline(small_config)
    assert baseline.run("all")
    return baseline


def boston_config(tmp_path, reference):
    cases_path = tmp_path / "boston_cases.parquet"
    pairs_path = tmp_path / "boston_pairs.parquet"

    cases = pd.DataFrame(
        {
            "case_id": ["A", "B", "C", "D", "E"],
            "sample_date": pd.date_range("2020-03-01", periods=5),
            "Exposure": ["Conference", "SNF", "BHCHP", "City", "Unlabeled"],
        }
    )
    cases.to_parquet(cases_path, index=False)

    pairs = pd.DataFrame(
        {
            "CaseID1": ["A", "A", "B", "B", "C"],
            "CaseID2": ["B", "C", "C", "D", "D"],
            "TD": [1.0, 2.0, 1.0, 2.0, 1.0],
            "GD": [2.0, 5.0, 3.0, 8.0, 1.0],
            "TN93_distance": [
                2.0 / 29903,
                5.0 / 29903,
                3.0 / 29903,
                8.0 / 29903,
                1.0 / 29903,
            ],
        }
    )
    pairs.to_parquet(pairs_path, index=False)

    config = {
        "schema_version": 1,
        "name": "test_boston",
        "baseline_run": str(reference.directory),
        "output_directory": str(tmp_path / "boston_outputs"),
        "inputs": {
            "cases_path": str(cases_path),
            "pairs_path": str(pairs_path),
        },
        "scorers": ["ES", "ED", "GD_S", "GD_D"],
    }
    path = tmp_path / "boston.yaml"
    path.write_text(yaml.safe_dump(config))
    return load_study_config(argv=[], config_path=path)


def test_boston_runs_without_training_artifacts_and_reuses_scores(
    evaluated_baseline, tmp_path, monkeypatch
):
    from epilink_evaluation.scorers.registry import LogisticScorer
    from epilink_evaluation.workflows import baseline as baseline_module
    from epilink_evaluation.workflows import reference as reference_module

    config = boston_config(tmp_path, evaluated_baseline)
    # Boston must run even when the synthetic fitted classifiers are absent.
    shutil.rmtree(evaluated_baseline.root / "artifacts/models")

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Boston must not fit, select, load training artifacts, or predict logistic scores"
        )

    monkeypatch.setattr(baseline_module, "fit_logistic", forbidden)
    monkeypatch.setattr(baseline_module, "select_operating_points", forbidden)
    monkeypatch.setattr(LogisticScorer, "predict", forbidden)
    monkeypatch.setattr(reference_module, "checked_artifact", forbidden)
    monkeypatch.setattr(reference_module, "command_identity", forbidden)
    study = BostonEmpirical(config)
    assert study.n_cases == 5
    assert study.n_observed_pairs == 5
    assert study.context.logistic_models == {}
    assert study.scoring_config["inference"] == evaluated_baseline.config["inference"]
    assert study.scoring_config["scorer"] == evaluated_baseline.config["scorer"]
    assert study.run()
    assert read_json(study.directory / "manifest.json")["status"] == "complete"
    assert (study.directory / "clusters" / "status.json").exists()
    cluster_status = read_json(study.directory / "clusters" / "status.json")
    assert cluster_status["status"] == "complete"
    assert (study.directory / "report.md").exists()
    assert (study.directory / "report.html").exists()
    definitions = read_json(study.directory / "settings.json")
    assert {d["score_name"] for d in definitions.values()} == set(config["scorers"])
    original = read_json(evaluated_baseline.directory / "settings.json")
    for key, definition in definitions.items():
        assert key == fingerprint(definition)[:20]
        source = original[definition["baseline_setting_id"]]
        assert (
            definition["baseline_score_name"]
            == BASELINE_SCORES[definition["score_name"]]
        )
        for field in (
            "threshold",
            "empty",
            "weight_policy",
            "resolution",
            "restarts",
            "algorithm_seed",
        ):
            assert definition.get(field) == source.get(field)
        if definition["kind"] != "pairwise":
            membership = pd.read_parquet(
                study.directory / "clusters" / key / "memberships.parquet"
            )
            assert set(membership.case_id) == {"A", "B", "C", "D", "E"}
            # E has no observed pairs and is retained as an isolated vertex.
            cluster = membership.loc[membership.case_id == "E", "cluster_id"].iloc[0]
            assert membership.cluster_id.eq(cluster).sum() == 1
    scores, score_id = study.score()
    assert list(scores.columns) == ["CaseID1", "CaseID2", "ES", "ED", "GD_S", "GD_D"]
    np.testing.assert_array_equal(scores.GD_S, scores.GD_D)
    assert len(scores) == 5  # Missing pairs have not been imputed or scored.
    first = pd.read_csv(study.directory / "clusters/metrics.csv")
    assert pd.api.types.is_numeric_dtype(first.size_mean)
    resumed = BostonEmpirical(config)
    monkeypatch.setattr(resumed.context, "epilink", forbidden)
    assert resumed.run()
    repeated, repeated_id = resumed.score()
    assert score_id == repeated_id
    pd.testing.assert_frame_equal(scores, repeated)
    pd.testing.assert_frame_equal(
        first, pd.read_csv(resumed.directory / "clusters/metrics.csv")
    )
    assert not (study.root / "artifacts/models").exists()


def test_default_boston_config_covers_expanded_models():
    root = Path(__file__).resolve().parents[1]
    config = load_study_config(
        config_path=root / "evaluation/03_boston_application/config.yaml"
    )
    assert config["schema_version"] == 1
    assert "exploration" not in config
    assert config["scorers"] == [
        "EDD",
        "EDS",
        "ESD",
        "ESS",
        "GD_S",
        "GD_D",
        "LOGIT_S",
        "LOGIT_D",
    ]
    assert "seeds" not in config
    assert config["assessment"]["treecluster_path"] is None
    assert config["trees"]["enabled"] is True
    assert Path(config["trees"]["alignment_path"]).exists()
    assert (
        Path(config["inputs"]["cases_path"])
        == root / "evaluation/03_boston_application/outputs/inputs/cases.parquet"
    )


def test_boston_cli_rejects_removed_explore_stage():
    from epilink_evaluation.cli import main

    with pytest.raises(SystemExit) as error:
        main(["boston", "--stage", "explore"])
    assert error.value.code == 2


def test_boston_workflow_prepares_shared_inputs(
    evaluated_baseline, tmp_path, monkeypatch
):
    from epilink_evaluation.inputs import boston as adapter

    config = boston_config(tmp_path, evaluated_baseline)
    cases = pd.read_parquet(config["inputs"]["cases_path"])
    pairs = pd.read_parquet(config["inputs"]["pairs_path"])
    prepared = tmp_path / "outputs/inputs"
    config["inputs"].update(
        data_root=str(tmp_path / "data"),
        cases_path=str(prepared / "cases.parquet"),
        pairs_path=str(prepared / "observed_pairs.parquet"),
    )
    calls = []

    def prepare(data_root, output):
        calls.append((data_root, output))
        output.mkdir(parents=True)
        cases.to_parquet(output / "cases.parquet", index=False)
        pairs.to_parquet(output / "observed_pairs.parquet", index=False)

    monkeypatch.setattr(adapter, "prepare_boston", prepare)
    study = BostonEmpirical(config)
    assert calls == [(str(tmp_path / "data"), prepared)]
    assert study.n_cases == len(cases)
    assert read_json(study.directory / "inputs.json")["cases_path"] == str(
        prepared / "cases.parquet"
    )


def test_boston_report_includes_tree_only_results(tmp_path):
    from epilink_evaluation.provenance import write_json
    from epilink_evaluation.reporting.boston import render_report

    write_json(
        tmp_path / "manifest.json", {"status": "complete", "requested_stage": "trees"}
    )
    write_json(tmp_path / "reference.json", {"run_directory": "baseline/run"})
    write_json(
        tmp_path / "inputs.json",
        {
            "cases_path": "cases.parquet",
            "pairs_path": "pairs.parquet",
            "n_cases": 3,
            "n_observed_pairs": 2,
            "n_all_pairs": 3,
            "treecluster_path": None,
            "treecluster_sha256": None,
            "alignment_path": "alignment.fasta",
            "alignment_sha256": "abc",
            "trees_enabled": True,
        },
    )
    write_json(tmp_path / "selection.json", {"operating_points": []})
    trees = tmp_path / "trees"
    trees.mkdir()
    write_json(
        trees / "status.json", {"status": "complete", "configured": 1, "completed": 1}
    )
    pd.DataFrame(
        {
            "setting_id": ["tree"],
            "pipeline": ["treecluster/empirical/stochastic/dated"],
            "n_clusters": [2],
        }
    ).to_csv(trees / "metrics.csv", index=False)

    render_report(tmp_path)
    report = (tmp_path / "report.md").read_text()
    assert "requested stage: trees" in report
    assert "Raw and dated TreeCluster status" in report
    assert "Frozen TreeCluster settings on Boston trees" in report


@pytest.mark.parametrize("names", [["LOGIT"], ["FOO"], [], ["ES", "ES"], ["ES", "ESS"]])
def test_boston_rejects_unsupported_scorers(tmp_path, names):
    path = tmp_path / "boston.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "baseline_run": "baseline/current.json",
                "output_directory": "outputs",
                "inputs": {},
                "scorers": names,
            }
        )
    )
    with pytest.raises(ValueError, match="Boston scorer"):
        load_study_config(config_path=path)


def test_training_free_scores_use_one_observed_distance_vector():
    observations = pd.DataFrame({"GD": [0.25, 4.5], "TD": [1.0, 3.0]})
    calls = []

    class Context:
        def epilink(self, process):
            def score_target(*, sample_time_difference, genetic_distance):
                calls.append(process)
                np.testing.assert_array_equal(genetic_distance, [0.25, 4.5])
                np.testing.assert_array_equal(sample_time_difference, [1.0, 3.0])
                return [0.8, 0.2] if process == "stochastic" else [0.9, 0.1]

            return SimpleNamespace(score_target=score_target)

    names = ["ES", "ED", "GD_S", "GD_D"]
    scores = score_observations(observations, Context(), names)
    assert calls == ["stochastic", "deterministic"]
    np.testing.assert_array_equal(scores.ES, [0.8, 0.2])
    np.testing.assert_array_equal(scores.ED, [0.9, 0.1])
    np.testing.assert_array_equal(scores.GD_S, observations.GD)
    np.testing.assert_array_equal(scores.GD_D, observations.GD)
    empty = score_observations(observations.iloc[:0], Context(), names)
    assert empty.empty and list(empty.columns) == names
    assert len(calls) == 2


def test_genetic_rules_retain_distinct_synthetic_thresholds():
    definitions, points = {}, []
    for name, threshold in (("GD_S", 2.0), ("GD_D", 5.0), ("LOGIT_S", 0.5)):
        definition = {
            "kind": "components",
            "pipeline": f"components/{name}",
            "score_name": name,
            "data_process": "deterministic" if name == "GD_D" else "stochastic",
            "threshold": threshold,
            "empty": False,
            "weight_policy": "binary",
        }
        key = fingerprint(definition)[:20]
        definitions[key] = definition
        points.append(
            {
                "pipeline": definition["pipeline"],
                "status": "selected",
                "criterion": "frozen",
                "setting_id": key,
                "definition": definition,
            }
        )
    reference = SimpleNamespace(
        config={"scorers": ["GD_S", "GD_D", "LOGIT_S"]},
        selected=definitions,
        frozen={"criteria": [], "operating_points": points},
        identity={"selection_fingerprint": "source"},
    )
    adapted, selection = operating_settings(reference, ["GD_S", "GD_D"])
    assert {d["score_name"]: d["threshold"] for d in adapted.values()} == {
        "GD_S": 2.0,
        "GD_D": 5.0,
    }
    assert len(selection["operating_points"]) == 2
    assert all(d["data_process"] == "empirical" for d in adapted.values())
    with pytest.raises(ValueError, match="source scorers"):
        operating_settings(reference, ["ES"])


def test_current_boston_reference_does_not_depend_on_adapter_hash(
    evaluated_baseline, tmp_path
):
    from copy import deepcopy

    from epilink_evaluation.workflows.reference import OperatingReference

    config = boston_config(tmp_path, evaluated_baseline)
    changed = deepcopy(config["implementation"])
    changed["evaluation"]["inputs/boston.py"] = "new empirical adapter"
    assert (
        OperatingReference(evaluated_baseline.directory, changed).directory
        == evaluated_baseline.directory
    )
    changed["evaluation"]["scorers/logistic.py"] = "changed baseline scorer"
    with pytest.raises(ValueError, match="Scientific implementation"):
        OperatingReference(evaluated_baseline.directory, changed)


def test_expanded_boston_scores_and_external_comparator(evaluated_baseline, tmp_path):
    from epilink_evaluation.scorers.logistic import predict_logistic

    config = boston_config(tmp_path, evaluated_baseline)
    config["scorers"] = [
        "EDD",
        "EDS",
        "ESD",
        "ESS",
        "GD_S",
        "GD_D",
        "LOGIT_S",
        "LOGIT_D",
    ]
    tree = tmp_path / "treecluster.tsv"
    tree.write_text("SequenceName\tClusterNumber\nA\t1\nB\t1\nC\t-1\nD\t2\nE\t-1\n")
    config["assessment"] = {
        "treecluster_path": str(tree),
        "focus_exposures": ["Conference", "SNF"],
        "min_cluster_size": 2,
    }
    study = BostonEmpirical(config)
    assert study.run()
    scores, _ = study.score()
    assert list(scores.columns[2:]) == config["scorers"]
    np.testing.assert_allclose(scores.EDD, scores.EDS)
    np.testing.assert_allclose(scores.ESD, scores.ESS)
    np.testing.assert_allclose(scores.GD_S, scores.GD_D)
    for scorer, process in (("LOGIT_S", "stochastic"), ("LOGIT_D", "deterministic")):
        expected = predict_logistic(
            study.observations.assign(**{f"GD_{process}": study.observations.GD}),
            process,
            study.context.logistic_models[process],
        )
        np.testing.assert_allclose(scores[scorer], expected)
    assert (scores.LOGIT_S != scores.LOGIT_D).any()
    report = (study.directory / "report.md").read_text()
    assert "Named exposure groups" in report
    assert "transmission truth" in report
    assessment = study.directory / "assessment"
    summary = pd.read_csv(assessment / "summary.csv")
    assert (
        len(summary) == read_json(study.directory / "clusters/status.json")["completed"]
    )
    assert (summary.n_observed_pairs == 5).all()
    assert (summary.candidate_coverage == 0.5).all()
    assert (assessment / "cluster_composition.csv").exists()
    assert (assessment / "named_cluster_overlaps.csv").exists()
    assert (assessment / "best_cluster_overlaps.csv").exists()
    assert not (study.root / "artifacts/models").exists()


def test_external_treecluster_assessment_counts_singletons_and_overlap(tmp_path):
    from epilink_evaluation.provenance import complete_artifact

    cases = pd.DataFrame(
        {
            "case_id": ["A", "B", "C", "D", "E"],
            "Exposure": ["SNF", "SNF", "SNF", "Conference", "Conference"],
            "Clade": ["A", "A", "B", "B", "B"],
        }
    )
    tree = tmp_path / "treecluster.tsv"
    tree.write_text("SequenceName\tClusterNumber\nA\t1\nB\t1\nC\t-1\nD\t2\nE\t-1\n")
    comparator = load_treecluster(tree, cases)
    assert comparator.treecluster_group.nunique() == 4  # the two -1 rows do not merge
    definitions = {
        "setting": {
            "kind": "components",
            "pipeline": "components/ESS",
            "score_name": "ESS",
            "baseline_setting_id": "source",
        }
    }
    artifact = tmp_path / "clusters/setting"
    artifact.mkdir(parents=True)
    pd.DataFrame({"case_id": cases.case_id, "cluster_id": [0, 0, 0, 1, 2]}).to_parquet(
        artifact / "memberships.parquet",
        index=False,
    )
    complete_artifact(artifact, {"test": "assessment"}, ["memberships.parquet"])
    assess_partitions(
        tmp_path, cases, definitions, ["SNF", "Conference"], 2, comparator, 3, 10
    )
    summary = pd.read_csv(tmp_path / "assessment/summary.csv")
    assert summary.loc[0, "n_singleton_cases"] == 2
    named = pd.read_csv(tmp_path / "assessment/named_cluster_overlaps.csv")
    snf = named.loc[named.exposure == "SNF"].iloc[0]
    assert snf.n_cases == 3 and snf.n_exposure == 3
    assert snf.treecluster_group == "cluster:1"
    assert snf.shared == 2 and snf.jaccard == pytest.approx(2 / 3)
    assert (named.exposure == "Conference").sum() == 0  # no non-singleton focus cluster
    best = pd.read_csv(tmp_path / "assessment/best_cluster_overlaps.csv")
    assert best.loc[best.treecluster_group == "cluster:1", "shared"].iloc[0] == 2
    with pytest.raises(ValueError, match="same unique sample IDs"):
        load_treecluster(tree, cases.iloc[:-1])
