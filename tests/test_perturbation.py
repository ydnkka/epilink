from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from epilink_evaluation.cli import main
from epilink_evaluation.provenance import (
    digest_file,
    implementation_signature,
    read_json,
    write_json,
)
from epilink_evaluation.scorers import SCORERS
from epilink_evaluation.workflows import baseline as baseline_module
from epilink_evaluation.workflows import perturbation as perturbation_module
from epilink_evaluation.workflows.baseline import Baseline
from epilink_evaluation.workflows.perturbation import (
    FrozenReplay,
    PerturbationStudy,
    paired_deltas,
    summarize,
)
from epilink_evaluation.workflows.perturbation_config import (
    load_study_config,
    scenarios,
)
from epilink_evaluation.workflows.reference import BaselineReference


@pytest.fixture
def evaluated_baseline(small_config, prepare_diagnostics):
    prepare_diagnostics(small_config)
    baseline = Baseline(small_config)
    assert baseline.run("all")
    return baseline


def study_config(tmp_path, reference, *, smoke=False):
    config = {
        "schema_version": 1,
        "name": "integration_perturbation",
        "baseline_run": str(reference.directory),
        "output_directory": "perturbation_outputs",
        "seeds": [81001, 81002],
        "modes": ["matched", "baseline_fixed"],
        "perturbations": [
            {"parameter": "incubation.mean", "multipliers": [0.75, 1.25]}
        ],
        "smoke": {"cases": 10, "seed": 91001, "parameters": ["incubation.mean"]},
    }
    path = tmp_path / "perturbation.yaml"
    path.write_text(yaml.safe_dump(config))
    return load_study_config(path, smoke=smoke)


def test_default_scenarios_are_valid_one_parameter_changes(small_config):
    root = Path(__file__).resolve().parents[1]
    config = load_study_config(
        root / "evaluation/02_synthetic_perturbation/config.yaml"
    )
    before = deepcopy(small_config["generation"])
    expanded = scenarios(config, before)
    assert len(expanded) == 13
    assert expanded[0]["generation"] == before
    for scenario in expanded[1:]:
        changed = []
        for name, value in before.items():
            if isinstance(value, dict):
                changed.extend(
                    f"{name}.{key}"
                    for key in value
                    if value[key] != scenario["generation"][name][key]
                )
            elif value != scenario["generation"][name]:
                changed.append(name)
        assert changed == [scenario["parameter"]]
    assert before == small_config["generation"]
    assert next(s for s in expanded if s["name"] == "relaxation_0")["value"] == 0


@pytest.mark.parametrize(
    "parameter,levels",
    [
        ("incubation.cv", [2.0]),
        ("testing_delay.mean", [-1.0]),
        ("substitution_rate", [0.0]),
        ("incubation.cv", [0.0]),
    ],
)
def test_invalid_natural_history_levels_rejected(small_config, parameter, levels):
    with pytest.raises(ValueError):
        scenarios(
            {"perturbations": [{"parameter": parameter, "values": levels}]},
            small_config["generation"],
        )


def test_paired_changes_use_same_seed_and_keep_missing_controls():
    frame = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "mode": "matched",
                "seed": seed,
                "pipeline": "method",
                "M0_f1": score,
            }
            for scenario, seed, score in [
                ("baseline", 1, 0.5),
                ("baseline", 2, 0.8),
                ("perturbed", 1, 0.6),
                ("perturbed", 2, 0.7),
                ("perturbed", 3, 0.9),
            ]
        ]
    )
    paired = paired_deltas(frame, ["mode", "seed", "pipeline"], ["M0_f1"])
    np.testing.assert_allclose(paired.delta_M0_f1, [0.1, -0.1, np.nan], atol=1e-15)
    assert paired.control_available.tolist() == [True, True, False]
    summary = summarize(paired, ["scenario", "mode", "pipeline"], ["delta_M0_f1"])
    row = summary.iloc[0]
    assert row.delta_M0_f1_mean == pytest.approx(0, abs=1e-15)
    assert row.delta_M0_f1_std == pytest.approx(np.sqrt(0.02))
    assert row.delta_M0_f1_count == row.n_controls == 2
    assert row.n_realizations == 3


def test_frozen_replay_pairing_no_refit_and_cache_reuse(
    small_config, prepare_diagnostics, tmp_path, monkeypatch
):
    small_config["scorers"] = list(SCORERS)
    prepare_diagnostics(small_config)
    baseline = Baseline(small_config)
    assert baseline.run("all")
    config = study_config(tmp_path, baseline, smoke=True)
    original = {
        str(p): (digest_file(p), p.stat().st_mtime_ns)
        for p in baseline.root.rglob("*")
        if p.is_file()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("A perturbation study must never fit or select")

    monkeypatch.setattr(baseline_module, "fit_logistic", forbidden)
    monkeypatch.setattr(baseline_module, "select_operating_points", forbidden)
    # Replacing the live derived tree cannot replace the frozen reference topology.
    Path(small_config["inputs"]["tree_path"]).write_text("a different current input")
    study = PerturbationStudy(config)
    assert len(study.tree) == 10
    assert study.reference.identity["n_cases"] == 15
    assert study.run()
    assert read_json(study.directory / "selection.json") == read_json(
        baseline.directory / "selection/operating_points.json"
    )
    results = pd.read_csv(study.directory / "results.csv")
    assert len(results) == 3 * 2 * len(study.reference.frozen["operating_points"])
    assert set(results.seed) == {91001}
    assert set(results.setting_id) == set(study.reference.selected)
    assert (
        read_json(study.directory / "reference.json")["training_fingerprint"]
        == baseline.training_id
    )
    assert "smoke validation" in (study.directory / "report.md").read_text()
    assert read_json(study.directory / "manifest.json")["status"] == "complete"
    ambiguity = pd.read_csv(study.directory / "ambiguity.csv")
    assert len(ambiguity) == 3 * 12
    assert "mode" not in ambiguity
    for scenario in study.scenarios:
        saved_scores, dataset_ids = {}, []
        for mode in config["modes"]:
            replay = study.directory / "scenarios" / scenario["name"] / mode
            stage = read_json(replay / "evaluation/seed_91001/pairwise/manifest.json")
            score_dir = study.root / "artifacts/scores" / stage["signature"]["score_id"]
            score_manifest = read_json(score_dir / "manifest.json")
            assert score_manifest["signature"]["training"] == baseline.training_id
            dataset_ids.append(score_manifest["signature"]["dataset"])
            saved_scores[mode] = pd.read_parquet(score_dir / "scores.parquet")
        assert dataset_ids[0] == dataset_ids[1]
        columns = ["GD_D", "GD_S", "LOGIT_D", "LOGIT_S"]
        pd.testing.assert_frame_equal(
            saved_scores["matched"][columns], saved_scores["baseline_fixed"][columns]
        )
        if scenario["name"] == "baseline":
            pd.testing.assert_frame_equal(
                saved_scores["matched"], saved_scores["baseline_fixed"]
            )
        else:
            assert not np.allclose(
                saved_scores["matched"].ESS, saved_scores["baseline_fixed"].ESS
            )
    copied_models = (
        study.root / "artifacts/models" / baseline.training_id[:20] / "models.json"
    )
    assert digest_file(copied_models) == study.reference.identity["model_sha256"]
    observations = {
        p: p.stat().st_mtime_ns
        for p in (study.root / "artifacts/observations").glob("*/pairs.parquet")
    }
    resumed = PerturbationStudy(config)
    assert resumed.directory == study.directory
    assert resumed.run()
    assert observations == {p: p.stat().st_mtime_ns for p in observations}
    pd.testing.assert_frame_equal(results, pd.read_csv(study.directory / "results.csv"))
    assert original == {
        str(p): (digest_file(p), p.stat().st_mtime_ns)
        for p in baseline.root.rglob("*")
        if p.is_file()
    }


def test_perturbed_observation_ambiguity_pairing_and_cache(
    evaluated_baseline, tmp_path, monkeypatch
):
    config = study_config(tmp_path, evaluated_baseline)
    study = PerturbationStudy(config)
    calls = []
    original = perturbation_module.observation_diagnostics

    def diagnose(observations, truth):
        calls.append(len(observations))
        return original(observations, truth)

    def forbidden(*args, **kwargs):
        raise AssertionError("Observation diagnostics must not replay frozen methods")

    monkeypatch.setattr(perturbation_module, "observation_diagnostics", diagnose)
    monkeypatch.setattr(FrozenReplay, "run", forbidden)
    assert study.run("observations")
    assert len(calls) == len(study.scenarios) * len(config["seeds"])
    manifest = read_json(study.directory / "manifest.json")
    assert manifest["requested_stage"] == "observations"
    coverage = pd.read_csv(study.directory / "ambiguity_coverage.csv")
    assert coverage.status.eq("complete").all()
    assert coverage.completed.eq(12).all()
    frame = pd.read_csv(study.directory / "ambiguity.csv")
    keys = ["seed", "process", "feature_set", "endpoint"]
    assert "mode" not in frame
    assert not frame.duplicated(["scenario", *keys]).any()
    assert len(frame) == len(calls) * 12
    assert set(frame.process) == {"deterministic", "stochastic"}
    assert set(frame.feature_set) == {"GD", "GD_TD"}
    assert set(frame.endpoint) == {"M0", "Mle1", "Mle2"}

    # Verify cell-based ambiguity and minimum error from saved cell counts.
    for row in coverage.itertuples():
        cells = pd.read_parquet(Path(row.artifact) / "cells.parquet")
        summary = frame.loc[(frame.scenario == row.scenario) & (frame.seed == row.seed)]
        for result in summary.itertuples():
            group = cells.loc[
                (cells.process == result.process)
                & (cells.feature_set == result.feature_set)
                & (cells.endpoint == result.endpoint)
            ]
            mixed = group.n_target.gt(0) & group.n_other.gt(0)
            assert result.mixed_cell_fraction == pytest.approx(mixed.mean())
            assert result.pair_fraction_in_mixed_cells == pytest.approx(
                group.loc[mixed, "n_pairs"].sum() / group.n_pairs.sum()
            )
            assert result.minimum_feature_only_misclassification_rate == pytest.approx(
                np.minimum(group.n_target, group.n_other).sum() / group.n_pairs.sum()
            )

    deltas = pd.read_csv(study.directory / "ambiguity_deltas.csv")
    control = frame.loc[frame.scenario == "baseline"].set_index(keys)
    assert deltas.control_available.all()
    for row in deltas.itertuples():
        baseline = control.loc[tuple(getattr(row, key) for key in keys)]
        assert row.delta_mixed_cell_fraction == pytest.approx(
            row.mixed_cell_fraction - baseline.mixed_cell_fraction
        )
    summary = pd.read_csv(study.directory / "ambiguity_delta_summary.csv")
    assert summary.n_controls.eq(len(config["seeds"])).all()
    assert summary.delta_mixed_cell_fraction_count.eq(len(config["seeds"])).all()
    assert "Paired feature-ambiguity changes" in (study.directory / "report.md").read_text()

    cached = {
        Path(row.artifact) / "cells.parquet": (Path(row.artifact) / "cells.parquet").stat().st_mtime_ns
        for row in coverage.itertuples()
    }
    assert study.run("observations")
    assert len(calls) == len(coverage)
    assert cached == {path: path.stat().st_mtime_ns for path in cached}
    # A changed diagnostic file must invalidate and rebuild its artifact.
    damaged = Path(coverage.iloc[0].artifact) / "summary.csv"
    damaged.write_text("changed evidence\n")
    assert study.run("observations")
    assert len(calls) == len(coverage) + 1
    pd.testing.assert_frame_equal(frame, pd.read_csv(study.directory / "ambiguity.csv"))


def test_ambiguity_missing_control_is_visible(evaluated_baseline, tmp_path, monkeypatch):
    study = PerturbationStudy(study_config(tmp_path, evaluated_baseline, smoke=True))
    original = perturbation_module.prepare_observations

    def fail_control(config, *args):
        if config["generation"] == study.reference.config["generation"]:
            raise RuntimeError("Injected observation control failure")
        return original(config, *args)

    monkeypatch.setattr(perturbation_module, "prepare_observations", fail_control)
    assert not study.run("observations")
    coverage = pd.read_csv(study.directory / "ambiguity_coverage.csv")
    assert set(coverage.loc[coverage.scenario == "baseline", "status"]) == {"failed"}
    deltas = pd.read_csv(study.directory / "ambiguity_deltas.csv")
    assert not deltas.empty
    assert not deltas.control_available.any()
    assert deltas.delta_mixed_cell_fraction.isna().all()
    assert read_json(study.directory / "manifest.json")["status"] == "partial"


def test_missing_control_is_reported_as_partial(
    evaluated_baseline, tmp_path, monkeypatch
):
    config = study_config(tmp_path, evaluated_baseline, smoke=True)
    original = FrozenReplay.pairwise

    def fail_control(self, *args, **kwargs):
        if self.signature["scenario"]["name"] == "baseline":
            raise RuntimeError("Injected control failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FrozenReplay, "pairwise", fail_control)
    study = PerturbationStudy(config)
    assert not study.run()
    coverage = pd.read_csv(study.directory / "coverage.csv")
    assert set(coverage.loc[coverage.scenario == "baseline", "status"]) == {"failed"}
    deltas = pd.read_csv(study.directory / "results_deltas.csv")
    assert not deltas.empty
    assert not deltas.control_available.any()
    assert deltas.delta_M0_f1.isna().all()
    assert "partial" in (study.directory / "report.md").read_text()


@pytest.mark.parametrize(
    "change", ["models", "development", "evaluation", "implementation"]
)
def test_changed_or_incomplete_reference_is_rejected(evaluated_baseline, change):
    baseline = evaluated_baseline
    implementation = implementation_signature()
    if change == "models":
        path = (
            baseline.root
            / "artifacts/models"
            / baseline.training_id[:20]
            / "models.json"
        )
        path.write_text("{}")
    elif change == "development":
        path = baseline.directory / "development/metrics.csv"
        path.write_text("changed development evidence")
    elif change == "evaluation":
        seed = baseline.config["splits"]["evaluation"][0]
        path = (
            baseline.directory / "evaluation" / f"seed_{seed}" / "clusters/status.json"
        )
        saved = read_json(path)
        saved["status"] = "partial"
        write_json(path, saved)
    else:
        implementation["evaluation"]["metrics/pairwise.py"] = "changed"
    with pytest.raises(ValueError):
        BaselineReference(baseline.directory, implementation)


def test_new_workflow_code_can_consume_existing_baseline(evaluated_baseline):
    implementation = implementation_signature()
    implementation["evaluation"]["cli.py"] = "new CLI"
    implementation["evaluation"]["reporting/perturbation.py"] = "new report"
    reference = BaselineReference(evaluated_baseline.directory, implementation)
    assert reference.training_id == evaluated_baseline.training_id


@pytest.mark.parametrize("invalid", ["reused_seed", "same_output"])
def test_study_is_separate_from_baseline(evaluated_baseline, tmp_path, invalid):
    config = study_config(tmp_path, evaluated_baseline)
    if invalid == "reused_seed":
        config["seeds"] = evaluated_baseline.config["splits"]["train"]
    else:
        config["output_directory"] = str(evaluated_baseline.root)
    with pytest.raises(ValueError):
        PerturbationStudy(config)


def test_cli_rejects_perturbation_selection_stage():
    with pytest.raises(SystemExit) as exc:
        main(["perturbation", "--stage", "select"])
    assert exc.value.code == 2
