from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from epilink_evaluation.cli import main
from epilink_evaluation.provenance import digest_file, implementation_signature, read_json
from epilink_evaluation.workflows import baseline as baseline_module
from epilink_evaluation.workflows.baseline import Baseline
from epilink_evaluation.workflows.perturbation import ClusteringReplay, PerturbationStudy, paired_deltas, summarize
from epilink_evaluation.workflows.perturbation_config import MODES, load_study_config, scenarios
from epilink_evaluation.workflows.reference import EpiLinkClusteringReference
from epilink_evaluation.workflows.settings import settings_registry


@pytest.fixture
def evaluated_baseline(small_config, prepare_diagnostics):
    small_config["scorers"] = ["ESD", "ESS", "GD_D"]
    small_config["clustering"]["algorithms"] = ["leiden"]
    prepare_diagnostics(small_config)
    baseline = Baseline(small_config)
    assert baseline.run("all")
    return baseline


def study_config(tmp_path, reference, *, smoke=False):
    config = {
        "schema_version": 2, "name": "integration_perturbation",
        "baseline_run": str(reference.directory), "output_directory": "perturbation_outputs",
        "development_seeds": [80001, 80002], "seeds": [81001, 81002],
        "scorers": ["ESD", "ESS"], "criterion": "balanced_M0", "modes": list(MODES),
        "perturbations": [{"parameter": "incubation.mean", "multipliers": [0.75, 1.25]}],
        "smoke": {"cases": 10, "seed": 91001, "development_seed": 90001, "parameters": ["incubation.mean"]},
    }
    path = tmp_path / "perturbation.yaml"
    path.write_text(yaml.safe_dump(config))
    return load_study_config(path, smoke=smoke)


def test_default_scenarios_are_valid_one_parameter_changes(small_config):
    root = Path(__file__).resolve().parents[1]
    config = load_study_config(root / "evaluation/02_synthetic_perturbation/config.yaml")
    before = deepcopy(small_config["generation"])
    expanded = scenarios(config, before)
    assert len(expanded) == 13
    assert expanded[0]["generation"] == before
    for scenario in expanded[1:]:
        changed = []
        for name, value in before.items():
            if isinstance(value, dict):
                changed.extend(f"{name}.{key}" for key in value if value[key] != scenario["generation"][name][key])
            elif value != scenario["generation"][name]:
                changed.append(name)
        assert changed == [scenario["parameter"]]
    assert before == small_config["generation"]
    assert next(s for s in expanded if s["name"] == "relaxation_0")["value"] == 0


@pytest.mark.parametrize("parameter,levels", [
    ("incubation.cv", [2.0]), ("testing_delay.mean", [-1.0]),
    ("substitution_rate", [0.0]), ("incubation.cv", [0.0]),
])
def test_invalid_natural_history_levels_rejected(small_config, parameter, levels):
    with pytest.raises(ValueError):
        scenarios({"perturbations": [{"parameter": parameter, "values": levels}]}, small_config["generation"])


def test_paired_changes_allow_updated_settings_and_keep_missing_controls():
    frame = pd.DataFrame([
        {"scenario": scenario, "mode": "updated", "seed": seed, "pipeline": "method", "setting_id": setting, "M0_f1": score}
        for scenario, seed, setting, score in [
            ("baseline", 1, "control_setting", 0.5), ("baseline", 2, "control_setting", 0.8),
            ("perturbed", 1, "adapted_setting", 0.6), ("perturbed", 2, "adapted_setting", 0.7),
            ("perturbed", 3, "adapted_setting", 0.9),
        ]
    ])
    paired = paired_deltas(frame, ["mode", "seed", "pipeline"], ["M0_f1"])
    np.testing.assert_allclose(paired.delta_M0_f1, [0.1, -0.1, np.nan], atol=1e-15)
    assert paired.control_available.tolist() == [True, True, False]
    assert paired.baseline_setting_id.iloc[0] == "control_setting"
    row = summarize(paired, ["scenario", "mode", "pipeline"], ["delta_M0_f1"]).iloc[0]
    assert row.delta_M0_f1_mean == pytest.approx(0, abs=1e-15)
    assert row.delta_M0_f1_std == pytest.approx(np.sqrt(0.02))
    assert row.delta_M0_f1_count == row.n_controls == 2
    assert row.n_realizations == 3


def test_binary_epilink_leiden_is_removed_but_comparators_remain(small_config):
    small_config["scorers"] = ["EDD", "EDS", "ESD", "ESS", "GD_D", "LOGIT_D"]
    small_config["clustering"]["algorithms"] = ["leiden"]
    pipelines = {d["pipeline"] for d in settings_registry(small_config).values()}
    assert not any(f"leiden/{score}/binary" in pipelines for score in ("EDD", "EDS", "ESD", "ESS"))
    assert all(f"leiden/{score}/native" in pipelines for score in ("EDD", "EDS", "ESD", "ESS"))
    assert {"leiden/GD_D/binary", "leiden/LOGIT_D/binary"} <= pipelines


def test_four_arms_development_selection_pairing_and_cache(evaluated_baseline, tmp_path, monkeypatch):
    baseline = evaluated_baseline
    config = study_config(tmp_path, baseline, smoke=True)
    original = {str(p): (digest_file(p), p.stat().st_mtime_ns) for p in baseline.root.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Clustering perturbation must not fit models or run pairwise analysis")

    monkeypatch.setattr(baseline_module, "fit_logistic", forbidden)
    monkeypatch.setattr(ClusteringReplay, "pairwise", forbidden)
    study = PerturbationStudy(config)
    assert len(study.tree) == 10
    assert study.run()
    results = pd.read_csv(study.directory / "results.csv")
    assert len(results) == 3 * 4 * 2
    assert set(results.seed) == {91001}
    assert set(results.pipeline) == {"leiden/ESD/native", "leiden/ESS/native"}
    assert not (study.directory / "rankings.csv").exists()
    assert not (study.root / "artifacts/models").exists()
    assert "smoke validation" in (study.directory / "report.md").read_text()
    for scenario in study.scenarios:
        score_ids, dataset_ids = {}, set()
        for mode, (inference, clustering) in MODES.items():
            directory = study.directory / "scenarios" / scenario["name"] / mode
            selection = read_json(directory / "selection.json")
            assert selection["development_seeds"] == ([90001] if clustering == "updated" else [])
            for point in selection["operating_points"]:
                artifact = directory / "evaluation/seed_91001/clusters" / point["setting_id"]
                signature = read_json(artifact / "manifest.json")["signature"]
                scores = study.root / "artifacts/scores" / signature["score_id"]
                score_manifest = read_json(scores / "manifest.json")
                assert score_manifest["signature"]["training"] is None
                dataset_ids.add(score_manifest["signature"]["dataset"])
                score_ids[inference, clustering] = scores
                if clustering == "baseline":
                    assert study.reference.selected[point["setting_id"]] == point["definition"]
                else:
                    # Independent check: selected mean objective maximizes complete development evidence.
                    evidence = pd.read_csv(directory / "development/metrics.csv")
                    evidence = evidence.loc[evidence.pipeline == point["pipeline"]]
                    means = evidence.groupby("setting_id").M0_f1.mean()
                    assert means[point["setting_id"]] == pytest.approx(means.max())
                    assert set(evidence.seed) == {90001}
            assert not (directory / "evaluation/seed_91001/pairwise").exists()
        assert len(dataset_ids) == 1
        for inference in ("baseline", "matched"):
            assert score_ids[inference, "baseline"] == score_ids[inference, "updated"]
        matched = pd.read_parquet(score_ids["matched", "baseline"] / "scores.parquet")
        fixed = pd.read_parquet(score_ids["baseline", "baseline"] / "scores.parquet")
        if scenario["name"] == "baseline":
            pd.testing.assert_frame_equal(matched, fixed)
        else:
            assert not np.allclose(matched.ESS, fixed.ESS)
    deltas = pd.read_csv(study.directory / "results_deltas.csv")
    controls = results.loc[results.scenario == "baseline"].set_index(["mode", "seed", "pipeline"])
    for row in deltas.itertuples():
        control = controls.loc[(row.mode, row.seed, row.pipeline)]
        assert row.delta_M0_f1 == pytest.approx(row.M0_f1 - control.M0_f1)
        assert row.baseline_setting_id == control.setting_id
    cached = {p: p.stat().st_mtime_ns for p in study.root.glob("artifacts/*/*/*.parquet")}
    resumed = PerturbationStudy(config)
    assert resumed.directory == study.directory
    assert resumed.run()
    assert cached == {p: p.stat().st_mtime_ns for p in cached}
    pd.testing.assert_frame_equal(results, pd.read_csv(study.directory / "results.csv"))
    assert original == {str(p): (digest_file(p), p.stat().st_mtime_ns) for p in baseline.root.rglob("*") if p.is_file()}


def test_evaluation_is_inaccessible_before_updated_selection(evaluated_baseline, tmp_path):
    study = PerturbationStudy(study_config(tmp_path, evaluated_baseline))
    replay = ClusteringReplay(study, study.scenarios[1], "matched_inference_updated_clustering")
    with pytest.raises(ValueError, match="Freeze clustering"):
        replay.dataset(study.config["seeds"][0])


def test_missing_control_is_reported_as_partial(evaluated_baseline, tmp_path, monkeypatch):
    original = ClusteringReplay.clusters

    def fail_control(self, *args, **kwargs):
        if self.signature["scenario"]["name"] == "baseline":
            raise RuntimeError("Injected control failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ClusteringReplay, "clusters", fail_control)
    study = PerturbationStudy(study_config(tmp_path, evaluated_baseline, smoke=True))
    assert not study.run()
    coverage = pd.read_csv(study.directory / "coverage.csv")
    assert set(coverage.loc[coverage.scenario == "baseline", "status"]) == {"failed"}
    deltas = pd.read_csv(study.directory / "results_deltas.csv")
    assert not deltas.empty
    assert not deltas.control_available.any()
    assert deltas.delta_M0_f1.isna().all()


def test_failed_development_selection_never_releases_evaluation(evaluated_baseline, tmp_path, monkeypatch):
    study = PerturbationStudy(study_config(tmp_path, evaluated_baseline, smoke=True))
    replay = ClusteringReplay(study, study.scenarios[1], "matched_inference_updated_clustering")

    def failed_development(split, selected):
        assert split == "development"
        return False

    monkeypatch.setattr(replay, "clusters", failed_development)
    assert replay.run()["status"] == "failed"
    assert not replay._evaluation_released
    assert not (replay.directory / "evaluation").exists()


@pytest.mark.parametrize("change", ["development", "evaluation", "implementation"])
def test_changed_reference_is_rejected(evaluated_baseline, change):
    baseline = evaluated_baseline
    implementation = implementation_signature()
    if change == "development":
        (baseline.directory / "development/metrics.csv").write_text("changed evidence")
    elif change == "evaluation":
        seed = baseline.config["splits"]["evaluation"][0]
        point = next(p for p in read_json(baseline.directory / "selection/operating_points.json")["operating_points"] if p["pipeline"] == "leiden/ESD/native")
        (baseline.directory / "evaluation" / f"seed_{seed}" / "clusters" / point["setting_id"] / "metrics.json").write_text("{}")
    else:
        implementation["evaluation"]["metrics/partitions.py"] = "changed"
    with pytest.raises(ValueError):
        EpiLinkClusteringReference(baseline.directory, implementation, ["ESD", "ESS"], "balanced_M0")


@pytest.mark.parametrize("invalid", ["reused_seed", "reused_development_seed", "overlap", "same_output"])
def test_study_split_integrity(evaluated_baseline, tmp_path, invalid):
    config = study_config(tmp_path, evaluated_baseline)
    if invalid == "reused_seed":
        config["seeds"] = evaluated_baseline.config["splits"]["train"]
    elif invalid == "reused_development_seed":
        config["development_seeds"] = evaluated_baseline.config["splits"]["development"]
    elif invalid == "overlap":
        config["development_seeds"] = config["seeds"]
    else:
        config["output_directory"] = str(evaluated_baseline.root)
    with pytest.raises(ValueError):
        PerturbationStudy(config)


def test_cli_rejects_obsolete_perturbation_stages():
    for stage in ("select", "observations"):
        with pytest.raises(SystemExit) as exc:
            main(["perturbation", "--stage", stage])
        assert exc.value.code == 2
