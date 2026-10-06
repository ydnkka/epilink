"""Check manuscript contrasts and scenario labels against paired saved evidence."""

import importlib
import json

import pandas as pd
import pytest

manuscript = importlib.import_module("evaluation.results._perturbation.common")


def study_fixture(tmp_path):
    scenarios = [
        {"name": "baseline", "parameter": None},
        {
            "name": "incubation_low",
            "parameter": "incubation.mean",
            "multiplier": 0.75,
            "value": 4.1,
        },
        {
            "name": "relaxation_zero",
            "parameter": "relaxation",
            "multiplier": None,
            "value": 0.0,
        },
    ]
    points = {
        ("balanced_M0", "leiden/ESD/native"): {
            "criterion": "balanced_M0",
            "pipeline": "leiden/ESD/native",
            "status": "selected",
            "setting_id": "frozen",
            "definition": {
                "data_process": "deterministic",
                "kind": "leiden",
                "score_name": "ESD",
            },
        }
    }
    return manuscript.Study(tmp_path, {"seeds": [11, 12]}, scenarios, points)


def test_paired_changes_keep_scenario_order_and_original_metric_units(tmp_path):
    study = study_fixture(tmp_path)
    assert study.scenario_labels == ["Incubation mean · 0.75×", "Clock relaxation · 0"]
    assert study.group_boundaries == [0.5]
    frame = pd.DataFrame(
        [
            {
                "scenario": "relaxation_zero",
                "mode": "matched",
                "score_name": "ESD",
                "n_realizations": 2,
                "n_controls": 2,
                "delta_M0_AP_mean": 0.25,
                "delta_M0_AP_count": 2,
            },
            {
                "scenario": "incubation_low",
                "mode": "matched",
                "score_name": "ESD",
                "n_realizations": 2,
                "n_controls": 2,
                "delta_M0_AP_mean": -0.1,
                "delta_M0_AP_count": 2,
            },
        ]
    )
    values, counts = study.matrix(
        frame, identifiers=("ESD",), key="score_name", metric="M0_AP"
    )
    assert values[:, 0].tolist() == [-10.0, 25.0]
    assert counts[:, 0].tolist() == [2, 2]
    frame.loc[0, "n_controls"] = 1
    with pytest.raises(ValueError, match="control coverage"):
        study.matrix(frame, identifiers=("ESD",), key="score_name", metric="M0_AP")


def test_scenario_ranges_use_seed_min_max_without_pooling_perturbations(tmp_path):
    study = study_fixture(tmp_path)
    frame = pd.DataFrame(
        [
            {
                "scenario": "relaxation_zero",
                "mode": "matched",
                "score_name": "ESD",
                "n_realizations": 2,
                "n_controls": 2,
                "delta_M0_AP_mean": float("nan"),
                "delta_M0_AP_min": float("nan"),
                "delta_M0_AP_max": float("nan"),
                "delta_M0_AP_count": 0,
            },
            {
                "scenario": "incubation_low",
                "mode": "matched",
                "score_name": "ESD",
                "n_realizations": 2,
                "n_controls": 2,
                "delta_M0_AP_mean": -0.10,
                "delta_M0_AP_min": -0.20,
                "delta_M0_AP_max": 0.0,
                "delta_M0_AP_count": 2,
            },
        ]
    )
    ranges = manuscript.paired_ranges(
        study, frame, metric="M0_AP", key="score_name", identifiers=("ESD",)
    )
    assert ranges.scenario.tolist() == study.scenario_names
    assert ranges.loc[0, ["mean_pp", "min_pp", "max_pp"]].tolist() == [
        -10,
        -20,
        0,
    ]
    assert pd.isna(ranges.loc[1, "mean_pp"])
    assert ranges.loc[1, "count"] == 0

    frame.loc[1, "delta_M0_AP_mean"] = 0.05
    with pytest.raises(ValueError, match="mean/range"):
        manuscript.paired_ranges(
            study, frame, metric="M0_AP", key="score_name", identifiers=("ESD",)
        )


def test_cluster_range_rejects_a_setting_other_than_frozen(tmp_path):
    study = study_fixture(tmp_path)
    frame = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "mode": "matched",
                "criterion": "balanced_M0",
                "pipeline": "leiden/ESD/native",
                "setting_id": "other",
                "n_realizations": 2,
                "n_controls": 2,
                "delta_M0_f1_mean": -0.1,
                "delta_M0_f1_min": -0.2,
                "delta_M0_f1_max": 0.0,
                "delta_M0_f1_count": 2,
            }
            for scenario in study.scenario_names
        ]
    )
    with pytest.raises(ValueError, match="Frozen setting mismatch"):
        manuscript.paired_ranges(
            study,
            frame,
            metric="M0_f1",
            key="pipeline",
            identifiers=("leiden/ESD/native",),
            criterion="balanced_M0",
        )


def test_mode_benefit_is_seed_paired_difference_of_control_deltas(tmp_path):
    study = study_fixture(tmp_path)
    rows = []
    for scenario in study.scenario_names:
        for seed in study.seeds:
            rows += [
                {
                    "scenario": scenario,
                    "seed": seed,
                    "score_name": "ESD",
                    "mode": "matched",
                    "delta_M0_AP": 0.3 if seed == 11 else 0.0,
                    "control_available": True,
                },
                {
                    "scenario": scenario,
                    "seed": seed,
                    "score_name": "ESD",
                    "mode": "baseline_fixed",
                    "delta_M0_AP": 0.1,
                    "control_available": True,
                },
            ]
    pd.DataFrame(rows[::-1]).to_csv(tmp_path / "rankings_deltas.csv", index=False)
    contrast = manuscript.paired_mode_contrast(
        study, ranking=True, metric="M0_AP", identifiers=("ESD",)
    ).set_index("scenario")
    assert contrast.loc["incubation_low", "mean"] == pytest.approx(5)
    assert contrast.loc["incubation_low", "min"] == pytest.approx(-10)
    assert contrast.loc["incubation_low", "max"] == pytest.approx(20)
    assert contrast.loc["incubation_low", "count"] == 2
    pd.DataFrame(rows[:-1]).to_csv(tmp_path / "rankings_deltas.csv", index=False)
    with pytest.raises(ValueError, match="lack paired seed"):
        manuscript.paired_mode_contrast(
            study, ranking=True, metric="M0_AP", identifiers=("ESD",)
        )


def test_run_requires_full_coverage_and_pinned_reference(tmp_path):
    study = study_fixture(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "config": {
                    "smoke_mode": False,
                    "seeds": study.seeds,
                    "modes": ["matched", "baseline_fixed"],
                },
            }
        )
    )
    (tmp_path / "scenarios.json").write_text(json.dumps(study.scenarios))
    (tmp_path / "reference.json").write_text(json.dumps({"run_fingerprint": "pinned"}))
    (tmp_path / "selection.json").write_text(
        json.dumps(
            {
                "run_fingerprint": "pinned",
                "operating_points": list(study.points.values()),
            }
        )
    )
    coverage = pd.DataFrame(
        [
            {
                "scenario": scenario["name"],
                "mode": mode,
                "status": "complete",
                "completed": 2,
                "expected": 2,
            }
            for scenario in study.scenarios
            for mode in ("matched", "baseline_fixed")
        ]
    )
    coverage.to_csv(tmp_path / "coverage.csv", index=False)
    assert manuscript.Study.load(tmp_path).scenario_names == study.scenario_names
    coverage.loc[0, "completed"] = 1
    coverage.to_csv(tmp_path / "coverage.csv", index=False)
    with pytest.raises(ValueError, match="Incomplete perturbation"):
        manuscript.Study.load(tmp_path)
