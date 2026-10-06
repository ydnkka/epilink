"""Exercise frozen Boston assessment joins and exposure denominators."""

import importlib
import json

import pandas as pd
import pytest

manuscript = importlib.import_module("evaluation.results._boston.common")


def completed_study(tmp_path):
    cases = pd.DataFrame(
        {
            "case_id": [str(i) for i in range(6)],
            "Exposure": [
                "Conference",
                "Conference",
                "SNF",
                "SNF",
                "Unlabeled",
                "Unlabeled",
            ],
        }
    )
    cases.to_parquet(tmp_path / "cases.parquet", index=False)
    (tmp_path / "clusters").mkdir()
    (tmp_path / "trees").mkdir()
    (tmp_path / "assessment").mkdir()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "requested_stage": "all",
                "config": {"assessment": {"focus_exposures": ["Conference", "SNF"]}},
            }
        )
    )
    (tmp_path / "inputs.json").write_text(
        json.dumps(
            {
                "cases_path": str(tmp_path / "cases.parquet"),
                "n_cases": 6,
                "n_observed_pairs": 12,
                "n_all_pairs": 15,
            }
        )
    )
    (tmp_path / "reference.json").write_text(
        json.dumps({"selection_fingerprint": "same"})
    )
    for folder, count in (("clusters", 3), ("trees", 2)):
        (tmp_path / folder / "status.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "configured": count,
                    "completed": count,
                    "errors": [],
                }
            )
        )
    settings, points, summaries, named = {}, [], [], []
    for index, (_, pipeline, _) in enumerate(manuscript.FOCUS):
        setting = f"setting{index}"
        baseline = f"baseline{index}"
        kind = "treecluster" if pipeline.startswith("treecluster/") else "leiden"
        definition = {
            "kind": kind,
            "pipeline": pipeline,
            "baseline_setting_id": baseline,
            "score_name": "TREE"
            if kind == "treecluster"
            else ("ESD" if index == 0 else "LOGIT_D" if index == 1 else "GD_D"),
        }
        if kind == "treecluster":
            definition.update(
                method="max_clade",
                tree_kind=pipeline.rsplit("/", 1)[-1],
                baseline_data_process="deterministic",
                threshold=0.1,
            )
        else:
            definition["weight_policy"] = "native" if index < 2 else "binary"
        settings[setting] = definition
        points.append(
            {
                "criterion": "balanced_M0",
                "pipeline": pipeline,
                "status": "selected",
                "setting_id": setting,
                "baseline_setting_id": baseline,
                "definition": definition,
            }
        )
        summaries.append(
            {
                "setting_id": setting,
                "pipeline": pipeline,
                "baseline_setting_id": baseline,
                "n_clusters": 3,
                "n_singleton_cases": 1,
                "largest_cluster": 3,
                "n_observed_pairs": 12,
                "n_all_pairs": 15,
                "candidate_coverage": 12 / 15,
            }
        )
        for exposure, cluster_n, exposed_n in (("Conference", 3, 2), ("SNF", 2, 1)):
            named.append(
                {
                    "setting_id": setting,
                    "pipeline": pipeline,
                    "baseline_setting_id": baseline,
                    "exposure": exposure,
                    "n_cases": cluster_n,
                    "n_exposure": exposed_n,
                    "exposure_total": 2,
                    "exposure_fraction": exposed_n / cluster_n,
                    "exposure_recovery": exposed_n / 2,
                }
            )
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    (tmp_path / "selection.json").write_text(
        json.dumps(
            {
                "reference_selection_fingerprint": "same",
                "operating_points": points,
            }
        )
    )
    pd.DataFrame(summaries).to_csv(tmp_path / "assessment/summary.csv", index=False)
    pd.DataFrame(named).to_csv(
        tmp_path / "assessment/named_cluster_overlaps.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "setting_id": f"setting{graph}",
                "tree_setting_id": f"setting{tree}",
                "n_cases": 6,
                "adjusted_rand": 0.1 * (tree - 2),
                "adjusted_mutual_information": 0.05 * (graph + 1),
            }
            for graph in range(3)
            for tree in range(3, 5)
        ]
    ).to_csv(tmp_path / "assessment/tree_agreement.csv", index=False)
    return manuscript.BostonStudy.load(tmp_path)


def test_focused_boston_outputs_keep_both_exposure_denominators(tmp_path):
    study = completed_study(tmp_path)
    focus = study.focus_rows()
    assert len(focus) == 10
    assert focus.loc[0, "n_exposure"] == 2
    assert focus.loc[0, "exposure_fraction"] == pytest.approx(2 / 3)
    assert focus.loc[0, "exposure_recovery"] == 1
    assert focus.loc[1, "exposure_fraction"] == 0.5
    assert focus.loc[1, "exposure_recovery"] == 0.5
    ari, ami = importlib.import_module("evaluation.results.fig19").agreement_matrix(
        study
    )
    assert ari.shape == (3, 2)
    assert ami[2, 1] == pytest.approx(0.15)


def test_missing_exposure_cluster_stays_undefined_and_wrong_denominator_fails(tmp_path):
    study = completed_study(tmp_path)
    point = study.point(manuscript.FOCUS[0][1])
    study.named = study.named.loc[
        ~(
            (study.named.setting_id == point["setting_id"])
            & (study.named.exposure == "SNF")
        )
    ]
    assert study.representative(point, "SNF") is None
    assert pd.isna(study.focus_rows().loc[1, "exposure_recovery"])
    study.named.loc[
        study.named.setting_id == point["setting_id"], "exposure_fraction"
    ] = 0.1
    with pytest.raises(ValueError, match="denominators"):
        study.representative(point, "Conference")


def test_stale_explore_manifest_requires_complete_previous_all_run(tmp_path):
    completed_study(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest.update(status="running", requested_stage="explore")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "report.md").write_text("Status: complete (requested stage: all).")
    with pytest.warns(UserWarning, match="running 'explore'"):
        manuscript.BostonStudy.load(tmp_path)
    status_path = tmp_path / "trees/status.json"
    status = json.loads(status_path.read_text())
    status["completed"] = 1
    status_path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="Incomplete Boston partition coverage"):
        manuscript.BostonStudy.load(tmp_path)
