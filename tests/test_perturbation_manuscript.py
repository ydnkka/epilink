"""Check full-graph clustering displays against seed-paired, adapted resolutions."""

import importlib
import json

import pandas as pd
import pytest

from epilink_evaluation.workflows.perturbation import paired_deltas, summarize
from epilink_evaluation.workflows.perturbation_config import MODES
from epilink_evaluation.scorers import SCORERS

manuscript = importlib.import_module("evaluation.results._perturbation.common")


def study_fixture(tmp_path, scorers=("ESD",)):
    scenarios = [
        {"name": "baseline", "parameter": None},
        {"name": "incubation_low", "parameter": "incubation.mean", "multiplier": 0.75, "value": 4.1},
        {"name": "relaxation_zero", "parameter": "relaxation", "multiplier": None, "value": 0.0},
    ]
    config = {"schema_version": 2, "smoke_mode": False, "seeds": [11, 12], "development_seeds": [21, 22], "modes": list(MODES), "scorers": list(scorers), "criterion": "balanced_M0"}
    points, rows = {}, []
    for scenario in scenarios:
        for mode, (inference, clustering) in MODES.items():
            selected = []
            for index, score in enumerate(scorers):
                pipeline = f"leiden/{score}"
                setting = f"frozen_{score}" if clustering == "baseline" else f"{scenario['name']}_{mode}_{score}"
                point = {
                    "criterion": "balanced_M0", "pipeline": pipeline, "status": "selected", "setting_id": setting,
                    "definition": {"data_process": SCORERS[score].spec.data_process, "kind": "leiden", "score_name": score, "graph_mode": "full", "resolution": 0.1},
                }
                selected.append(point)
                if clustering == "baseline":
                    points["balanced_M0", pipeline] = point
                for seed in config["seeds"]:
                    delta = 0 if scenario["name"] == "baseline" else ((0.3 if seed == 11 else 0.0) if inference == "matched" else 0.1)
                    delta *= 1 + index * 0.1
                    rows.append({"scenario": scenario["name"], "seed": seed, "mode": mode, "pipeline": pipeline, "criterion": "balanced_M0", "setting_id": setting, "M0_f1": 0.5 + delta, "Mge3_contamination": 0.05})
            path = tmp_path / "scenarios" / scenario["name"] / mode
            path.mkdir(parents=True)
            (path / "selection.json").write_text(json.dumps({"operating_points": selected}))
    frame = pd.DataFrame(rows)
    deltas = paired_deltas(frame, ["mode", "seed", "pipeline", "criterion"], ["M0_f1", "Mge3_contamination"])
    deltas.to_csv(tmp_path / "results_deltas.csv", index=False)
    group = ["scenario", "mode", "pipeline", "criterion", "setting_id"]
    summarize(frame, group, ["M0_f1", "Mge3_contamination"]).to_csv(tmp_path / "results_summary.csv", index=False)
    summarize(deltas, group, ["delta_M0_f1", "delta_Mge3_contamination"]).to_csv(tmp_path / "results_delta_summary.csv", index=False)
    return manuscript.Study(tmp_path, config, scenarios, points)


def test_matrix_validates_scenario_specific_resolutions_and_metric_units(tmp_path):
    study = study_fixture(tmp_path)
    assert study.scenario_labels == ["Incubation mean · 0.75×", "Clock relaxation · 0"]
    assert study.group_boundaries == [0.5]
    frame = manuscript.cluster_summary(study, ("M0_f1",))
    values, counts = study.matrix(frame, identifiers=("leiden/ESD",), metric="M0_f1", mode="matched_inference_updated_clustering")
    assert values[:, 0].tolist() == pytest.approx([15, 15])
    assert counts[:, 0].tolist() == [2, 2]
    subset = frame["mode"].eq("matched_inference_updated_clustering")
    frame.loc[subset, "n_controls"] = 1
    with pytest.raises(ValueError, match="control coverage"):
        study.matrix(frame, identifiers=("leiden/ESD",), metric="M0_f1", mode="matched_inference_updated_clustering")


@pytest.mark.parametrize("clustering", ["baseline", "updated"])
def test_mode_benefit_is_seed_paired_without_matching_setting_ids(tmp_path, clustering):
    study = study_fixture(tmp_path)
    contrast = manuscript.paired_mode_contrast(study, metric="M0_f1", identifiers=("leiden/ESD",), clustering_mode=clustering).set_index("scenario")
    assert contrast.loc["incubation_low", "mean"] == pytest.approx(5)
    assert contrast.loc["incubation_low", "min"] == pytest.approx(-10)
    assert contrast.loc["incubation_low", "max"] == pytest.approx(20)
    assert contrast.loc["incubation_low", "count"] == 2
    frame = pd.read_csv(tmp_path / "results_deltas.csv")
    frame = frame.loc[~(frame["mode"].eq(f"matched_inference_{clustering}_clustering") & frame.seed.eq(11) & frame.scenario.eq("incubation_low"))]
    frame.to_csv(tmp_path / "results_deltas.csv", index=False)
    with pytest.raises(ValueError, match="lack paired seed"):
        manuscript.paired_mode_contrast(study, metric="M0_f1", identifiers=("leiden/ESD",), clustering_mode=clustering)


def test_run_requires_full_four_arm_coverage_and_pinned_reference(tmp_path):
    study = study_fixture(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "complete", "config": study.config}))
    (tmp_path / "scenarios.json").write_text(json.dumps(study.scenarios))
    (tmp_path / "reference.json").write_text(json.dumps({"run_fingerprint": "pinned"}))
    (tmp_path / "selection.json").write_text(json.dumps({"run_fingerprint": "pinned", "operating_points": list(study.points.values())}))
    coverage = pd.DataFrame([
        {"scenario": s["name"], "mode": mode, "status": "complete", "completed": 2, "expected": 2}
        for s in study.scenarios for mode in MODES
    ])
    coverage.to_csv(tmp_path / "coverage.csv", index=False)
    assert manuscript.Study.load(tmp_path).scenario_names == study.scenario_names
    coverage.loc[0, "completed"] = 1
    coverage.to_csv(tmp_path / "coverage.csv", index=False)
    with pytest.raises(ValueError, match="Incomplete perturbation"):
        manuscript.Study.load(tmp_path)


@pytest.mark.parametrize("scorers", [("ESD",), ("EDD", "EDS", "ESD", "ESS")])
def test_clustering_only_displays_render_all_four_modes(tmp_path, monkeypatch, scorers):
    from epilink_evaluation.utils import style

    study = study_fixture(tmp_path, scorers)
    save = style.save_figure

    def inspect_panels(fig, *args, **kwargs):
        titles = [axis.get_title().split("\n")[0] for axis in fig.axes if axis.get_title()]
        assert set(titles) == set(scorers)
        return save(fig, *args, dpi=300, **kwargs)

    monkeypatch.setattr(style, "save_figure", inspect_panels)
    for name in ("fig12", "fig14", "fig15"):
        module = importlib.import_module(f"evaluation.results.{name}")
        module.create_figure(study, tmp_path, fmt="png")
        assert list(tmp_path.glob(f"{name}_*.png"))
    table = importlib.import_module("evaluation.results.tab03")
    rows = table.build_rows(study)
    assert len(rows) == 4 * len(scorers)
    assert all("resolution" in row[2].lower() for row in rows)
    plotted = pd.read_csv(tmp_path / "fig12_cluster_sensitivity.csv")
    assert set(plotted.score_name) == set(scorers)
    assert set(plotted["mode"]) == set(MODES)
