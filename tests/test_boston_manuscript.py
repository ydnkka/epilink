"""Exercise frozen Boston assessment joins and exposure denominators."""

import importlib
import json

import numpy as np
import pandas as pd
import pytest

manuscript = importlib.import_module("evaluation.results._boston.common")


def completed_study(tmp_path, *, other_exposures=False):
    exposures = [
        "Conference", "Conference", "SNF", "SNF", "Unlabeled", "Unlabeled"
    ]
    clusters = {
        0: {"Conference": 2, "Unlabeled": 1},
        1: {"SNF": 1, "Unlabeled": 1},
        2: {"SNF": 1},
    }
    if other_exposures:
        exposures = [
            "Conference", "Conference", "SNF", "SNF", "BHCHP", "City",
            "Unlabeled", "Unlabeled", "BHCHP", "City",
        ]
        clusters = {
            0: {"Conference": 2, "Unlabeled": 1},
            1: {"SNF": 1, "BHCHP": 1, "City": 1},
            2: {"SNF": 1},
            3: {"Unlabeled": 1},
            4: {"BHCHP": 1},
            5: {"City": 1},
        }
    n_cases = len(exposures)
    n_all_pairs = n_cases * (n_cases - 1) // 2
    cases = pd.DataFrame(
        {
            "case_id": [str(i) for i in range(n_cases)],
            "Exposure": exposures,
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
                "config": {
                    "assessment": {
                        "focus_exposures": ["Conference", "SNF"],
                        "min_cluster_size": 2,
                    }
                },
            }
        )
    )
    (tmp_path / "inputs.json").write_text(
        json.dumps(
            {
                "cases_path": str(tmp_path / "cases.parquet"),
                "n_cases": n_cases,
                "n_observed_pairs": 12,
                "n_all_pairs": n_all_pairs,
                "alignment_length": 30000,
            }
        )
    )
    (tmp_path / "reference.json").write_text(
        json.dumps({"selection_fingerprint": "same"})
    )
    settings, points, summaries, named, composition = {}, [], [], [], []
    graph_scorers = {
        "components": (
            "EDD", "EDS", "ESD", "ESS", "GD_D", "GD_S", "LOGIT_D", "LOGIT_S"
        ),
        "leiden": ("EDD", "EDS", "ESD", "ESS", "LOGIT_D", "LOGIT_S"),
    }
    templates = []
    for kind, scores in graph_scorers.items():
        for score in scores:
            definition = {
                "kind": kind,
                "pipeline": f"{kind}/{score}",
                "score_name": score,
                "data_process": "empirical",
                "baseline_data_process": "deterministic"
                if score.endswith("D")
                else "stochastic",
                "threshold": 0 if score.startswith("GD_") else 0.5,
            }
            if kind == "leiden":
                definition.update(graph_mode="full", threshold=None, objective="CPM")
            templates.append(definition)
    for tree in ("raw", "dated"):
        for source in ("deterministic", "stochastic"):
            templates.append(
                {
                    "kind": "treecluster",
                    "pipeline": f"treecluster/empirical/{source}/{tree}",
                    "score_name": "TREE",
                    "data_process": "empirical",
                    "baseline_data_process": source,
                    "tree_kind": tree,
                    "method": "max_clade" if source == "deterministic" else "avg_clade",
                }
            )
    for criterion_index, criterion in enumerate(
        ("balanced_M0", "balanced_Mle1", "balanced_Mle2")
    ):
        for index, template in enumerate(templates):
            setting = f"setting{criterion_index}_{index}"
            baseline = f"baseline{criterion_index}_{index}"
            definition = {**template, "baseline_setting_id": baseline}
            if definition["kind"] == "leiden":
                definition["resolution"] = 0.1 * (criterion_index + 1)
            elif definition["kind"] == "treecluster":
                cutoff = 5 * (criterion_index + 1)
                if definition["tree_kind"] == "raw":
                    definition.update(
                        threshold=cutoff / 30000,
                        threshold_snps=cutoff,
                        threshold_input=cutoff,
                        baseline_threshold=cutoff / 5000,
                        threshold_units="snps",
                    )
                else:
                    definition.update(threshold=cutoff * 10, threshold_units="days")
            settings[setting] = definition
            base = {
                "setting_id": setting,
                "pipeline": definition["pipeline"],
                "baseline_setting_id": baseline,
            }
            points.append(
                {
                    **base,
                    "criterion": criterion,
                    "status": "selected",
                    "definition": definition,
                }
            )
            summaries.append(
                {
                    **base,
                    "n_clusters": len(clusters),
                    "n_singleton_cases": sum(sum(counts.values()) == 1 for counts in clusters.values()),
                    "largest_cluster": max(sum(counts.values()) for counts in clusters.values()),
                    "n_observed_pairs": 12,
                    "n_all_pairs": n_all_pairs,
                    "candidate_coverage": 12 / n_all_pairs,
                }
            )
            for cluster_id, counts in clusters.items():
                size = sum(counts.values())
                for exposure, count in counts.items():
                    composition.append(
                        {
                            **base,
                            "cluster_id": cluster_id,
                            "n_cases": size,
                            "evidence_type": "Exposure",
                            "evidence_label": exposure,
                            "n_evidence": count,
                            "fraction_in_cluster": count / size,
                        }
                    )
                composition.append(
                    {
                        **base,
                        "cluster_id": cluster_id,
                        "n_cases": size,
                        "evidence_type": "Clade",
                        "evidence_label": "20A",
                        "n_evidence": size,
                        "fraction_in_cluster": 1,
                    }
                )
            snf_size = 3 if other_exposures else 2
            for exposure, cluster_n, exposed_n in (("Conference", 3, 2), ("SNF", snf_size, 1)):
                named.append(
                    {
                        **base,
                        "exposure": exposure,
                        "n_cases": cluster_n,
                        "n_exposure": exposed_n,
                        "exposure_total": 2,
                        "exposure_fraction": exposed_n / cluster_n,
                        "exposure_recovery": exposed_n / 2,
                    }
                )
    graph_ids = [
        key for key, definition in settings.items() if definition["kind"] != "treecluster"
    ]
    tree_ids = [
        key for key, definition in settings.items() if definition["kind"] == "treecluster"
    ]
    for folder, count in (("clusters", len(graph_ids)), ("trees", len(tree_ids))):
        (tmp_path / folder / "status.json").write_text(
            json.dumps(
                {"status": "complete", "configured": count, "completed": count, "errors": []}
            )
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
    pd.DataFrame(composition).to_csv(
        tmp_path / "assessment/cluster_composition.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "setting_id": graph,
                "tree_setting_id": tree,
                "n_cases": n_cases,
                "adjusted_rand": 0.1 * (tree_index % 4 + 1),
                "adjusted_mutual_information": 0.02 * (graph_index % 14 + 1),
            }
            for graph_index, graph in enumerate(graph_ids)
            for tree_index, tree in enumerate(tree_ids)
        ]
    ).to_csv(tmp_path / "assessment/tree_agreement.csv", index=False)
    return manuscript.BostonStudy.load(tmp_path)


@pytest.mark.parametrize("criterion", ["balanced_M0", "balanced_Mle1", "balanced_Mle2"])
def test_all_cluster_table_keeps_every_exposure_cluster_and_singleton(tmp_path, criterion):
    table = importlib.import_module("evaluation.results.tab05")
    study = completed_study(tmp_path)
    frame = table.cluster_exposure_rows(study, criterion)
    assert len(frame) == 54
    assert frame.setting_id.nunique() == 18
    assert set(frame.criterion) == {criterion}
    assert set(frame.exposure) == {"Conference", "SNF"}
    assert not frame.duplicated(["setting_id", "cluster_id", "exposure"]).any()
    assert frame.n_cases.eq(1).sum() == 18
    point = study.point("components/EDD", criterion)
    subset = frame.loc[frame.setting_id == point["setting_id"]]
    assert subset.cluster_id.tolist() == [0, 1, 2]
    assert subset.n_cases.tolist() == [3, 2, 1]
    assert subset.n_exposure.tolist() == [2, 1, 1]
    np.testing.assert_allclose(subset.concentration_percent, [100 * 2 / 3, 50, 100])
    np.testing.assert_allclose(subset.recovery_percent, [100, 50, 50])
    assert all(len(row) == 9 for row in table.build_rows(frame))


def test_cluster_table_limits_exposures_but_preserves_full_cluster_sizes(tmp_path):
    table = importlib.import_module("evaluation.results.tab05")
    study = completed_study(tmp_path, other_exposures=True)
    frame = table.cluster_exposure_rows(study)
    assert len(frame) == 54
    assert set(frame.exposure) == {"Conference", "SNF"}
    # Clusters with only unlabelled, BHCHP, or City sequences are excluded.
    assert set(frame.cluster_id) == {0, 1, 2}
    assert len(frame.drop_duplicates(["setting_id", "cluster_id"])) == 54
    point = study.point("components/EDD")
    mixed = frame.loc[(frame.setting_id == point["setting_id"]) & frame.cluster_id.eq(1)]
    assert mixed.exposure.tolist() == ["SNF"]
    assert mixed.n_cases.eq(3).all()
    np.testing.assert_allclose(mixed.concentration_percent, 100 / 3)
    assert mixed.loc[mixed.exposure == "SNF", "recovery_percent"].iloc[0] == 50
    tex_path, csv_path = table.create_table(study, tmp_path / "tables")
    assert tex_path.exists() and csv_path.exists()
    caption = tex_path.read_text()
    assert "Clusters of one sequence are included" in caption
    assert "Conference or SNF" in caption
    assert "BHCHP" not in caption
    assert "\\begin{longtable}" in caption
    exported = pd.read_csv(csv_path)
    pd.testing.assert_frame_equal(exported, frame, check_dtype=False)


@pytest.mark.parametrize(
    "damage", ["duplicate", "missing_setting", "missing_cluster", "missing_count", "fraction", "pipeline", "totals"]
)
def test_all_cluster_table_rejects_incomplete_or_inconsistent_composition(tmp_path, damage):
    table = importlib.import_module("evaluation.results.tab05")
    study = completed_study(tmp_path)
    path = tmp_path / "assessment/cluster_composition.csv"
    composition = pd.read_csv(path)
    identifier = study.point("components/EDD")["setting_id"]
    if damage == "duplicate":
        composition = pd.concat([composition, composition.iloc[[0]]], ignore_index=True)
    elif damage == "missing_setting":
        composition = composition.loc[composition.setting_id != identifier]
    elif damage == "missing_cluster":
        composition = composition.loc[
            ~((composition.setting_id == identifier) & composition.cluster_id.eq(2))
        ]
    elif damage == "missing_count":
        composition = composition.loc[
            ~(
                (composition.setting_id == identifier) & composition.cluster_id.eq(0)
                & composition.evidence_type.eq("Exposure") & composition.evidence_label.eq("Unlabeled")
            )
        ]
    elif damage == "fraction":
        composition.loc[0, "fraction_in_cluster"] = 0.9
    elif damage == "pipeline":
        composition.loc[0, "pipeline"] = "leiden/EDD"
    else:
        composition.loc[0, "evidence_label"] = "SNF"
    composition.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Boston.*(exposure|cluster|composition)"):
        table.cluster_exposure_rows(study)


def test_focused_boston_outputs_keep_both_exposure_denominators(tmp_path):
    study = completed_study(tmp_path)
    focus = study.focus_rows()
    assert len(focus) == 10
    assert focus.loc[0, "n_exposure"] == 2
    assert focus.loc[0, "exposure_fraction"] == pytest.approx(2 / 3)
    assert focus.loc[0, "exposure_recovery"] == 1
    assert focus.loc[1, "exposure_fraction"] == 0.5
    assert focus.loc[1, "exposure_recovery"] == 0.5
    assert set(focus.pipeline) == {
        "leiden/ESD",
        "leiden/LOGIT_D",
        "components/GD_D",
        "treecluster/empirical/deterministic/raw",
        "treecluster/empirical/deterministic/dated",
    }
    rows = importlib.import_module("evaluation.results.tab04").build_rows(study)
    assert len(rows) == 10
    assert all(len(row) == 10 for row in rows)
    assert rows[0][1:5] == ["CPM resolution 0.1", "Conference", "2 / 2", "3"]
    assert rows[4][:2] == ["Connected components GDD", "GD <= 0 SNP"]
    assert "Maximum clade" in rows[6][0] and "D source" in rows[6][0]
    assert rows[6][1] == "5 SNP"
    assert rows[8][1] == "50 d"


def test_partition_context_groups_all_methods_without_agreement_panel(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    context = importlib.import_module("evaluation.results.fig18")
    study = completed_study(tmp_path)
    study.agreement = pd.DataFrame()
    rows = context.partition_rows(study)
    assert rows.groupby("kind").size().to_dict() == {
        "components": 8,
        "leiden": 6,
        "treecluster": 4,
    }
    assert rows.n_cases.eq(6).all()
    assert rows.n_singleton_cases.eq(1).all()
    assert rows.largest_cluster.eq(3).all()
    np.testing.assert_allclose(rows.singleton_percent, 100 / 6)
    np.testing.assert_allclose(rows.largest_percent, 50)

    figures = []

    def capture_figure(fig, *args, **kwargs):
        figures.append(fig)
        fig.canvas.draw()
        return {}

    monkeypatch.setattr(context.style, "save_figure", capture_figure)
    try:
        context.create_figure(study, tmp_path)
        fig = figures[0]
        assert len(fig.axes) == 3
        assert not any(ax.images for ax in fig.axes)
        for ax, count in zip(fig.axes, (6, 8, 4)):
            assert len(ax.get_yticklabels()) == count
            assert len(ax.patches) == 2 * count
            np.testing.assert_allclose(
                [patch.get_width() for patch in ax.patches[:count]], 100 / 6
            )
            np.testing.assert_allclose(
                [patch.get_width() for patch in ax.patches[count:]], 50
            )
        plotted = pd.read_csv(tmp_path / "fig18_boston_partition_context.csv")
        assert set(plotted.setting_id) == set(rows.setting_id)
    finally:
        for fig in figures:
            plt.close(fig)


@pytest.mark.parametrize("missing_cluster", [False, True])
def test_boston_tradeoffs_plot_all_frozen_methods_with_separate_legends(
    tmp_path, monkeypatch, missing_cluster
):
    import matplotlib.pyplot as plt

    tradeoffs = importlib.import_module("evaluation.results.fig17")
    study = completed_study(tmp_path)
    point = study.point("components/EDD")  # Outside the old five-method focus.
    if missing_cluster:
        study.named = study.named.loc[
            ~(
                (study.named.setting_id == point["setting_id"])
                & (study.named.exposure == "SNF")
            )
        ]
    rows = tradeoffs.exposure_rows(study)
    assert len(rows) == 36
    assert set(rows.pipeline) == {
        point["pipeline"] for point in study.selected_points("balanced_M0")
    }
    assert rows.groupby("method").size().to_dict() == {
        "components": 16,
        "leiden": 12,
        "raw": 4,
        "dated": 4,
    }
    assert set(rows.criterion) == {"balanced_M0"}
    assert set(rows.baseline_data_process) == {"deterministic", "stochastic"}
    assert rows.filled.eq(rows.baseline_data_process == "stochastic").all()
    assert rows.exposure_recovery.isna().sum() == int(missing_cluster)
    if missing_cluster:
        missing = rows.loc[
            (rows.pipeline == "components/EDD") & (rows.exposure == "SNF")
        ]
        assert missing.exposure_fraction.isna().all()

    figures = []

    def capture_figure(fig, *args, **kwargs):
        fig.canvas.draw()
        figures.append(fig)
        return {}

    monkeypatch.setattr(tradeoffs.style, "save_figure", capture_figure)
    try:
        tradeoffs.create_figure(study, tmp_path)
        assert len(figures) == 1
        fig = figures[0]
        assert [len(ax.collections) for ax in fig.axes] == [18, 18 - int(missing_cluster)]
        assert len(fig.legends) == 2
        assert [len(legend.get_texts()) for legend in fig.legends] == [4, 4]
        for ax, exposure in zip(fig.axes, ("Conference", "SNF")):
            visible = rows.loc[rows.exposure == exposure].dropna(
                subset=["exposure_fraction", "exposure_recovery"]
            ).sort_values("filled", ascending=False)
            for collection, row in zip(ax.collections, visible.itertuples(index=False)):
                assert bool(len(collection.get_facecolors())) == row.filled
        assert "Synthetic source rule" in fig.legends[0].get_title().get_text()
        for legend in fig.legends:
            positions = [text.get_window_extent().y0 for text in legend.get_texts()]
            np.testing.assert_allclose(positions, [positions[0]] * 4, atol=1)
        plotted = pd.read_csv(tmp_path / "fig17_boston_exposure_tradeoffs.csv")
        assert set(plotted.setting_id) == set(rows.setting_id)
        assert plotted.exposure_recovery.isna().sum() == int(missing_cluster)
    finally:
        for fig in figures:
            plt.close(fig)


@pytest.mark.parametrize("criterion", ["balanced_M0", "balanced_Mle1", "balanced_Mle2"])
def test_supplements_include_all_current_graph_and_tree_rules(tmp_path, criterion):
    study = completed_study(tmp_path)
    points = study.selected_points(criterion)
    assert len(points) == 18
    assert {point["criterion"] for point in points} == {criterion}
    labels, concentration, recovery = importlib.import_module(
        "evaluation.results.fig19"
    ).all_rows(study, criterion)
    assert concentration.shape == recovery.shape == (18, 2)
    assert sum(label.startswith("Connected components") for label in labels) == 8
    assert sum(label.startswith("Leiden") for label in labels) == 6
    assert sum(label.startswith("TreeCluster") for label in labels) == 4
    assert all("\n" not in label and "CPM" not in label for label in labels)
    resolution = {"balanced_M0": "0.10", "balanced_Mle1": "0.20", "balanced_Mle2": "0.30"}[
        criterion
    ]
    assert all(
        label.endswith(f"; resolution {resolution}")
        for label in labels if label.startswith("Leiden")
    )
    assert not any(
        "GDD" in label or "GDS" in label for label in labels if label.startswith("Leiden")
    )
    np.testing.assert_allclose(concentration[:, 0], 100 * 2 / 3)
    np.testing.assert_allclose(recovery[:, 1], 50)
    values, graph_labels, tree_labels = importlib.import_module(
        "evaluation.results.fig20"
    ).agreement_grid(study, criterion)
    assert len(graph_labels) == 14
    assert len(tree_labels) == 4
    assert all(matrix.shape == (14, 4) for matrix in values.values())
    assert "D source" in tree_labels[0] and "S source" in tree_labels[1]
    assert "Maximum clade" in tree_labels[0] and "Average clade" in tree_labels[1]
    assert "SNP" in tree_labels[0] and " d" in tree_labels[2]
    assert "subs/site" not in tree_labels[0]  # Raw transfer preserves SNP counts.


def test_missing_focused_genetic_comparator_is_not_silently_dropped(tmp_path):
    study = completed_study(tmp_path)
    del study.selected["balanced_M0", "components/GD_D"]
    with pytest.raises(ValueError, match="No frozen Boston setting.*components/GD_D"):
        study.focus_rows()


@pytest.mark.parametrize(
    "field,value",
    [("graph_mode", "thresholded"), ("threshold", 0.5), ("score_name", "GD_D")],
)
def test_boston_displays_require_current_full_graph_leiden_design(tmp_path, field, value):
    study = completed_study(tmp_path)
    point = study.point("leiden/ESD")
    selection_path = tmp_path / "selection.json"
    selection = json.loads(selection_path.read_text())
    for saved in selection["operating_points"]:
        if saved["setting_id"] == point["setting_id"]:
            saved["definition"][field] = value
            study.settings[point["setting_id"]] = saved["definition"]
    selection_path.write_text(json.dumps(selection))
    (tmp_path / "settings.json").write_text(json.dumps(study.settings))
    with pytest.raises(ValueError, match="full similarity-weighted graphs"):
        manuscript.BostonStudy.load(tmp_path)


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
    labels, concentration, recovery = importlib.import_module(
        "evaluation.results.fig19"
    ).all_rows(study, "balanced_M0")
    index = study.selected_points("balanced_M0").index(point)
    assert np.isnan(concentration[index, 1]) and np.isnan(recovery[index, 1])
    table_rows = importlib.import_module("evaluation.results.tab04").build_rows(study)
    assert table_rows[1][-2:] == ["--", "--"]
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
