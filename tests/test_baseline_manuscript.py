"""Check manuscript extraction against frozen evidence and original threshold units."""

import importlib
import json

import numpy as np
import pandas as pd
import pytest

manuscript = importlib.import_module("evaluation.results._baseline.common")


def test_operating_table_requires_every_frozen_heldout_seed(tmp_path):
    (tmp_path / "evaluation").mkdir()
    point = {
        "criterion": "balanced_M0",
        "pipeline": "pairwise/GD_D",
        "setting_id": "fixed",
        "status": "selected",
        "rule": {"objective": "M0_f1"},
        "definition": {
            "kind": "pairwise",
            "score_name": "GD_D",
            "data_process": "deterministic",
        },
    }
    (tmp_path / "evaluation/selection_used.json").write_text(
        json.dumps({"operating_points": [point]})
    )
    pd.DataFrame(
        [
            {
                "criterion": "balanced_M0",
                "pipeline": "pairwise/GD_D",
                "setting_id": "fixed",
                "objective": "M0_f1",
                "objective_endpoint": "M0",
                "n_realizations": 2,
            }
        ]
    ).to_csv(tmp_path / "evaluation/operating_summary.csv", index=False)
    results = pd.DataFrame(
        [
            {
                "criterion": "balanced_M0",
                "pipeline": "pairwise/GD_D",
                "setting_id": "fixed",
                "seed": seed,
            }
            for seed in (11, 12)
        ]
    )
    results.to_csv(tmp_path / "evaluation/operating_results.csv", index=False)
    config = {"splits": {"evaluation": [11, 12]}}
    summary, points = manuscript.selected_summary(tmp_path, config)
    assert (
        summary.loc["pairwise/GD_D", "setting_id"]
        == points["pairwise/GD_D"]["setting_id"]
    )

    results.loc[1, "seed"] = 13
    results.to_csv(tmp_path / "evaluation/operating_results.csv", index=False)
    with pytest.raises(ValueError, match="Incomplete or unfrozen"):
        manuscript.selected_summary(tmp_path, config)


def test_table_preserves_units_and_undefined_metrics():
    config = {"simulation": {"sequence_length": 5000}}
    assert (
        manuscript.setting_label(
            {"kind": "treecluster", "tree_kind": "raw", "threshold": 4 / 5000}, config
        )
        == "4 SNP"
    )
    assert (
        manuscript.setting_label(
            {"kind": "treecluster", "tree_kind": "dated", "threshold": 42}, config
        )
        == "42 d"
    )
    row = pd.Series(
        {
            "n_realizations": 3,
            "M0_precision_mean": 0.2123,
            "M0_precision_std": 0.0101,
            "M0_precision_count": 2,
            "largest_cluster_fraction_mean": np.nan,
            "largest_cluster_fraction_std": np.nan,
            "largest_cluster_fraction_count": 0,
        }
    )
    assert manuscript.format_metric(row, "M0_precision") == "21.2 (1.0) [n=2]"
    assert manuscript.format_metric(row, "largest_cluster_fraction") == "--"


def test_display_envelopes_discard_dominated_settings():
    plots = importlib.import_module("evaluation.results._baseline.plots")
    points = pd.DataFrame(
        {
            "recall": [0.9, 0.8, 0.6, 0.5, 0.6],
            "precision": [0.3, 0.2, 0.7, 0.6, 0.7],
            "contamination": [0.05, 0.10, 0.20, 0.30, 0.20],
            "f1": [0.3, 0.5, 0.4, 0.7, 0.4],
        }
    )
    pr = plots.tradeoff_frontier(points, x="recall", y="precision", minimize_x=False)
    assert list(pr.itertuples(index=False, name=None)) == [(0.6, 0.7), (0.9, 0.3)]
    distant = plots.tradeoff_frontier(
        points, x="contamination", y="f1", minimize_x=True
    )
    assert list(distant.itertuples(index=False, name=None)) == [
        (0.05, 0.3),
        (0.1, 0.5),
        (0.3, 0.7),
    ]


def test_display_variants_match_available_pipelines():
    bars = importlib.import_module("evaluation.results._baseline.bars")
    assert [label for label, _ in bars.graph_variants("leiden", "stochastic")] == [
        "EDS",
        "ESS",
        "LGS",
    ]
    assert bars.graph_variants("leiden", "deterministic") == [
        ("EDD", "leiden/EDD"),
        ("ESD", "leiden/ESD"),
        ("LGD", "leiden/LOGIT_D"),
    ]
    assert [
        label for label, _ in bars.graph_variants("components", "deterministic")
    ] == ["EDD", "ESD", "GDD", "LGD"]
    assert [
        label for label, _ in bars.graph_variants("components", "stochastic")
    ] == ["EDS", "ESS", "GDS", "LGS"]
    points = {
        "treecluster/deterministic/raw": {
            "status": "selected",
            "definition": {
                "kind": "treecluster",
                "tree_kind": "raw",
                "data_process": "deterministic",
                "method": "avg_clade",
                "threshold": 4 / 5000,
            },
        }
    }
    assert bars.treecluster_variant(
        "raw", "deterministic", points, {"simulation": {"sequence_length": 5000}}
    ) == ("Deterministic\nAverage clade, 4 SNP", "treecluster/deterministic/raw")


@pytest.mark.parametrize("missing", [None, "leiden/LOGIT_D", "components/GD_D"])
def test_cluster_comparison_uses_supported_frozen_pipelines(tmp_path, monkeypatch, missing):
    comparison = importlib.import_module("evaluation.results.fig12")
    table = importlib.import_module("evaluation.results.tab01")
    pipelines_by_process = {
        "deterministic": [
            "components/EDD",
            "components/ESD",
            "components/GD_D",
            "components/LOGIT_D",
            "leiden/EDD",
            "leiden/ESD",
            "leiden/LOGIT_D",
            "treecluster/deterministic/raw",
            "treecluster/deterministic/dated",
        ],
        "stochastic": [
            "components/EDS",
            "components/ESS",
            "components/GD_S",
            "components/LOGIT_S",
            "leiden/EDS",
            "leiden/ESS",
            "leiden/LOGIT_S",
            "treecluster/stochastic/raw",
            "treecluster/stochastic/dated",
        ],
    }
    points, results = {}, []
    for process, pipelines in pipelines_by_process.items():
        for pipeline in pipelines:
            points[pipeline] = {
                "status": "selected",
                "setting_id": pipeline,
                "definition": {"data_process": process},
            }
            results.append(
                {
                    "pipeline": pipeline,
                    "M0_f1_mean": 0.4,
                    "M0_f1_std": 0.05,
                    "M0_f1_count": 3,
                    "Mge3_contamination_mean": 0.2,
                    "Mge3_contamination_std": 0.01,
                    "Mge3_contamination_count": 3,
                }
            )
    if missing:
        del points[missing]
    summary = pd.DataFrame(results).set_index("pipeline")
    monkeypatch.setattr(comparison, "selected_summary", lambda *args: (summary, points))
    if missing:
        with pytest.raises(ValueError, match=f"Missing selected cluster method: {missing}"):
            comparison.cluster_rows(None, {})
    else:
        rows = comparison.cluster_rows(None, {})
        assert len(rows) == 18
        for process, pipelines in pipelines_by_process.items():
            assert rows.loc[rows.process == process, "pipeline"].tolist() == pipelines
            assert table.main_pipelines(process) == [
                pipeline.replace("components/", "pairwise/") for pipeline in pipelines
            ]
        assert rows.f1_percent.eq(40).all()
        assert rows.distant_percent.eq(20).all()
        assert not rows.loc[rows.process == "deterministic", "filled"].any()
        assert rows.loc[rows.process == "stochastic", "filled"].all()

        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        figures = []
        monkeypatch.setattr(
            comparison.style,
            "save_figure",
            lambda fig, *args, **kwargs: figures.append(fig) or {},
        )
        try:
            comparison.create_figure(None, {}, tmp_path)
            fig = figures[0]
            fig.canvas.draw()
            assert [len(ax.collections) for ax in fig.axes] == [9, 9]
            for ax, filled in zip(fig.axes, (False, True)):
                assert all(
                    bool(len(point.get_facecolors())) == filled for point in ax.collections
                )
            markers = fig.legends[0].findobj(match=Line2D)
            assert [marker.get_markerfacecolor() for marker in markers] == [
                "none", "0.2"
            ] * 4
            assert "Genetic observations" in fig.legends[0].get_title().get_text()
            for legend in fig.legends:
                positions = [text.get_window_extent().y0 for text in legend.get_texts()]
                np.testing.assert_allclose(positions, [positions[0]] * 4, atol=1)
        finally:
            for fig in figures:
                plt.close(fig)


@pytest.mark.parametrize("missing", [None, "development", "selection"])
def test_leiden_figure_requires_only_supported_scorers(tmp_path, monkeypatch, missing):
    import matplotlib.pyplot as plt

    plots = importlib.import_module("evaluation.results._baseline.plots")
    scores_by_process = {
        "deterministic": ("EDD", "ESD", "LOGIT_D"),
        "stochastic": ("EDS", "ESS", "LOGIT_S"),
    }
    definitions, points, rows = {}, {}, []
    for process, scores in scores_by_process.items():
        for score in scores:
            pipeline = f"leiden/{score}"
            definition = {
                "kind": "leiden",
                "pipeline": pipeline,
                "score_name": score,
                "data_process": process,
                "graph_mode": "full",
                "threshold": None,
                "resolution": 0.1,
            }
            definitions[score] = definition
            points[pipeline] = {
                "status": "selected",
                "pipeline": pipeline,
                "setting_id": score,
                "definition": definition,
            }
            rows.append(
                {
                    "pipeline": pipeline,
                    "setting_id": score,
                    "n_realizations": 2,
                    "M0_precision_mean": 0.6,
                    "M0_recall_mean": 0.4,
                    "M0_f1_mean": 0.48,
                    "Mge3_contamination_mean": 0.1,
                }
            )
    if missing == "development":
        rows = [row for row in rows if row["pipeline"] != "leiden/LOGIT_D"]
    elif missing == "selection":
        del points["leiden/LOGIT_D"]
    (tmp_path / "development").mkdir()
    (tmp_path / "settings.json").write_text(json.dumps(definitions))
    pd.DataFrame(rows).to_csv(tmp_path / "development/summary.csv", index=False)
    config = {"splits": {"development": [11, 12]}}
    monkeypatch.setattr(plots, "selected_summary", lambda *args: (pd.DataFrame(), points))
    figures = []
    monkeypatch.setattr(
        plots.style, "save_figure", lambda fig, *args, **kwargs: figures.append(fig) or {}
    )
    existing_figures = set(plt.get_fignums())
    try:
        if missing:
            with pytest.raises(ValueError, match="Missing leiden.*leiden/LOGIT_D"):
                plots.create_approach_figure(tmp_path, config, tmp_path, "leiden")
        else:
            plots.create_approach_figure(tmp_path, config, tmp_path, "leiden")
            assert len(figures) == 1
            fig = figures[0]
            fig.canvas.draw()
            for ax, scores in zip(fig.axes[:2], scores_by_process.values()):
                labels = [text.get_text() for text in ax.get_legend().get_texts()]
                assert labels == [
                    *[manuscript.SCORE_DESCRIPTIONS[score] for score in scores],
                    "Selected during development",
                ]
                assert len(ax.lines) == 3
                assert len(ax.collections) == 6  # Sweep and frozen point per scorer.
    finally:
        for number in set(plt.get_fignums()) - existing_figures:
            plt.close(number)


def test_shared_resolution_regret_uses_full_graph_reference_and_minimax():
    regret = importlib.import_module("evaluation.results.fig07")
    specs = [
        ("b1", "EDD", 0.1, 0.8),
        ("b4", "EDD", 0.2, 0.6),
        ("nref", "ESD", 0.05, 0.8),
        ("n1", "ESD", 0.1, 0.5),
        ("n2", "ESD", 0.2, 0.65),
    ]
    definitions = {}
    rows = []
    for identifier, score, resolution, f1 in specs:
        pipeline = f"leiden/{score}"
        definitions[identifier] = {
            "kind": "leiden",
            "pipeline": pipeline,
            "score_name": score,
            "data_process": "deterministic",
            "resolution": resolution,
            "threshold": None,
            "graph_mode": "full",
        }
        for seed in (11, 12):
            rows.append(
                {
                    "split": "development",
                    "seed": seed,
                    "pipeline": pipeline,
                    "setting_id": identifier,
                    "M0_f1": f1,
                    **{name: 0.5 for name in regret.SECONDARY_METRICS},
                }
            )
    pipelines = ("leiden/EDD", "leiden/ESD")
    _, summary, reference = regret.regret_tables(
        pd.DataFrame(rows),
        definitions,
        {"name": "balanced_M0", "objective": "M0_f1", "constraints": {}},
        [11, 12],
        pipelines,
    )
    assert regret.common_resolutions(definitions, pipelines) == [0.1, 0.2]
    assert reference["leiden/ESD"]["definition"]["resolution"] == 0.05
    assert summary.loc[summary.resolution == 0.1, "mean_regret_pp"].iloc[
        0
    ] == pytest.approx(15)
    assert summary.loc[summary.resolution == 0.2, "mean_regret_pp"].iloc[
        0
    ] == pytest.approx(17.5)
    assert summary.loc[summary.selected_default, "resolution"].tolist() == [0.2]
    assert summary.loc[summary.selected_default, "max_regret_pp"].iloc[
        0
    ] == pytest.approx(20)
