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
        "GDS",
        "LGS",
    ]
    assert [
        label for label, _ in bars.graph_variants("components", "deterministic")
    ] == ["EDD", "ESD", "GDD", "LGD"]
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


def test_shared_resolution_regret_uses_full_graph_reference_and_minimax():
    regret = importlib.import_module("evaluation.results.fig06")
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
    details, summary, reference = regret.regret_tables(
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
