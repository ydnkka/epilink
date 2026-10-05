"""Check manuscript extraction against frozen evidence and native threshold units."""

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

manuscript = importlib.import_module("evaluation.01_synthetic_baseline.manuscript_common")


def test_operating_table_requires_every_frozen_heldout_seed(tmp_path):
    (tmp_path / "evaluation").mkdir()
    point = {
        "criterion": "balanced_M0", "pipeline": "pairwise/GD_D", "setting_id": "fixed",
        "status": "selected", "rule": {"objective": "M0_f1"},
        "definition": {"kind": "pairwise", "score_name": "GD_D", "data_process": "deterministic"},
    }
    (tmp_path / "evaluation/selection_used.json").write_text(
        json.dumps({"operating_points": [point]})
    )
    pd.DataFrame([{
        "criterion": "balanced_M0", "pipeline": "pairwise/GD_D", "setting_id": "fixed",
        "objective": "M0_f1", "objective_endpoint": "M0", "n_realizations": 2,
    }]).to_csv(tmp_path / "evaluation/operating_summary.csv", index=False)
    results = pd.DataFrame([
        {"criterion": "balanced_M0", "pipeline": "pairwise/GD_D", "setting_id": "fixed", "seed": seed}
        for seed in (11, 12)
    ])
    results.to_csv(tmp_path / "evaluation/operating_results.csv", index=False)
    config = {"splits": {"evaluation": [11, 12]}}
    summary, points = manuscript.selected_summary(tmp_path, config)
    assert summary.loc["pairwise/GD_D", "setting_id"] == points["pairwise/GD_D"]["setting_id"]

    results.loc[1, "seed"] = 13
    results.to_csv(tmp_path / "evaluation/operating_results.csv", index=False)
    with pytest.raises(ValueError, match="Incomplete or unfrozen"):
        manuscript.selected_summary(tmp_path, config)


def test_table_preserves_units_and_undefined_metrics():
    config = {"simulation": {"sequence_length": 5000}}
    assert manuscript.setting_label(
        {"kind": "treecluster", "tree_kind": "raw", "threshold": 4 / 5000}, config
    ) == "4 SNP"
    assert manuscript.setting_label(
        {"kind": "treecluster", "tree_kind": "dated", "threshold": 42}, config
    ) == "42 d"
    row = pd.Series({
        "n_realizations": 3, "M0_precision_mean": 0.2123,
        "M0_precision_std": 0.0101, "M0_precision_count": 2,
        "largest_cluster_fraction_mean": np.nan,
        "largest_cluster_fraction_std": np.nan,
        "largest_cluster_fraction_count": 0,
    })
    assert manuscript.format_metric(row, "M0_precision") == "21.2 (1.0) [n=2]"
    assert manuscript.format_metric(row, "largest_cluster_fraction") == "--"


def test_display_envelopes_discard_dominated_settings(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(manuscript.__file__).parent))
    plots = importlib.import_module("manuscript_plots")
    points = pd.DataFrame({
        "recall": [0.9, 0.8, 0.6, 0.5, 0.6],
        "precision": [0.3, 0.2, 0.7, 0.6, 0.7],
        "contamination": [0.05, 0.10, 0.20, 0.30, 0.20],
        "f1": [0.3, 0.5, 0.4, 0.7, 0.4],
    })
    pr = plots.tradeoff_frontier(points, x="recall", y="precision", minimize_x=False)
    assert list(pr.itertuples(index=False, name=None)) == [(0.6, 0.7), (0.9, 0.3)]
    distant = plots.tradeoff_frontier(
        points, x="contamination", y="f1", minimize_x=True
    )
    assert list(distant.itertuples(index=False, name=None)) == [
        (0.05, 0.3), (0.1, 0.5), (0.3, 0.7)
    ]


def test_display_variants_match_available_pipelines(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(manuscript.__file__).parent))
    bars = importlib.import_module("manuscript_bars")
    assert [label for label, _ in bars.graph_variants(
        "leiden_native", "stochastic"
    )] == ["EDS", "ESS", "LGS"]
    assert [label for label, _ in bars.graph_variants(
        "components", "deterministic"
    )] == ["EDD", "ESD", "GDD", "LGD"]
    points = {"treecluster/deterministic/raw": {
        "status": "selected", "definition": {
            "kind": "treecluster", "tree_kind": "raw", "data_process": "deterministic",
            "method": "avg_clade", "threshold": 4 / 5000,
        },
    }}
    assert bars.treecluster_variant("raw", "deterministic", points, {
        "simulation": {"sequence_length": 5000}
    }) == ("Deterministic\nAvg clade, 4 SNP", "treecluster/deterministic/raw")
