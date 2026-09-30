"""Tests for Boston empirical clustering workflow."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from epilink_evaluation.provenance import read_json
from epilink_evaluation.workflows.boston import BostonEmpirical
from epilink_evaluation.workflows.boston_config import load_study_config


@pytest.fixture
def evaluated_baseline(small_config, tmp_path):
    from epilink_evaluation.workflows.baseline import Baseline

    small_config["scorers"] = ["EDD", "GD_D"]
    small_config["clustering"]["algorithms"] = ["components"]
    small_config["treecluster"]["enabled"] = False
    baseline = Baseline(small_config)
    assert baseline.run("all")
    return baseline


def boston_config(tmp_path, reference, *, smoke=False):
    cases_path = tmp_path / "boston_cases.parquet"
    pairs_path = tmp_path / "boston_pairs.parquet"

    cases = pd.DataFrame({
        "case_id": ["A", "B", "C", "D", "E"],
        "sample_date": pd.date_range("2020-03-01", periods=5),
        "Exposure": ["Conference", "SNF", "BHCHP", "City", "Unlabeled"],
    })
    cases.to_parquet(cases_path, index=False)

    pairs = pd.DataFrame({
        "CaseID1": ["A", "A", "B", "B", "C"],
        "CaseID2": ["B", "C", "C", "D", "D"],
        "TD": [1.0, 2.0, 1.0, 3.0, 1.0],
        "GD": [2.0, 5.0, 3.0, 8.0, 1.0],
        "TN93_distance": [2.0 / 29903, 5.0 / 29903, 3.0 / 29903, 8.0 / 29903, 1.0 / 29903],
    })
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
        "seeds": [82001],
        "scorers": ["EDD", "GD_D"],
        "clustering": {
            "algorithms": ["components"],
            "leiden": {"objective": "CPM", "weight_policies": ["binary"], "resolutions": [0.1], "restarts": 2, "seed": 65001},
        },
        "treecluster": {"enabled": False},
    }
    path = tmp_path / "boston.yaml"
    path.write_text(yaml.safe_dump(config))
    return load_study_config(argv=[], config_path=path)


def test_boston_workflow_runs_smoke(evaluated_baseline, tmp_path):
    config = boston_config(tmp_path, evaluated_baseline, smoke=True)
    study = BostonEmpirical(config)
    assert study.n_cases == 5
    assert study.n_observed_pairs == 5
    assert study.run()
    assert read_json(study.directory / "manifest.json")["status"] == "complete"
    assert (study.directory / "clusters" / "status.json").exists()
    cluster_status = read_json(study.directory / "clusters" / "status.json")
    assert cluster_status["status"] == "complete"
    assert (study.directory / "report.md").exists()
    assert (study.directory / "report.html").exists()


def test_boston_config_validation(evaluated_baseline, tmp_path):
    config = boston_config(tmp_path, evaluated_baseline)
    assert config["schema_version"] == 1
    assert len(config["seeds"]) == 1
    assert config["seeds"][0] == 82001


def test_boston_requires_fresh_seeds(evaluated_baseline, tmp_path):
    config = boston_config(tmp_path, evaluated_baseline)
    config["seeds"] = evaluated_baseline.config["splits"]["train"]
    with pytest.raises(ValueError, match="fresh"):
        BostonEmpirical(config)
