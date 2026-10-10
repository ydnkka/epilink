from copy import deepcopy
from pathlib import Path

import networkx as nx
import pytest

from epilink_evaluation.cli import smoke_config
from epilink_evaluation.config import load_config


@pytest.fixture
def small_config(tmp_path):
    """A portable integration experiment; no preserved inputs or external tools."""
    root = Path(__file__).resolve().parents[1]
    config = smoke_config(
        load_config(root / "evaluation/01_synthetic_baseline/config.yaml")
    )
    # Resolved fixtures can be serialized as standalone configs after local edits.
    config.pop("experiment_config", None)
    config["experiment_root"] = str(tmp_path / "shared")
    tree = nx.balanced_tree(2, 3, create_using=nx.DiGraph)
    tree = nx.relabel_nodes(tree, lambda node: f"case_{node}")
    path = tmp_path / "backbone.gml"
    nx.write_gml(tree, path)
    config["inputs"].update(tree_path=str(path), smoke_cases=None)
    config["output_directory"] = str(tmp_path / "outputs")
    config["simulation"]["sequence_length"] = 128
    config["scorers"] = ["GD_D", "GD_S", "LOGIT_D", "LOGIT_S"]
    config["clustering"]["algorithms"] = ["components"]
    config["treecluster"]["enabled"] = False
    return config


@pytest.fixture
def prepare_diagnostics(tmp_path, monkeypatch):
    """Explicit development producer with small, tool-free diagnostic controls."""
    from epilink_evaluation.reporting import diagnostics as reporting
    from epilink_evaluation.workflows.diagnostics import Diagnostics

    def prepare(config):
        diagnostic_config = deepcopy(config)
        diagnostic_config["output_directory"] = str(tmp_path / "diagnostics")
        diagnostic_config.setdefault(
            "diagnostics",
            {
                "leiden": {
                    "objective": "CPM",
                    "resolution_grid": [0.2, 0.8],
                    "restarts": 1,
                    "seed": 65001,
                },
                "treecluster": {
                    "enabled": False,
                    "methods": ["max_clade"],
                    "threshold_hops": [0, 2],
                    "command_timeout_seconds": 30,
                    "executables": {"treecluster": "TreeCluster.py"},
                },
            },
        )
        diagnostics = Diagnostics(diagnostic_config)
        # Real reporting is exercised in test_diagnostics_workflow.py. Keep this
        # preparation helper cheap without skipping computation or completion checks.
        with monkeypatch.context() as patch:
            patch.setattr(reporting, "render_report", lambda directory: None)
            assert diagnostics.run("all")
        diagnostics.exp.require_diagnostics()
        return diagnostics

    return prepare
