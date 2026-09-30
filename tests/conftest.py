from pathlib import Path

import networkx as nx
import pytest

from epilink_evaluation.cli import smoke_config
from epilink_evaluation.config import load_config


@pytest.fixture
def small_config(tmp_path):
    """A portable integration experiment; no preserved inputs or external tools."""
    root = Path(__file__).resolve().parents[1]
    config = smoke_config(load_config(root / "synthetic_baseline/config.yaml"))
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
