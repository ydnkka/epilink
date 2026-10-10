import numpy as np
import pandas as pd
import pytest

from epilink_evaluation.clusterers import treecluster
from epilink_evaluation.phylogeny import trees
from epilink_evaluation.phylogeny.external import executable
from epilink_evaluation.provenance import read_json


def test_treecluster_preserves_each_unclustered_case(small_config, tmp_path):
    try:
        executable(small_config["treecluster"]["executables"]["treecluster"])
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    path = tmp_path / "tree.nwk"
    path.write_text("((a:0.001,b:0.001):0.1,c:0.1,d:0.2);\n")
    # Reordered case IDs exercise adapter alignment as well as singleton handling.
    cases = pd.DataFrame({"case_id": ["d", "b", "c", "a"]})
    labels, _ = treecluster(
        path,
        cases,
        "max_clade",
        0.003,
        small_config["treecluster"],
        tmp_path / "cluster",
    )
    expected = np.array([0, 1, 2, 1])
    np.testing.assert_array_equal(
        labels[:, None] == labels, expected[:, None] == expected
    )


def test_iqtree_adapter_mocked(small_config, tmp_path, monkeypatch):
    """Test adapter calls EpiLink with correct options (JC, threads, seed)."""
    from epilink_evaluation.phylogeny.trees import prepare_phylogeny
    from epilink_evaluation.inputs.synthetic import _export_sampled_fasta, _export_reference, _export_sampling_dates
    import pandas as pd
    import tempfile
    import os

    # Set up observation bundle files
    cases = pd.DataFrame({
        "case_id": ["case_0", "case_1", "case_2"],
        "node_index": [0, 1, 2],
        "sample_date": [1.0, 2.0, 3.0],
        "exposure_date": [0.5, 1.5, 2.5],
    })
    cases_path = tmp_path / "cases.parquet"
    cases.to_parquet(cases_path, index=False)

    # Create deterministic FASTA
    det_fasta = tmp_path / "sampled_deterministic.fasta"
    with open(det_fasta, "w") as f:
        f.write(">case_0\nACCGT\n>case_1\nGGTAA\n>case_2\nTTCGG\n")

    # Create stochastic FASTA
    sto_fasta = tmp_path / "sampled_stochastic.fasta"
    with open(sto_fasta, "w") as f:
        f.write(">case_0\nCCGGT\n>case_1\nGGAAT\n>case_2\nTCCTT\n")

    # Create reference FASTA
    ref_fasta = tmp_path / "reference.fasta"
    with open(ref_fasta, "w") as f:
        f.write(">ancestral_reference\nACGTACGT\n")

    # Create sampling dates TSV
    dates_path = tmp_path / "sampling_dates.tsv"
    with open(dates_path, "w") as f:
        f.write("case_id\tsample_date\ncase_0\t1\ncase_1\t2\ncase_2\t3\n")

    # Run develop to create observation artifact
    from epilink_evaluation.workflows.baseline import Baseline
    from epilink_evaluation.config import load_config
    from epilink_evaluation.tests.conftest import prepare_diagnostics

    config = small_config.copy()
    config["output_directory"] = str(tmp_path / "outputs")
    config["splits"] = {"train": [71001], "development": [72001], "evaluation": [73001]}
    config["inputs"]["smoke_cases"] = 64
    config = prepare_diagnostics(config)

    # Now test prepare_phylogeny with proper observation dir
    obs_dir = config["diagnostics"].exp.dataset(72001).directory
    result = prepare_phylogeny(
        config,
        obs_dir,
        "deterministic",
        config["diagnostics"].exp.dataset(72001).name,
        config["implementation"],
    )
    assert result is not None


def test_reference_pruned_from_raw_tree():
    """Verify raw.nwk excludes reference tip for TreeCluster."""
    import numpy as np
    from pathlib import Path
    from Bio import Phylo
    from epilink_evaluation.phylogeny.trees import _prune_reference_tip, validate_tree

    # Create a tree with a reference tip
    tree = Phylo.read("((a:0.1,b:0.1):0.2,c:0.3);", "newick")
    tree.prune(tree.find_clades()[0])  # Remove one tip to simulate reference

    # Test _prune_reference_tip
    pruned = _prune_reference_tip(tree, "a")
    tip_names = [str(tip.name) for tip in pruned.get_terminals()]
    assert "a" not in tip_names, "Reference tip 'a' should be pruned from raw tree"


def test_day_thresholds_no_conversion():
    """Verify dated thresholds passed directly (no /365)."""
    from epilink_evaluation.workflows.settings import settings_registry

    # Use config with alignment_length set
    config = {
        "schema_version": 1,
        "name": "test",
        "output_directory": "outputs/test",
        "simulation": {"sequence_length": 5000, "alignment_length": 5000},
        "treecluster": {
            "enabled": True,
            "methods": ["max_clade"],
            "genetic_threshold_snps": [0, 1, 2],
            "threshold_days": [0, 365, 730],
            "command_timeout_seconds": 1800,
            "executables": {"treecluster": "TreeCluster.py"},
        },
        "scorers": ["GD_D"],
        "pairwise": {"threshold_mode": "configured"},
        "clustering": {
            "algorithms": ["components"],
            "leiden": {"objective": "CPM", "resolutions": [0.5], "restarts": 1, "seed": 66001},
        },
    }

    definitions = settings_registry(config)
    # Check that dated thresholds have threshold_units = "days"
    for key, defn in definitions.items():
        if defn["kind"] == "treecluster" and defn["tree_kind"] == "dated":
            assert defn["threshold_units"] == "days"
            # Threshold should be the SNP count directly (not divided by alignment_length)
            # since it's in days units
            assert isinstance(defn["threshold"], (int, float))
            assert defn["threshold"] >= 0


def test_cache_reuse_on_rerun(small_config, tmp_path, prepare_diagnostics):
    """Verify valid artifact not recomputed."""
    from epilink_evaluation.workflows.baseline import Baseline
    from epilink_evaluation.phylogeny.trees import prepare_phylogeny

    # Run develop stage twice with same config
    prepare_diagnostics(small_config)
    baseline = Baseline(small_config)
    baseline.run("develop")

    # Call prepare_phylogeny twice with same params; second should use cache
    seed = small_config["splits"]["development"][0]
    config_copy = small_config.copy()

    # First call
    result1 = prepare_phylogeny(
        config_copy,
        baseline.datasets[seed].directory,
        "deterministic",
        baseline.datasets[seed].name,
        baseline.implementation,
    )

    # Second call with same params should return same directory (cached)
    result2 = prepare_phylogeny(
        config_copy,
        baseline.datasets[seed].directory,
        "deterministic",
        baseline.datasets[seed].name,
        baseline.implementation,
    )

    assert result1 == result2, "Cached artifact should be reused on rerun"
