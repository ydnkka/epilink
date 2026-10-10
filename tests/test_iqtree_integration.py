"""Tests for IQ-TREE integration and observation bundle persistence."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from io import BytesIO

from epilink_evaluation.inputs.synthetic import (
    _export_reference,
    _export_sampling_dates,
    load_tree_inputs,
)
from epilink_evaluation.phylogeny.trees import _prune_reference_tip
from epilink_evaluation.workflows.settings import settings_registry
from epilink_evaluation.config import load_config


class TestObservationBundleExport:
    """Test FASTA/reference/date export functions."""
    
    def test_export_reference(self, tmp_path):
        """Test reference sequence export."""
        ref_seq = "ACGTACGTACGT"
        filepath = tmp_path / "reference.fasta"
        _export_reference(ref_seq, filepath)
        
        content = filepath.read_text()
        assert ">ancestral_reference" in content
        assert "ACGTACGTACGT" in content
    
    def test_export_sampling_dates(self, tmp_path):
        """Test sampling dates TSV export."""
        cases = pd.DataFrame({
            "case_id": ["A", "B", "C"],
            "sample_date": [1.0, 5.5, 10.0]
        })
        filepath = tmp_path / "dates.tsv"
        _export_sampling_dates(cases, filepath)
        
        loaded = pd.read_csv(filepath, sep="\t")
        assert list(loaded.columns) == ["case_id", "sample_date"]
        assert len(loaded) == 3
        assert loaded.iloc[0]["case_id"] == "A"
        assert loaded.iloc[0]["sample_date"] == 1.0
    
    def test_load_tree_inputs_missing_files(self, tmp_path):
        """Test load_tree_inputs raises on missing files."""
        # Create incomplete bundle
        (tmp_path / "pairs.parquet").touch()
        (tmp_path / "cases.parquet").touch()
        
        with pytest.raises(FileNotFoundError, match="missing files"):
            load_tree_inputs(tmp_path)


class TestThresholdScaling:
    """Test genetic threshold scaling preserves absolute SNP counts."""
    
    def test_synthetic_threshold_scaling(self, tmp_path):
        """Test synthetic thresholds use alignment_length=5000."""
        # Load real config and modify
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "evaluation/01_synthetic_baseline/config.yaml")
        config["output_directory"] = str(tmp_path / "out")
        config["simulation"]["alignment_length"] = 5000
        config["treecluster"]["genetic_threshold_snps"] = [0, 5, 10]
        config["treecluster"]["enabled"] = True
        
        settings = settings_registry(config)
        
        # Find raw tree settings
        raw_settings = [
            s for s in settings.values()
            if s.get("kind") == "treecluster" and s.get("tree_kind") == "raw"
        ]
        assert len(raw_settings) > 0
        
        # Check thresholds are scaled correctly: snp_count / 5000
        thresholds = sorted(set(s["threshold"] for s in raw_settings))
        expected = [0.0, 5.0/5000, 10.0/5000]
        assert thresholds == pytest.approx(expected)
    
    def test_boston_threshold_scaling(self, tmp_path):
        """Test Boston thresholds use alignment_length=29903."""
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "evaluation/01_synthetic_baseline/config.yaml")
        config["output_directory"] = str(tmp_path / "out")
        config["simulation"]["alignment_length"] = 29903
        config["treecluster"]["genetic_threshold_snps"] = [0, 5, 10]
        config["treecluster"]["enabled"] = True
        
        settings = settings_registry(config)
        
        raw_settings = [
            s for s in settings.values()
            if s.get("kind") == "treecluster" and s.get("tree_kind") == "raw"
        ]
        assert len(raw_settings) > 0
        
        # Check thresholds are scaled correctly: snp_count / 29903
        thresholds = sorted(set(s["threshold"] for s in raw_settings))
        expected = [0.0, 5.0/29903, 10.0/29903]
        assert thresholds == pytest.approx(expected)


class TestReferencePruning:
    """Test reference tip removal from IQ-TREE raw trees."""
    
    def test_prune_reference_tip(self, tmp_path):
        """Test reference tip is removed from tree."""
        from Bio import Phylo
        
        # Create a tree with reference tip
        tree_content = "((A:0.1,B:0.1):0.1,reference:0.2,C:0.3);"
        tree_path = tmp_path / "test.nwk"
        tree_path.write_text(tree_content)
        
        tree = Phylo.read(tree_path, "newick")
        tip_names = [tip.name for tip in tree.get_terminals()]
        assert "reference" in tip_names
        assert len(tip_names) == 4
        
        # Prune reference
        pruned = _prune_reference_tip(tree, "reference")
        pruned_names = [tip.name for tip in pruned.get_terminals()]
        
        assert "reference" not in pruned_names
        assert len(pruned_names) == 3
        assert set(pruned_names) == {"A", "B", "C"}
    
    def test_prune_preserves_sampled_tips(self, tmp_path):
        """Test pruning preserves all sampled case tips."""
        from Bio import Phylo
        
        # Create a larger tree
        tree_content = "(((A:0.1,B:0.1):0.1,C:0.2):0.1,ref:0.3,(D:0.2,E:0.2):0.1);"
        tree_path = tmp_path / "test.nwk"
        tree_path.write_text(tree_content)
        
        tree = Phylo.read(tree_path, "newick")
        original_tips = {tip.name for tip in tree.get_terminals()}
        
        pruned = _prune_reference_tip(tree, "ref")
        pruned_tips = {tip.name for tip in pruned.get_terminals()}
        
        assert pruned_tips == original_tips - {"ref"}
        assert len(pruned_tips) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
