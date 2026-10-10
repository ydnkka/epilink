"""IQ-TREE discovery and empirical alignment contracts."""

import pandas as pd
import pytest

from epilink_evaluation.phylogeny import external
from epilink_evaluation.phylogeny.boston import alignment_length


def test_iqtree_generic_name_discovers_versioned_executable_and_respects_explicit_path(
    tmp_path, monkeypatch
):
    versioned, custom = tmp_path / "iqtree2", tmp_path / "custom-iqtree"
    versioned.write_text("versioned executable")
    custom.write_text("custom executable")
    calls = []

    def which(name):
        calls.append(name)
        return {"iqtree2": str(versioned), str(custom): str(custom)}.get(name)

    monkeypatch.setattr(external.shutil, "which", which)
    monkeypatch.setattr(external.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(external.os, "access", lambda *args: False)
    assert external.command_identity("iqtree")["path"] == str(versioned)
    assert calls == ["iqtree3", "iqtree2"]
    calls.clear()
    assert external.command_identity(custom)["path"] == str(custom)
    assert calls == [str(custom)]


@pytest.mark.parametrize(
    "sequences",
    [
        ">A\nACGT\n>A\nACGT\n",
        ">A\nACGT\n>C\nACGT\n",
        ">A\nACGT\n>B\nACG\n",
        "",
    ],
)
def test_boston_alignment_rejects_duplicate_missing_extra_or_unequal_sequences(
    tmp_path, sequences
):
    path = tmp_path / "alignment.fasta"
    path.write_text(sequences)
    with pytest.raises(ValueError, match="Boston alignment"):
        alignment_length(path, pd.DataFrame({"case_id": ["A", "B"]}))


def test_boston_alignment_order_does_not_change_sample_universe(tmp_path):
    path = tmp_path / "alignment.fasta"
    path.write_text(">B\nACGT\n>A\nACGT\n")
    assert alignment_length(path, pd.DataFrame({"case_id": ["A", "B"]})) == 4
