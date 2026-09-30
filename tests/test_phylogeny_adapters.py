import numpy as np
import pandas as pd
import pytest

from epilink_evaluation.clusterers import treecluster
from epilink_evaluation.phylogeny import trees
from epilink_evaluation.phylogeny.external import executable
from epilink_evaluation.provenance import read_json


def test_dated_tree_preserves_named_nodes_dates_and_year_lengths(
    small_config, tmp_path, monkeypatch
):
    raw = tmp_path / "raw.nwk"
    raw.write_text("((a:0.001,b:0.001):0.002,c:0.003);\n")
    cases = pd.DataFrame({"case_id": ["a", "b", "c"], "sample_date": [0, 1, 2]})
    monkeypatch.setattr(
        trees,
        "command_identity",
        lambda _: {"path": "/fake/treetime", "sha256": "test"},
    )

    def export_treetime(argv, directory, name, timeout):
        output = directory / "treetime"
        output.mkdir()
        # TreeTime's actual export combines internal names and date comments.
        (output / "timetree.nexus").write_text(
            "#NEXUS\nBegin Taxa;\n Dimensions NTax=3;\n TaxLabels a b c;\nEnd;\n"
            "Begin Trees;\n Tree tree1=((a:0.01[&date=2020.00],"
            "b:0.02[&date=2020.01])NODE_0000001:0.03[&date=2019.99],"
            "c:0.04[&date=2020.02])NODE_0000000:0.0[&date=2019.98];\nEnd;\n"
        )

    monkeypatch.setattr(trees, "run_command", export_treetime)
    path = trees.dated_tree(small_config, raw, cases, "deterministic", "dataset", {})
    tree = trees.validate_tree(path, cases.case_id)
    assert tree.distance("a", "b") == pytest.approx(0.03)
    assert tree.distance("a", "c") == pytest.approx(0.08)
    internal = tree.find_any(name="NODE_0000001")
    assert internal is not None
    assert internal.confidence is None
    assert internal.comment == "&date=2019.99"
    manifest = read_json(path.parent / "manifest.json")
    assert manifest["units"] == "calendar_years"
    assert not manifest["root_changed"]
    assert manifest["signature"]["rng_seed"] == small_config["treecluster"]["rng_seed"]
    argv = manifest["command"]
    assert argv[argv.index("--rng-seed") + 1] == str(
        small_config["treecluster"]["rng_seed"]
    )
    dates = pd.read_csv(path.parent / "dates.csv")
    assert dates.date.tolist() == ["2020-01-01", "2020-01-02", "2020-01-03"]


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
