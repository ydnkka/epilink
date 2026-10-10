from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import epilink
import numpy as np
import pandas as pd
import pytest
from Bio.Phylo._io import read as read_phylo

from epilink_evaluation.clusterers import treecluster
from epilink_evaluation.phylogeny import trees
from epilink_evaluation.phylogeny.external import executable
from epilink_evaluation.provenance import implementation_signature, read_json
from epilink_evaluation.workflows import baseline as baseline_module
from epilink_evaluation.workflows.settings import settings_registry


def test_treecluster_preserves_each_unclustered_case(small_config, tmp_path):
    try:
        executable(small_config["treecluster"]["executable"])
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


@pytest.fixture
def observation_bundle(tmp_path):
    directory = tmp_path / "observations"
    directory.mkdir()
    cases = pd.DataFrame(
        {
            "case_id": ["case_0", "case_1", "case_2"],
            "node_index": [0, 1, 2],
            "sample_date": [1.0, 2.0, 3.0],
            "exposure_date": [0.5, 1.5, 2.5],
        }
    )
    cases.to_parquet(directory / "cases.parquet", index=False)
    cases[["case_id", "sample_date"]].to_csv(
        directory / "sampling_dates.tsv", sep="\t", index=False
    )
    (directory / "sampled_deterministic.fasta").write_text(
        ">case_0\nACCGT\n>case_1\nGGTAA\n>case_2\nTTCGG\n"
    )
    (directory / "sampled_stochastic.fasta").write_text(
        ">case_0\nCCGGT\n>case_1\nGGAAT\n>case_2\nTCCTT\n"
    )
    (directory / "reference.fasta").write_text(">ancestral_reference\nACGTA\n")
    return directory


@pytest.fixture
def mocked_iqtree(monkeypatch):
    calls = []

    def command_identity(command):
        return {
            "path": f"/mock/tools/{Path(command).name}",
            "sha256": "mock-executable",
        }

    def build(**kwargs):
        calls.append(kwargs)
        cases = pd.read_parquet(
            Path(kwargs["alignment_fasta"]).parent / "cases.parquet"
        ).case_id.tolist()
        directory = Path(kwargs["output_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        raw, dated = directory / "raw.nwk", directory / "dated.nwk"
        raw.write_text(
            "("
            + ",".join(f"{case}:0.001" for case in [*cases, "ancestral_reference"])
            + ");\n"
        )
        dated.write_text("(" + ",".join(f"{case}:1" for case in cases) + ");\n")
        return SimpleNamespace(
            output_paths={"raw_tree": raw, "dated_tree": dated},
            reference_name="ancestral_reference",
            node_dates=pd.DataFrame({"case_id": cases, "date": 1.0}),
            clock_rate=kwargs["clock_rate"],
            date_origin="2020-01-01",
        )

    monkeypatch.setattr(trees, "command_identity", command_identity)
    monkeypatch.setattr(baseline_module, "command_identity", command_identity)
    monkeypatch.setattr(epilink, "build_phylogenetic_tree_from_fasta", build)
    return calls


@pytest.mark.parametrize("process", ["deterministic", "stochastic"])
def test_iqtree_adapter_mocked(
    small_config, observation_bundle, mocked_iqtree, process
):
    result = trees.prepare_phylogeny(
        small_config,
        observation_bundle,
        process,
        "test-dataset",
        implementation_signature(),
    )
    (call,) = mocked_iqtree
    phylogeny = small_config["phylogeny"]
    assert call == {
        "alignment_fasta": str(observation_bundle / f"sampled_{process}.fasta"),
        "reference_fasta": str(observation_bundle / "reference.fasta"),
        "dates": str(observation_bundle / "sampling_dates.tsv"),
        "dated": True,
        "output_dir": str(result / "backend"),
        "model": phylogeny["model"],
        "threads": phylogeny["threads"],
        "seed": phylogeny["seed"],
        "clock_rate": phylogeny["clock_rate"],
        "iqtree_executable": read_json(result / "manifest.json")["signature"][
            "iqtree_executable"
        ]["path"],
        "timeout": phylogeny["timeout"],
    }
    cases = pd.read_parquet(observation_bundle / "cases.parquet").case_id
    trees.validate_tree(result / "raw.nwk", cases)
    trees.validate_tree(result / "dated.nwk", cases)
    assert read_json(result / "manifest.json")["status"] == "complete"


def test_reference_pruned_from_raw_tree():

    original = read_phylo(StringIO("((a:0.1,b:0.1):0.2,c:0.3);"), "newick")
    pruned = trees._prune_reference_tip(original, "a")
    assert {tip.name for tip in pruned.get_terminals()} == {"b", "c"}
    assert {tip.name for tip in original.get_terminals()} == {"a", "b", "c"}
    assert pruned.distance("b", "c") == pytest.approx(original.distance("b", "c"))


def test_day_thresholds_no_conversion(small_config):
    small_config["simulation"]["alignment_length"] = 5000
    small_config["treecluster"].update(
        enabled=True,
        methods=["max_clade"],
        genetic_threshold_snps=[0, 1, 2],
        temporal_threshold_days=[0, 365, 730],
    )
    definitions = settings_registry(small_config).values()
    dated = [
        d
        for d in definitions
        if d["kind"] == "treecluster" and d["tree_kind"] == "dated"
    ]
    raw = [
        d for d in definitions if d["kind"] == "treecluster" and d["tree_kind"] == "raw"
    ]
    assert {d["threshold_units"] for d in dated} == {"days"}
    assert {d["threshold"] for d in dated} == {0, 365, 730}
    assert {d["threshold_units"] for d in raw} == {"snps"}
    assert {d["threshold"] for d in raw} == {0, 1 / 5000, 2 / 5000}


def test_cache_reuse_on_rerun(small_config, observation_bundle, mocked_iqtree):
    implementation = implementation_signature()
    result = trees.prepare_phylogeny(
        small_config,
        observation_bundle,
        "deterministic",
        "test-dataset",
        implementation,
    )
    saved = {p: p.stat().st_mtime_ns for p in result.iterdir() if p.is_file()}
    resumed = trees.prepare_phylogeny(
        small_config,
        observation_bundle,
        "deterministic",
        "test-dataset",
        implementation,
    )
    assert resumed == result
    assert len(mocked_iqtree) == 1
    assert saved == {p: p.stat().st_mtime_ns for p in saved}


@pytest.mark.parametrize(
    "invalid_tree, message",
    [
        ("(case_0:1,case_1:1,missing:1);", "case universe"),
        ("(case_0:1,case_1:-1,case_2:1);", "negative branch"),
    ],
)
def test_invalid_dated_tree_is_not_completed(
    small_config, observation_bundle, mocked_iqtree, monkeypatch, invalid_tree, message
):
    original = epilink.build_phylogenetic_tree_from_fasta

    def build(**kwargs):
        result = original(**kwargs)
        result.output_paths["dated_tree"].write_text(invalid_tree)
        return result

    monkeypatch.setattr(epilink, "build_phylogenetic_tree_from_fasta", build)
    with pytest.raises(ValueError, match=message):
        trees.prepare_phylogeny(
            small_config,
            observation_bundle,
            "deterministic",
            "test-dataset",
            implementation_signature(),
        )
    backend = Path(mocked_iqtree[0]["output_dir"])
    assert not (backend.parent / "manifest.json").exists()


def test_baseline_treecluster_uses_observation_paths(
    small_config, prepare_diagnostics, mocked_iqtree, monkeypatch
):
    small_config["treecluster"].update(
        enabled=True,
        methods=["max_clade"],
        genetic_threshold_snps=[1],
        temporal_threshold_days=[7],
    )

    def cluster(path, cases, *args):
        trees.validate_tree(path, cases.case_id)
        return np.zeros(len(cases), dtype=int), {}

    monkeypatch.setattr(baseline_module, "treecluster", cluster)
    prepare_diagnostics(small_config)
    baseline = baseline_module.Baseline(small_config)
    assert baseline.run("develop")
    seed = small_config["splits"]["development"][0]
    dataset = baseline.dataset(seed)
    assert len(mocked_iqtree) == 2  # Raw/dated consumers reuse each genetic process.
    assert all(
        Path(call["alignment_fasta"]).parent == dataset for call in mocked_iqtree
    )
    status = read_json(
        baseline.directory / "development" / f"seed_{seed}" / "clusters/status.json"
    )
    assert status["status"] == "complete"
    assert not status["errors"]
