"""Shared input locations and reuse across preparation and study entry points."""

from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
import yaml

from epilink_evaluation.cli import main, smoke_config
from epilink_evaluation.config import load_config
from epilink_evaluation.inputs import scovmod
from epilink_evaluation.provenance import digest_file, read_json, valid_artifact
from epilink_evaluation.workflows.baseline import Baseline
from epilink_evaluation.workflows.boston_config import load_study_config


@pytest.fixture
def scovmod_config(small_config, tmp_path):
    (tmp_path / "infections.csv").write_text(
        "time,location,ids\n1,1,[1]\n2,1,[2]\n1,2,[10]\n"
    )
    (tmp_path / "transmissions.csv").write_text(
        'time,location,ids\n1,1,"[2,3]"\n2,1,"[4,5,6,7]"\n1,2,"[11,12]"\n'
    )
    small_config["inputs"] = {
        "tree_path": "outputs/inputs/transmission_tree.gml",
        "infection_path": "infections.csv",
        "transmission_path": "transmissions.csv",
        "target_component_size": 7,
        "tree_seed": 42,
        "smoke_cases": None,
    }
    small_config["output_directory"] = "outputs/baseline"
    path = tmp_path / "baseline.yaml"
    path.write_text(yaml.safe_dump(small_config))
    return path


def test_default_shared_input_paths(tmp_path):
    root = Path(__file__).resolve().parents[1]
    baseline = load_config(root / "evaluation/01_synthetic_baseline/config.yaml")
    expected = (
        root / "evaluation/01_synthetic_baseline/outputs/inputs/transmission_tree.gml"
    )
    assert Path(baseline["inputs"]["tree_path"]) == expected
    assert smoke_config(baseline)["inputs"]["tree_path"] == str(expected)
    boston = load_study_config(
        config_path=root / "evaluation/03_boston_application/config.yaml",
        output=tmp_path / "custom_runs",
    )
    assert Path(boston["inputs"]["cases_path"]) == (
        root / "evaluation/03_boston_application/outputs/inputs/cases.parquet"
    )


def test_scovmod_commands_and_baseline_share_cached_tree(scovmod_config, monkeypatch):
    config = load_config(scovmod_config)
    path = Path(config["inputs"]["tree_path"])
    assert main(["scovmod", "--stage", "prepare", "--config", str(scovmod_config)]) == 0
    source = read_json(path.with_suffix(".source.json"))
    assert (source["n_cases"], source["target_size"], source["seed"]) == (7, 7, 42)
    before = path.stat().st_mtime_ns

    def forbidden(*args, **kwargs):
        raise AssertionError("Matching prepared inputs must not rebuild the tree")

    monkeypatch.setattr(scovmod, "build_tree", forbidden)
    assert main([
        "scovmod", "--stage", "prepare", "--config", str(scovmod_config), "--smoke"
    ]) == 0
    assert main(["scovmod", "--config", str(scovmod_config)]) == 0
    assert not Path(config["output_directory"]).exists()
    assert not Path(config["output_directory"] + "_smoke").exists()
    study = Baseline(smoke_config(config))
    inputs = read_json(study.directory / "inputs.json")
    assert inputs["tree_path"] == str(path)
    assert inputs["tree_source_path"] == str(path.with_suffix(".source.json"))
    assert inputs["tree_sha256"] == digest_file(path)
    assert path.stat().st_mtime_ns == before
    manifest = read_json(path.parent / "manifest.json")
    assert valid_artifact(path.parent, manifest["signature"])


@pytest.mark.parametrize("stage", ["develop", "report", "all", "trees"])
def test_scovmod_rejects_computational_stages(stage, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["scovmod", "--stage", stage, "--config", str(tmp_path / "missing.yaml")])
    assert exc.value.code == 2
    assert "SCoVMod supports --stage prepare only" in capsys.readouterr().err


def test_scovmod_cache_tracks_settings_sources_and_integrity(scovmod_config):
    config = load_config(scovmod_config)
    path = scovmod.prepare_tree(config)
    manifest_path = path.parent / "manifest.json"
    first = read_json(manifest_path)
    config["inputs"]["target_component_size"] = 3
    config["inputs"]["tree_seed"] = 43
    assert scovmod.prepare_tree(config) == path
    assert len(nx.read_gml(path)) == 3
    source = read_json(path.with_suffix(".source.json"))
    assert (source["target_size"], source["seed"]) == (3, 43)
    second = read_json(manifest_path)
    assert second["fingerprint"] != first["fingerprint"]
    transmissions = Path(config["inputs"]["transmission_path"])
    transmissions.write_text(
        transmissions.read_text().replace("[11,12]", "[11,12,13]")
    )
    scovmod.prepare_tree(config)
    assert len(nx.read_gml(path)) == 4
    third = read_json(manifest_path)
    assert third["fingerprint"] != second["fingerprint"]
    path.write_text("interrupted write")
    scovmod.prepare_tree(config)
    assert valid_artifact(path.parent, third["signature"])


def test_scovmod_honors_custom_source_path_and_prebuilt_trees(scovmod_config, tmp_path):
    config = load_config(scovmod_config)
    path = Path(config["inputs"]["tree_path"])
    source = path.parent / "origin.json"
    config["inputs"]["tree_source_path"] = str(source)
    scovmod.prepare_tree(config)
    assert source.exists()
    assert not path.with_suffix(".source.json").exists()
    assert read_json(source)["tree_sha256"] == digest_file(path)
    supplied = tmp_path / "supplied.gml"
    nx.write_gml(nx.balanced_tree(2, 2, create_using=nx.DiGraph), supplied)
    assert scovmod.prepare_tree({"inputs": {"tree_path": str(supplied)}}) == supplied


@pytest.mark.parametrize("explicit_paths", [False, True])
def test_boston_preparation_entry_points_share_inputs(tmp_path, explicit_paths):
    raw = tmp_path / "data/raw/boston"
    raw.mkdir(parents=True)
    prefix = "MGH_DPH_98percent_772samples_"
    pd.DataFrame(
        {
            "seq_id": ["A", "B"],
            "collection_date": ["2020-03-01", "2020-03-02"],
            "CONF_A_EXPOSURE": ["YES", "NO"],
            "SNF_A_EXPOSURE": ["NO", "NO"],
            "BHCHP": ["NO", "NO"],
            "CITY_A_EXPOSURE": ["NO", "NO"],
        }
    ).to_csv(raw / f"{prefix}metadata.csv", index=False)
    pd.DataFrame(
        {
            "seqName": ["A", "B"],
            "clade": ["20A", "20A"],
            "substitutions": ["C2416T", "C2416T"],
            "qc.overallStatus": ["good", "good"],
        }
    ).to_csv(raw / f"{prefix}nextclade.tsv", sep="\t", index=False)
    pd.DataFrame({"ID1": ["A"], "ID2": ["B"], "Distance": [0.0001]}).to_csv(
        raw / f"{prefix}tn93_distances.csv", index=False
    )
    inputs = {"data_root": "data"}
    if explicit_paths:
        inputs.update(
            cases_path="outputs/inputs/cases.parquet",
            pairs_path="outputs/inputs/observed_pairs.parquet",
        )
    config_path = tmp_path / "boston.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "baseline_run": "absent/current.json",
                "output_directory": "outputs/boston",
                "inputs": inputs,
            }
        )
    )
    assert main(["prepare-boston", "--config", str(config_path)]) == 0
    prepared = tmp_path / "outputs/inputs"
    manifest = read_json(prepared / "manifest.json")
    assert manifest["n_cases"] == 2
    assert manifest["n_observed_pairs"] == 1
    before = (prepared / "manifest.json").stat().st_mtime_ns
    assert main(
        [
            "boston", "--stage", "prepare", "--config", str(config_path),
            "--output", str(tmp_path / "custom_runs"),
        ]
    ) == 0
    assert (prepared / "manifest.json").stat().st_mtime_ns == before
    assert valid_artifact(prepared, manifest["signature"])
    assert not (tmp_path / "outputs/boston/inputs").exists()
    assert not (tmp_path / "custom_runs/inputs").exists()
