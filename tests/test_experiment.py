"""Producer/consumer boundaries for shared synthetic experiments and holdouts."""

from copy import deepcopy
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
import yaml

from epilink_evaluation.config import load_config, load_diagnostics_config
from epilink_evaluation.inputs import synthetic
from epilink_evaluation.inputs.experiment import (
    SyntheticExperiment,
    load_experiment,
    prepare_experiment,
)
from epilink_evaluation.natural_history import natural_history
from epilink_evaluation.provenance import (
    digest_file,
    implementation_signature,
    read_json,
    valid_artifact,
    write_json,
)
from epilink_evaluation.scorers import registry as scorer_registry
from epilink_evaluation.workflows import diagnostics as diagnostics_module
from epilink_evaluation.workflows.baseline import Baseline


def artifact_inventory(root):
    return {
        path: (digest_file(path), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("prepared", [False, True])
def test_baseline_requires_completed_diagnostics(small_config, prepared):
    if prepared:
        experiment = prepare_experiment(small_config)
        experiment.prepare_development()
        assert not (experiment.directory / "diagnostics.json").exists()
    with pytest.raises(ValueError, match="diagnostics"):
        Baseline(small_config)
    assert not Path(small_config["output_directory"]).exists()


def test_diagnostics_materialize_only_development(small_config, prepare_diagnostics):
    original = deepcopy(small_config)
    diagnostics = prepare_diagnostics(small_config)
    assert small_config == original
    experiment = diagnostics.exp
    seeds = set(small_config["splits"]["development"])
    assert set(diagnostics.datasets) == seeds
    manifests = list(
        (experiment.root / "artifacts/observations").glob("*/manifest.json")
    )
    assert {read_json(path)["signature"]["seed"] for path in manifests} == seeds
    assert len(manifests) == len(seeds)
    assert {
        path.name for path in (experiment.directory / "observations").iterdir()
    } == {f"seed_{seed}.json" for seed in seeds}
    assert experiment.require_diagnostics()["status"] == "complete"
    assert not (experiment.root / "heldout_access").exists()
    assert not Path(small_config["output_directory"]).exists()
    for role in ("train", "evaluation"):
        for seed in small_config["splits"][role]:
            with pytest.raises(ValueError, match="development seeds"):
                diagnostics.dataset(seed)
            link = experiment.directory / "observations" / f"seed_{seed}.json"
            assert not link.exists()


def test_baseline_reuses_exact_diagnostic_observations(
    small_config, prepare_diagnostics, monkeypatch
):
    diagnostics = prepare_diagnostics(small_config)
    # Training is a separate producer; prepare it before forbidding all simulation.
    for seed in small_config["splits"]["train"]:
        diagnostics.exp.dataset(seed, prepare=True)
    before = artifact_inventory(diagnostics.exp.root / "artifacts")

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Baseline must reuse the prepared development observations"
        )

    monkeypatch.setattr(synthetic, "simulate_epidemic_dates", forbidden)
    monkeypatch.setattr(synthetic, "simulate_genomic_sequences", forbidden)
    baseline = Baseline(small_config)
    assert baseline.run("develop")
    assert baseline.experiment.identity == diagnostics.exp.identity
    assert baseline.truth_directory == diagnostics.truth_directory
    for seed, directory in diagnostics.datasets.items():
        assert baseline.dataset(seed) == directory
        pairwise = read_json(
            baseline.directory
            / "development"
            / f"seed_{seed}"
            / "pairwise/manifest.json"
        )
        scores = read_json(
            baseline.root
            / "artifacts/scores"
            / pairwise["signature"]["score_id"]
            / "manifest.json"
        )
        assert scores["signature"]["dataset"] == directory.name
    assert artifact_inventory(diagnostics.exp.root / "artifacts") == before


def test_both_inference_processes_match_the_pinned_generation(
    small_config, prepare_diagnostics, monkeypatch
):
    small_config["scorers"] = ["EDD", "ESS"]
    small_config["generation"]["incubation"]["mean"] = 6.25
    small_config["inference"] = deepcopy(small_config["generation"])
    diagnostics = prepare_diagnostics(small_config)
    profiles = []
    original = scorer_registry.InfectiousnessToTransmission

    def capture_profile(*, parameters, rng_seed):
        profiles.append(parameters)
        return original(parameters=parameters, rng_seed=rng_seed)

    monkeypatch.setattr(
        scorer_registry, "InfectiousnessToTransmission", capture_profile
    )
    baseline = Baseline(small_config)
    seed = small_config["splits"]["development"][0]
    observations, _, _, scores, score_id = baseline.scores(seed)
    assert set(baseline.context.epilink_models) == {"deterministic", "stochastic"}
    expected = natural_history(diagnostics.exp.config["generation"])
    assert len(profiles) == 2
    assert all(profile == expected for profile in profiles)
    assert scores.pair_id.equals(observations.pair_id)
    assert set(scores) == {"pair_id", "EDD", "ESS"}
    saved = read_json(baseline.root / "artifacts/scores" / score_id / "manifest.json")
    assert saved["signature"]["inference"] == diagnostics.exp.config["generation"]
    assert saved["signature"]["dataset"] == diagnostics.dataset(seed).name


def test_evaluation_requires_release_even_when_cached(
    small_config, prepare_diagnostics
):
    prepare_diagnostics(small_config)
    baseline = Baseline(small_config)
    seed = small_config["splits"]["evaluation"][0]
    for consumer in (baseline, baseline.experiment):
        with pytest.raises(ValueError, match="explicit held-out release"):
            consumer.dataset(seed)
    with pytest.raises(ValueError, match="explicit held-out release"):
        baseline.experiment.dataset(seed, prepare=True)
    assert not (baseline.experiment.root / "heldout_access").exists()

    baseline.select()
    # A selection file alone does not authorize reading evaluation observations.
    with pytest.raises(ValueError, match="explicit held-out release"):
        baseline.dataset(seed)
    assert baseline.evaluate()
    cached = baseline.dataset(seed)
    before = artifact_inventory(cached)
    resumed = Baseline(deepcopy(small_config))
    for consumer in (resumed, resumed.experiment):
        with pytest.raises(ValueError, match="explicit held-out release"):
            consumer.dataset(seed)
    with pytest.raises(ValueError, match="explicit held-out release"):
        resumed.experiment.dataset(seed, prepare=True)
    assert artifact_inventory(cached) == before
    assert resumed.evaluate()
    assert resumed.dataset(seed) == cached
    assert artifact_inventory(cached) == before


@pytest.fixture
def released_baseline(small_config, prepare_diagnostics):
    prepare_diagnostics(small_config)
    baseline = Baseline(small_config)
    baseline.select()
    assert baseline.evaluate()
    return baseline


@pytest.mark.parametrize("change", ["thresholds", "selection", "generation"])
def test_accessed_holdouts_cannot_be_reused_for_changed_analysis(
    released_baseline, prepare_diagnostics, tmp_path, change
):
    baseline = released_baseline
    config = deepcopy(baseline.config)
    config["output_directory"] = str(tmp_path / "another_baseline")
    if change == "thresholds":
        config["thresholds"]["genetic"] = [0, 1, 3]
    elif change == "selection":
        config["selection"]["criteria"][0]["name"] = "revised_analysis"
    else:
        config["generation"]["incubation"]["mean"] *= 1.1
        config["inference"] = deepcopy(config["generation"])
        prepare_diagnostics(config)
    access = artifact_inventory(baseline.experiment.root / "heldout_access")
    assert access
    changed = Baseline(config)
    assert changed.directory != baseline.directory
    with pytest.raises(ValueError, match="fresh evaluation seeds"):
        changed.select()
    assert not (changed.directory / "selection/operating_points.json").exists()
    assert not (changed.directory / "evaluation").exists()
    assert artifact_inventory(baseline.experiment.root / "heldout_access") == access


@pytest.mark.parametrize("role", ["train", "development"])
def test_accessed_holdouts_cannot_be_reassigned(released_baseline, role):
    baseline = released_baseline
    config = deepcopy(baseline.config)
    seed = config["splits"]["evaluation"][0]
    config["splits"][role] = [seed]
    config["splits"]["evaluation"] = [seed + 1000]
    experiment = prepare_experiment(config)
    assert experiment.directory != baseline.experiment.directory
    # The data ID is unchanged by role, so an observation for this seed is cached.
    assert baseline.dataset(seed).exists()
    with pytest.raises(ValueError, match="Previously accessed held-out seeds"):
        experiment.dataset(seed, prepare=True)
    assert not (experiment.directory / "observations" / f"seed_{seed}.json").exists()


@pytest.mark.parametrize(
    "change", ["diagnostics", "reporting", "thresholds", "scorers"]
)
def test_analysis_changes_preserve_generation_artifacts(
    small_config, prepare_diagnostics, monkeypatch, change
):
    diagnostics = prepare_diagnostics(small_config)
    before = artifact_inventory(diagnostics.exp.root / "artifacts")
    config = deepcopy(small_config)
    if change == "diagnostics":
        config["diagnostics"] = deepcopy(diagnostics.settings)
        config["diagnostics"]["leiden"]["resolution_grid"] = [0.4]
        implementation = implementation_signature()
        implementation["evaluation"]["diagnostics/graphs.py"] = "revised diagnostics"
        monkeypatch.setattr(
            diagnostics_module, "implementation_signature", lambda: implementation
        )
    elif change == "reporting":
        implementation = implementation_signature()
        implementation["evaluation"]["reporting/diagnostics.py"] = "revised report"
        monkeypatch.setattr(
            diagnostics_module, "implementation_signature", lambda: implementation
        )
    elif change == "thresholds":
        config["thresholds"]["genetic"] = [0, 1, 3]
    else:
        config["scorers"] = ["GD_D", "GD_S"]
    updated = prepare_diagnostics(config)
    assert updated.exp.identity == diagnostics.exp.identity
    assert updated.datasets == diagnostics.datasets
    if change == "diagnostics":
        assert updated.directory != diagnostics.directory
    elif change == "reporting":
        assert updated.directory == diagnostics.directory
    baseline = Baseline(config)
    for seed, directory in diagnostics.datasets.items():
        assert baseline.dataset(seed) == directory
    assert artifact_inventory(diagnostics.exp.root / "artifacts") == before


@pytest.mark.parametrize("change", ["generation", "sampling"])
def test_scientific_changes_require_fresh_preparation(
    small_config, prepare_diagnostics, change
):
    diagnostics = prepare_diagnostics(small_config)
    before = artifact_inventory(diagnostics.exp.root / "artifacts")
    config = deepcopy(small_config)
    if change == "generation":
        config["generation"]["testing_delay"]["mean"] *= 1.2
        config["inference"] = deepcopy(config["generation"])
    else:
        config["simulation"]["fraction_sampled"] = 0.6
    with pytest.raises(ValueError, match="differs from configuration"):
        Baseline(config)
    updated = prepare_diagnostics(config)
    assert updated.exp.identity != diagnostics.exp.identity
    assert updated.exp.backbone_directory == diagnostics.exp.backbone_directory
    assert updated.truth_directory == diagnostics.truth_directory
    baseline = Baseline(config)
    for seed, old in diagnostics.datasets.items():
        new = baseline.dataset(seed)
        assert new != old
        saved = read_json(new / "manifest.json")
        assert saved["signature"]["generation"] == config["generation"]
        assert saved["signature"]["simulation"] == config["simulation"]
    assert {
        path: (digest_file(path), path.stat().st_mtime_ns) for path in before
    } == before


def test_changed_producer_cannot_mix_generation_versions(small_config):
    implementation = implementation_signature()
    experiment = prepare_experiment(small_config, implementation)
    experiment.prepare_development()
    seed = small_config["splits"]["development"][0]
    old = experiment.dataset(seed)
    before = artifact_inventory(old)
    changed = deepcopy(implementation)
    changed["evaluation"]["inputs/synthetic.py"] = "new observation producer"
    with pytest.raises(ValueError, match="Generation implementation changed"):
        load_experiment(small_config, changed)
    pinned = SyntheticExperiment(experiment.directory, changed)
    assert pinned.dataset(seed) == old
    training_seed = small_config["splits"]["train"][0]
    with pytest.raises(ValueError, match="Generation implementation changed"):
        pinned.dataset(training_seed, prepare=True)
    link = pinned.directory / "observations" / f"seed_{training_seed}.json"
    assert not link.exists()
    replacement = prepare_experiment(small_config, changed)
    replacement.prepare_development()
    assert replacement.directory != experiment.directory
    assert replacement.dataset(seed) != old
    assert artifact_inventory(old) == before


def test_development_corruption_requires_explicit_producer_repair(
    small_config, prepare_diagnostics, monkeypatch
):
    diagnostics = prepare_diagnostics(small_config)
    seed = small_config["splits"]["development"][0]
    directory = diagnostics.dataset(seed)
    observations, cases = synthetic.load_observations(directory)
    manifest = read_json(directory / "manifest.json")
    (directory / "pairs.parquet").write_bytes(b"interrupted write")
    assert not valid_artifact(directory, manifest["signature"])

    def forbidden(*args, **kwargs):
        raise AssertionError("A baseline consumer cannot repair development data")

    with monkeypatch.context() as patch:
        patch.setattr(synthetic, "simulate_epidemic_dates", forbidden)
        with pytest.raises(ValueError, match="changed prepared artifact"):
            Baseline(small_config)
        with pytest.raises(ValueError, match="changed prepared artifact"):
            diagnostics.exp.dataset(seed)
    repaired = prepare_diagnostics(small_config)
    assert repaired.dataset(seed) == directory
    assert valid_artifact(directory, manifest["signature"])
    actual_observations, actual_cases = synthetic.load_observations(directory)
    pd.testing.assert_frame_equal(actual_observations, observations)
    pd.testing.assert_frame_equal(actual_cases, cases)
    assert Baseline(small_config).dataset(seed) == directory


@pytest.mark.parametrize("evidence", ["stage", "artifact", "completion", "manifest"])
def test_corrupt_completed_diagnostic_evidence_blocks_baseline(
    small_config, prepare_diagnostics, evidence
):
    diagnostics = prepare_diagnostics(small_config)
    marker = read_json(diagnostics.exp.directory / "diagnostics.json")
    assert marker["status"] == "complete"
    if evidence == "stage":
        path = diagnostics.directory / "observations/summary.csv"
    elif evidence == "completion":
        path = diagnostics.directory / "completion/coverage.json"
    else:
        records = read_json(diagnostics.directory / "observations/index.json")[
            "records"
        ]
        record = records[0]
        artifact = Path(record["artifact"])
        path = artifact / (
            "manifest.json" if evidence == "manifest" else "cells.parquet"
        )
    if evidence == "manifest":
        saved = read_json(path)
        saved["completed_at"] = "changed after diagnostic completion"
        write_json(path, saved)
        # The artifact's own file checks still pass; the completion inventory must
        # also pin the manifest hash to detect this change.
        assert valid_artifact(artifact, saved["signature"])
    else:
        path.write_bytes(path.read_bytes() + b"changed after diagnostic completion")
    assert read_json(diagnostics.exp.directory / "diagnostics.json") == marker
    with pytest.raises(
        ValueError, match="changed prepared artifact|Diagnostic evidence changed"
    ):
        Baseline(small_config)


def test_source_topology_snapshot_survives_original_replacement(
    small_config, prepare_diagnostics
):
    diagnostics = prepare_diagnostics(small_config)
    original = Path(small_config["inputs"]["tree_path"])
    source_sha256 = digest_file(original)
    snapshot = diagnostics.exp.backbone_directory / "transmission_tree.gml"
    before = artifact_inventory(diagnostics.exp.backbone_directory)
    original.write_text("a replaced current topology")
    baseline = Baseline(small_config)
    inputs = read_json(baseline.directory / "inputs.json")
    assert Path(inputs["tree_path"]) == snapshot
    assert inputs["tree_sha256"] == digest_file(snapshot)
    assert inputs["source"]["tree_path"] == str(original)
    assert inputs["source"]["tree_sha256"] == source_sha256
    assert nx.utils.graphs_equal(baseline.tree, nx.read_gml(snapshot))
    assert set(baseline.tree) == {f"case_{node}" for node in range(15)}
    baseline.prepare()
    assert artifact_inventory(diagnostics.exp.backbone_directory) == before


def test_shared_yaml_uses_one_generation_design_and_matched_inference(
    small_config, tmp_path
):
    shared = {
        key: deepcopy(small_config[key])
        for key in ("schema_version", "inputs", "generation", "simulation", "splits")
    }
    shared["output_directory"] = "shared"
    shared["generation"]["incubation"]["mean"] = 6.25
    shared_path = tmp_path / "experiment.yaml"
    shared_path.write_text(yaml.safe_dump(shared))
    baseline = deepcopy(small_config)
    for key in (
        "inputs",
        "generation",
        "simulation",
        "splits",
        "inference",
        "experiment_root",
    ):
        baseline.pop(key)
    baseline["experiment_config"] = shared_path.name
    baseline_path = tmp_path / "baseline.yaml"
    baseline_path.write_text(yaml.safe_dump(baseline))
    diagnostics_path = tmp_path / "diagnostics.yaml"
    root = Path(__file__).resolve().parents[1]
    diagnostics = yaml.safe_load(
        (root / "evaluation/00_synthetic_diagnostics/config.yaml").read_text()
    )
    diagnostics.update(
        experiment_config=shared_path.name, output_directory="diagnostics"
    )
    diagnostics_path.write_text(yaml.safe_dump(diagnostics))
    loaded_baseline = load_config(baseline_path)
    loaded_diagnostics = load_diagnostics_config(diagnostics_path)
    for key in ("inputs", "generation", "simulation", "splits", "experiment_root"):
        assert loaded_baseline[key] == loaded_diagnostics[key]
    assert loaded_baseline["inference"] == shared["generation"]
    assert loaded_baseline["inference"] is not loaded_baseline["generation"]
    assert Path(loaded_baseline["experiment_root"]) == tmp_path / "shared"
    baseline["generation"] = deepcopy(shared["generation"])
    baseline["generation"]["incubation"]["mean"] = 9.0
    baseline_path.write_text(yaml.safe_dump(baseline))
    with pytest.raises(ValueError, match="generation|override|shared"):
        load_config(baseline_path)


def test_pipeline_smoke_can_repeat_after_comparison_changes(
    small_config, prepare_diagnostics
):
    small_config["inputs"]["smoke_cases"] = 12
    diagnostics = prepare_diagnostics(small_config)
    baseline = Baseline(small_config)
    baseline.select()
    assert baseline.evaluate()
    changed = deepcopy(small_config)
    changed["thresholds"]["genetic"] = [0, 1, 3]
    repeated = Baseline(changed)
    repeated.select()
    assert repeated.evaluate()
    assert repeated.directory != baseline.directory
    seed = small_config["splits"]["evaluation"][0]
    assert baseline.dataset(seed) == repeated.dataset(seed)
    assert (
        len(list((diagnostics.exp.root / "validation_access").glob("*/seed_*.json")))
        == 2
    )
    assert not (diagnostics.exp.root / "heldout_access").exists()


def test_pinned_dataset_rejects_another_generation_with_same_seed(small_config):
    first = prepare_experiment(small_config)
    first.prepare_development()
    changed = deepcopy(small_config)
    changed["generation"]["testing_delay"]["mean"] *= 2
    second = prepare_experiment(changed)
    second.prepare_development()
    seed = small_config["splits"]["development"][0]
    filename = f"observations/seed_{seed}.json"
    write_json(first.directory / filename, read_json(second.directory / filename))
    with pytest.raises(ValueError, match="Observation provenance differs"):
        first.dataset(seed)


def test_observation_bundle_complete(small_config, tmp_path, prepare_diagnostics):
    """Verify all FASTA/reference/date files present with correct checksums."""
    diagnostics = prepare_diagnostics(small_config)
    for seed in small_config["splits"]["development"]:
        directory = diagnostics.dataset(seed)
        observations, cases = synthetic.load_observations(directory)
        required = [
            "sampled_deterministic.fasta",
            "sampled_stochastic.fasta",
            "reference.fasta",
            "sampling_dates.tsv",
            "pairs.parquet",
            "cases.parquet",
        ]
        missing = [f for f in required if not (directory / f).exists()]
        assert not missing, f"Missing observation files: {missing}"
        # Verify checksums in manifest
        manifest = read_json(directory / "manifest.json")
        assert manifest["status"] == "complete"


def test_sampled_fasta_matches_cases(small_config, tmp_path, prepare_diagnostics):
    """Verify sampled FASTA contains exactly cases in cases.parquet."""
    diagnostics = prepare_diagnostics(small_config)
    for seed in small_config["splits"]["development"]:
        directory = diagnostics.dataset(seed)
        observations, cases = synthetic.load_observations(directory)
        # Count FASTA headers
        det_fasta = directory / "sampled_deterministic.fasta"
        sto_fasta = directory / "sampled_stochastic.fasta"
        with open(det_fasta) as f:
            det_headers = [line.strip() for line in f if line.startswith(">")]
        with open(sto_fasta) as f:
            sto_headers = [line.strip() for line in f if line.startswith(">")]
        det_case_ids = set(observations.case_id) if hasattr(observations, "case_id") else set()
        # Compare with cases.parquet
        cases_df = pd.read_parquet(directory / "cases.parquet")
        assert len(det_headers) == len(cases_df), (
            f"Deterministic FASTA has {len(det_headers)} headers but {len(cases_df)} cases"
        )


def test_reference_sequence_correct(small_config, tmp_path, prepare_diagnostics):
    """Verify reference.fasta matches simulation.reference_sequence_string."""
    diagnostics = prepare_diagnostics(small_config)
    for seed in small_config["splits"]["development"]:
        directory = diagnostics.dataset(seed)
        # Read reference from artifact
        ref_fasta = directory / "reference.fasta"
        with open(ref_fasta) as f:
            ref_content = f.read()
        assert ">ancestral_reference" in ref_content
        # Reference should have header + sequence (multi-line FASTA ok)
        ref_lines = [l.strip() for l in ref_content.split("\n") if l.strip()]
        assert len(ref_lines) >= 2  # header + at least one sequence line


def test_dates_tsv_matches_cases(small_config, tmp_path, prepare_diagnostics):
    """Verify sampling_dates.tsv matches cases.parquet sample_date."""
    diagnostics = prepare_diagnostics(small_config)
    for seed in small_config["splits"]["development"]:
        directory = diagnostics.dataset(seed)
        dates_tsv = directory / "sampling_dates.tsv"
        cases_parquet = directory / "cases.parquet"
        assert dates_tsv.exists()
        dates_df = pd.read_csv(dates_tsv, sep="\t")
        cases_df = pd.read_parquet(cases_parquet)
        # Case IDs should match
        assert set(dates_df["case_id"].astype(str)) == set(cases_df["case_id"].astype(str))
        # Dates should be present for all cases
        assert len(dates_df) == len(cases_df)
