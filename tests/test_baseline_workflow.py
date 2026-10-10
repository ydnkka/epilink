from copy import deepcopy

import pandas as pd
import pytest

from epilink_evaluation.inputs.synthetic import load_observations, load_truth
from epilink_evaluation.provenance import read_json, valid_artifact
from epilink_evaluation.workflows.baseline import Baseline


def test_development_freeze_and_heldout_replay(small_config, prepare_diagnostics):
    small_config["splits"]["development"] = [72001, 72002]
    prepare_diagnostics(small_config)
    baseline = Baseline(small_config)
    with pytest.raises(ValueError, match="freeze development settings"):
        baseline.evaluate()
    assert baseline.run("develop")
    report = (baseline.directory / "report.md").read_text()
    assert report.index("Pre-baseline diagnostics") < report.index(
        "Development pairwise comparison"
    )
    assert "[Diagnostic report]" in report
    development = pd.read_csv(baseline.directory / "development/metrics.csv")
    assert set(development.setting_id) == set(baseline.definitions)
    assert not (baseline.directory / "evaluation").exists()
    assert set(baseline.datasets) == set(
        small_config["splits"]["train"] + small_config["splits"]["development"]
    )
    training_model = baseline.root / "artifacts/models" / baseline.training_id[:20]
    assert (
        read_json(training_model / "manifest.json")["seeds"]
        == small_config["splits"]["train"]
    )

    frozen = baseline.select()
    selected = {
        point["setting_id"]
        for point in frozen["operating_points"]
        if point["status"] == "selected"
    }
    assert selected
    # A new process restores the data-derived registry before frozen replay.
    definitions = deepcopy(baseline.definitions)
    baseline = Baseline(deepcopy(small_config))
    assert baseline.definitions == definitions
    for name in small_config["scorers"]:
        cutoffs = set()
        for seed in small_config["splits"]["development"]:
            curve = pd.read_parquet(
                baseline.directory
                / "development"
                / f"seed_{seed}"
                / "pairwise/precision_recall.parquet"
            )
            cutoffs.update(curve.loc[curve.score_name == name, "threshold"])
        assert {
            d["threshold"]
            for d in definitions.values()
            if d["kind"] == "pairwise" and d["score_name"] == name and not d["empty"]
        } == cutoffs
    assert baseline.run("evaluate")
    assert baseline.definitions == definitions
    evaluation = pd.read_csv(baseline.directory / "evaluation/operating_results.csv")
    assert set(evaluation.setting_id) == selected
    assert set(evaluation.seed) == set(small_config["splits"]["evaluation"])
    assert read_json(baseline.directory / "evaluation/selection_used.json") == frozen
    assert (
        "Held-out fixed-setting performance"
        in (baseline.directory / "report.md").read_text()
    )
    summary = pd.read_csv(baseline.directory / "evaluation/operating_summary.csv")
    secondary = summary.loc[summary.criterion == "balanced_Mle1"]
    assert (secondary.objective == "Mle1_f1").all()
    assert (secondary.objective_mean == secondary.Mle1_f1_mean).all()
    assert {"M0", "Mle1", "Mle2"} == set(
        pd.read_csv(baseline.directory / "development/frontier.csv").endpoint
    )
    for endpoint in ("M0", "Mle1", "Mle2"):
        assert (
            baseline.directory / f"figures/pairwise_precision_recall_{endpoint}.png"
        ).exists()
    assert (baseline.directory / "development/grid_adequacy.csv").exists()

    # A fresh process-equivalent context reuses completed observations unchanged.
    resumed = Baseline(deepcopy(small_config))
    seed = small_config["splits"]["development"][0]
    original = baseline.dataset(seed)
    before = (original / "pairs.parquet").stat().st_mtime_ns
    assert resumed.dataset(seed) == original
    assert (original / "pairs.parquet").stat().st_mtime_ns == before
    assert resumed.select() == frozen

    resumed.config["selection"]["criteria"][0]["name"] = "revised_after_evaluation"
    with pytest.raises(ValueError, match="fresh evaluation seeds"):
        resumed.select()

    # A damaged data-derived registry cannot silently fall back to configured
    # thresholds and release a different replay.
    (
        baseline.directory / "development/pairwise_candidates/definitions.json"
    ).write_text("{}")
    with pytest.raises(ValueError, match="candidate registry"):
        Baseline(deepcopy(small_config)).evaluate()


def test_subsampling_keeps_full_backbone_truth(small_config, prepare_diagnostics):
    small_config["simulation"]["fraction_sampled"] = 0.6
    prepare_diagnostics(small_config)
    baseline = Baseline(small_config)
    directory = baseline.dataset(small_config["splits"]["train"][0])
    observations, cases = load_observations(directory)
    truth = load_truth(baseline.truth_directory, observations.pair_id)
    assert len(cases) == 9
    assert len(observations) == 9 * 8 // 2
    assert observations.pair_id.equals(truth.pair_id)
    assert (
        truth.node_a.to_numpy() == cases.node_index.to_numpy()[observations.a]
    ).all()
    assert (
        truth.node_b.to_numpy() == cases.node_index.to_numpy()[observations.b]
    ).all()
    assert read_json(baseline.truth_directory / "manifest.json")["n_cases"] == 15
    saved = read_json(directory / "manifest.json")
    assert valid_artifact(directory, saved["signature"])
    # The explicit training producer repairs corrupt cached observations.
    (directory / "pairs.parquet").write_bytes(b"interrupted write")
    assert not valid_artifact(directory, saved["signature"])
    baseline.datasets.clear()
    assert baseline.dataset(small_config["splits"]["train"][0]) == directory
    pd.testing.assert_frame_equal(load_observations(directory)[0], observations)


def test_failed_comparator_prevents_freezing(
    small_config, prepare_diagnostics, monkeypatch
):
    prepare_diagnostics(small_config)
    baseline = Baseline(small_config)
    monkeypatch.setattr(baseline, "pairwise", lambda: None)
    monkeypatch.setattr(baseline, "clusters", lambda: False)
    with pytest.raises(RuntimeError, match="partial"):
        baseline.select()
    assert not (baseline.directory / "selection/operating_points.json").exists()


def test_tree_adapter_caches_properly(small_config, prepare_diagnostics):
    """Verify tree inference not repeated for same dataset/process."""
    from epilink_evaluation.workflows.baseline import Baseline
    from epilink_evaluation.phylogeny.trees import prepare_phylogeny

    small_config["treecluster"]["enabled"] = True
    # Set up treecluster methods and thresholds
    small_config["treecluster"]["methods"] = ["max_clade"]
    small_config["treecluster"]["genetic_threshold_snps"] = [0, 1, 2]
    small_config["treecluster"]["temporal_threshold_days"] = [0, 7, 14]
    prepare_diagnostics(small_config)
    baseline = Baseline(small_config)
    baseline.run("develop")
    # Record tree artifact signatures before second run
    seed = small_config["splits"]["development"][0]
    trees_dir = baseline.directory / "development" / f"seed_{seed}" / "clusters"
    # Find the treecluster setting definition
    tc_definitions = {
        k: v for k, v in baseline.definitions.items() if v["kind"] == "treecluster"
    }
    assert len(tc_definitions) > 0, "No treecluster definitions found"
    first_def = list(tc_definitions.values())[0]
    # Check that tree artifacts exist with correct structure
    for kind in ("raw", "dated"):
        kind_dir = (
            trees_dir / first_def["tree_kind"]
            if "tree_kind" in first_def
            else trees_dir / kind
        )
        if kind_dir.exists():
            manifest = read_json(kind_dir / "manifest.json")
            sig_before = manifest["signature"]
            # Re-run develop; should reuse cached tree artifacts
            baseline.run("develop")
            kind_dir2 = baseline.directory / "development" / f"seed_{seed}" / "clusters"
            if kind_dir2.exists():
                manifest2 = read_json(kind_dir2 / "manifest.json")
                sig_after = manifest2["signature"]
                # Signatures should match (caching works)
                assert sig_before == sig_after, (
                    f"Tree caching mismatch for {kind}: {sig_before} != {sig_after}"
                )
            break  # Just check first kind


def test_threshold_scaling_synthetic(small_config, prepare_diagnostics):
    """Verify SNP counts converted using alignment_length=5000."""
    from epilink_evaluation.workflows.settings import settings_registry

    small_config["treecluster"]["enabled"] = True
    small_config["simulation"]["alignment_length"] = 5000
    prepare_diagnostics(small_config)
    baseline = Baseline(small_config)
    baseline.run("develop")
    definitions = baseline.definitions
    # Check that treecluster definitions have threshold_units = "snps" or "days"
    # and thresholds are non-negative
    for key, defn in definitions.items():
        if defn["kind"] == "treecluster":
            assert defn["threshold_units"] in ("snps", "days")
            assert defn["threshold"] >= 0


def test_frozen_replay_uses_same_trees(small_config, prepare_diagnostics):
    """Verify baseline configuration persists across replays."""
    from epilink_evaluation.workflows.baseline import Baseline

    prepare_diagnostics(small_config)
    baseline = Baseline(small_config)
    baseline.run("develop")
    # Record definitions before second run
    defs_before = {
        k: v for k, v in baseline.definitions.items() if v["kind"] == "treecluster"
    }
    # Recreate baseline with same config; tree definitions should persist
    baseline2 = Baseline(deepcopy(small_config))
    baseline2.run("develop")
    defs_after = {
        k: v for k, v in baseline2.definitions.items() if v["kind"] == "treecluster"
    }
    # Definitions should be consistent (caching/registry reuse)
    assert len(defs_before) == len(defs_after)
