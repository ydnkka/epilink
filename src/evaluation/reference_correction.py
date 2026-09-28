"""Reproduce the September 2026 reference correction with paired legacy checks.

Run with PYTHONPATH=src:src/evaluation python -m evaluation.reference_correction
{baseline,experiments}. Existing results must first be archived beneath AUDIT.
Checkpoints retain per-resolution old/new scores on identical partitions.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import resource
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate
from .config import build_run_specs, load_config, project_root
from .experiments import _make_row
from .specs import MODEL_KEYS

ROOT = project_root()
AUDIT = ROOT / "results/reference_correction_20260928"
ARCHIVE = AUDIT / "archive/evaluation"


def rerun_treecluster():
    """Rescore existing dated trees; do not reuse the historical metric cache."""
    import shutil
    from .metrics import bcubed_scores

    command = shutil.which("TreeCluster.py")
    if command is None:
        candidate = Path(
            "/opt/homebrew/Caskroom/miniconda/base/envs/epilik_evaluation/bin/TreeCluster.py"
        )
        if candidate.is_file():
            command = str(candidate)
        else:
            raise FileNotFoundError("TreeCluster.py is required on PATH")
    tree_path = str(ROOT / "results/scovmod/scovmod_tree.gml")
    reference = evaluate._reference_memberships(tree_path)
    legacy = legacy_reference(evaluate._load_tree_template(tree_path))
    directory = ROOT / "phylo/synthetic"
    rows = []
    for process in ["deterministic", "stochastic"]:
        dated_tree = directory / f"treetime_{process}/timetree.nwk"
        for method in ["max_clade", "avg_clade", "single_linkage"]:
            for days in range(7, 77, 7):
                output = subprocess.check_output(
                    [
                        command,
                        "-i",
                        str(dated_tree),
                        "-t",
                        str(days / 365),
                        "-m",
                        method,
                    ],
                    text=True,
                )
                partition = pd.read_csv(io.StringIO(output), sep="\t")
                partition.columns = ["SequenceName", "ClusterNumber"]
                predicted = {
                    int(case): {int(label) if label != -1 else 10**7 + i}
                    for i, (case, label) in enumerate(
                        zip(partition.SequenceName, partition.ClusterNumber)
                    )
                }
                assert set(predicted) == set(reference)
                p, r, f1 = bcubed_scores(predicted, reference)
                lp, lr, lf = bcubed_scores(predicted, legacy)
                clustered = partition.ClusterNumber != -1
                rows.append(
                    {
                        "data_process": process,
                        "method": method,
                        "threshold_days": float(days),
                        "threshold_years": days / 365,
                        "bcubed_precision": p,
                        "bcubed_recall": r,
                        "bcubed_f1": f1,
                        "n_clusters": int(
                            partition.loc[clustered, "ClusterNumber"].nunique()
                        ),
                        "n_singletons": int((~clustered).sum()),
                        "legacy_f1": lf,
                    }
                )
            print(f"TreeCluster {process} {method} complete", flush=True)
    sweep = pd.DataFrame(rows)
    old = pd.read_parquet(ARCHIVE / "phylo/synthetic/treecluster_sweep.parquet")
    check = sweep.merge(
        old,
        on=["data_process", "method", "threshold_days"],
        suffixes=("", "_stored"),
        validate="one_to_one",
    )
    np.testing.assert_allclose(
        check.legacy_f1, check.bcubed_f1_stored, rtol=1e-12, atol=1e-14
    )
    sweep.to_parquet(AUDIT / "treecluster_paired_sweep.parquet", index=False)
    sweep.drop(columns="legacy_f1").to_parquet(
        directory / "treecluster_sweep.parquet", index=False
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            {str(case): sorted(labels) for case, labels in sorted(reference.items())},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    (directory / "treecluster_sweep.reference.json").write_text(
        json.dumps({"reference_sha256": fingerprint}, indent=2) + "\n"
    )
    print(
        "TreeCluster: all 60 historical F1 scores reproduced before correction",
        flush=True,
    )


def refresh_treecluster_comparison():
    """Refresh the notebook's comparison table, plot, and displayed outputs."""
    import base64
    import matplotlib.pyplot as plt

    notebook_path = ROOT / "synthetic_treecluster_comparison.ipynb"
    notebook = json.loads(notebook_path.read_text())
    directory = ROOT / "phylo/synthetic"
    namespace = {
        "np": np,
        "pd": pd,
        "plt": plt,
        "OUTDIR": directory,
        "sweep": pd.read_parquet(directory / "treecluster_sweep.parquet"),
        "perf": json.loads(
            (ROOT / "results/synthetic/baseline_performance.json").read_text()
        ),
    }
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] != "code":
            continue
        if source.startswith("tc_best =") or source.startswith(
            "from matplotlib.patches import Patch"
        ):
            exec(compile(source, str(notebook_path), "exec"), namespace)
            cell["execution_count"] = None
            if source.startswith("tc_best ="):
                table = namespace["comparison"]
                cell["outputs"] = [
                    {
                        "output_type": "display_data",
                        "data": {
                            "text/plain": [table.to_string(index=False)],
                            "text/html": [table.to_html(index=False)],
                        },
                        "metadata": {},
                    }
                ]
            else:
                encoded = base64.b64encode(
                    (directory / "treecluster_comparison.png").read_bytes()
                ).decode()
                cell["outputs"] = [
                    {
                        "output_type": "display_data",
                        "data": {"image/png": encoded},
                        "metadata": {},
                    }
                ]
        elif "sweep_path = OUTDIR" in source:
            cell["execution_count"] = None
            cell["outputs"] = [
                {
                    "output_type": "display_data",
                    "data": {
                        "text/plain": [
                            namespace["sweep"]
                            .sort_values("bcubed_f1", ascending=False)
                            .head(10)
                            .to_string(index=False)
                        ]
                    },
                    "metadata": {},
                }
            ]
    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    plt.close("all")


def write_report():
    """Validate completed outputs and write a compact, machine-readable audit."""
    import yaml

    from .metrics import get_reference_memberships
    from .stability import select_shared_resolution

    tree = evaluate._load_tree_template(str(ROOT / "results/scovmod/scovmod_tree.gml"))
    reference = get_reference_memberships(tree)
    assert len(reference) == 4990
    assert set(reference) == {int(node) for node in tree}
    roots = [int(node) for node in tree if tree.in_degree(node) == 0]
    assert roots == [4537061]
    assert all(len(labels) == (1 if case in roots else 2)
               for case, labels in reference.items())
    assert reference == evaluate._reference_memberships(
        str(ROOT / "results/scovmod/scovmod_tree.gml")
    )
    manifest = json.loads((AUDIT / "archive_manifest.json").read_text())
    for name, expected in manifest["input_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for name, expected in manifest["epilink_source_sha256"].items():
        source = Path(manifest["epilink_source"]) / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, name
    current_config = yaml.safe_load((ROOT / "config.yaml").read_text())
    original_config = yaml.safe_load((ARCHIVE / "config.yaml").read_text())
    current_config["workflows"]["boston"]["resolution"] = original_config[
        "workflows"]["boston"]["resolution"]
    assert current_config == original_config, "Only Boston's resolution may change"
    thresholds = Path("results/sparsification/optimal_thresholds.json")
    assert (ROOT / thresholds).read_bytes() == (ARCHIVE / thresholds).read_bytes()
    checkpoints = sorted((AUDIT / "runs").glob("*/result.json"))
    assert len(checkpoints) == 26, (
        f"Expected 26 completed runs; found {len(checkpoints)}"
    )
    checks = [
        row for path in checkpoints for row in json.loads(path.read_text())["checks"]
    ]
    max_deltas = {
        key: max(abs(row[key]) for row in checks)
        for key in ("ap_delta", "legacy_f1_delta", "resolution_stability_delta")
    }
    assert all(value < 1e-10 for value in max_deltas.values()), max_deltas
    base_checkpoint = json.loads(
        (AUDIT / "runs/matched__baseline/result.json").read_text()
    )
    assert base_checkpoint["baseline_scores_and_labels_reproduced"]
    result = pd.read_parquet(ROOT / "results/synthetic/results.parquet")
    assert (
        len(result) == 156
        and not result.duplicated(["condition", "scenario", "model"]).any()
    )
    keys = ["condition", "scenario", "model"]
    independent = ["ap", "ap_loss", "n_pairs", "prevalence", "mean_stability", "std_stability"]
    original_result = pd.read_parquet(ARCHIVE / "results/synthetic/results.parquet")
    original_baseline = original_result.loc[
        (original_result.condition == "matched") & (original_result.scenario == "baseline")
    ].set_index("model")
    pd.testing.assert_frame_equal(
        result.set_index(keys).sort_index()[independent],
        original_result.set_index(keys).sort_index()[independent],
        check_exact=True,
    )
    tc_paired = pd.read_parquet(AUDIT / "treecluster_paired_sweep.parquet")
    tc_current = pd.read_parquet(ROOT / "phylo/synthetic/treecluster_sweep.parquet")
    tc_original = pd.read_parquet(ARCHIVE / "phylo/synthetic/treecluster_sweep.parquet")
    assert len(tc_paired) == len(tc_current) == len(tc_original) == 60
    pd.testing.assert_frame_equal(tc_paired.drop(columns="legacy_f1"), tc_current)
    tc_keys = ["data_process", "method", "threshold_days"]
    tc_check = tc_paired.merge(tc_original, on=tc_keys, suffixes=("", "_stored"),
                               validate="one_to_one")
    np.testing.assert_allclose(tc_check.legacy_f1, tc_check.bcubed_f1_stored,
                               rtol=1e-12, atol=1e-14)
    tc_best = tc_current.loc[tc_current.groupby("data_process").bcubed_f1.idxmax()]
    cv_sweep = pd.read_parquet(
        AUDIT / "runs/mismatched__incubation_cv_1.25/resolution_sweep.parquet"
    )
    cv_best = cv_sweep.loc[cv_sweep.groupby("model").f1.idxmax(),
                           ["model", "resolution", "precision", "recall", "f1"]]
    cv_best = cv_best.merge(
        result.loc[(result.condition == "mismatched")
                   & (result.scenario == "incubation_cv_1.25"), ["model", "f1_loss"]],
        on="model", validate="one_to_one",
    )
    base = pd.read_parquet(
        ROOT / "results/synthetic/baseline_summary.parquet"
    ).set_index("model")
    old_base = pd.read_parquet(
        ARCHIVE / "results/synthetic/baseline_summary.parquet"
    ).set_index("model")
    unmodified = [
        c for c in base if c not in ("best_f1", "mean_stability", "std_stability")
    ]
    pd.testing.assert_frame_equal(base[unmodified], old_base[unmodified])
    for row in result.loc[result.f1_loss.notna()].itertuples():
        np.testing.assert_allclose(
            row.f1_loss,
            (row.best_f1 - base.loc[row.model, "best_f1"])
            / base.loc[row.model, "best_f1"],
            atol=1e-14,
        )
    old_scores = ARCHIVE / "results/synthetic/baseline_scores.parquet"
    assert (
        hashlib.sha256(old_scores.read_bytes()).digest()
        == hashlib.sha256(
            (ROOT / "results/synthetic/baseline_scores.parquet").read_bytes()
        ).digest()
    )
    selection = pd.read_parquet(
        ROOT / "results/stability/stability_resolution_selection.parquet"
    )
    paired = pd.read_parquet(AUDIT / "stability_paired_sweep.parquet")
    historical = pd.read_parquet(
        ARCHIVE / "results/stability/stability_resolution_selection.parquet"
    )
    matched = paired.merge(
        historical,
        on=["weight", "resolution"],
        suffixes=("", "_stored"),
        validate="one_to_one",
    )
    np.testing.assert_allclose(matched.legacy_f1, matched.f1_score_stored, atol=1e-14)
    pd.testing.assert_frame_equal(
        pd.read_parquet(ROOT / "results/stability/case_counts_over_time.parquet"),
        pd.read_parquet(ARCHIVE / "results/stability/case_counts_over_time.parquet"),
    )
    chosen = select_shared_resolution(selection)
    assert chosen == load_config()["workflows"]["boston"]["resolution"]
    baseline_sweep = pd.read_parquet(
        AUDIT / "runs/matched__baseline/resolution_sweep.parquet"
    )
    summary = []
    for model in MODEL_KEYS:
        best = (
            baseline_sweep.loc[baseline_sweep.model == model]
            .sort_values(["f1", "resolution"], ascending=[False, True])
            .iloc[0]
        )
        temporal = pd.read_parquet(
            ROOT / f"results/stability/temporal_stability_{model}.parquet"
        )
        assert len(temporal) == 25 and np.isfinite(temporal.jaccard).all()
        row = {
            "model": model,
            "old_f1": float(original_baseline.loc[model, "best_f1"]),
            "f1": float(base.loc[model, "best_f1"]),
            "precision": float(best.precision),
            "recall": float(best.recall),
            "best_resolution": float(best.resolution),
            "temporal_resolution": float(
                selection.loc[selection.weight == model]
                .sort_values(["f1_score", "resolution"], ascending=[False, True])
                .iloc[0].resolution
            ),
            "resolution_stability": float(base.loc[model, "mean_stability"]),
            "temporal_jaccard_mean": float(temporal.jaccard.mean()),
            "temporal_jaccard_min": float(temporal.jaccard.min()),
        }
        summary.append(row)
    report = {
        "status": "validated",
        "seed": 12345,
        "reference_cases": len(reference),
        "root_case": 4537061,
        "completed_synthetic_runs": len(checkpoints),
        "synthetic_result_rows": len(result),
        "max_reproduction_deltas": max_deltas,
        "baseline_pair_labels_and_scores_reproduced": True,
        "ap_summaries_retained": True,
        "inputs_and_model_sources_unchanged": True,
        "simulation_design_thresholds_grid_and_restarts_unchanged": True,
        "treecluster_settings_recomputed": len(tc_current),
        "treecluster_best": tc_best.to_dict(orient="records"),
        "increased_incubation_cv_under_mismatch": cv_best.to_dict(orient="records"),
        "boston_resolution": chosen,
        "boston_resolution_selection": json.loads(
            (AUDIT / "boston_resolution_selection.json").read_text()
        ),
        "baseline": summary,
        "boston": json.loads(
            (
                ROOT
                / "results/chapter3_descriptives/boston_descriptive_validation.json"
            ).read_text()
        ),
        "scope_note": "Boston's empirical EpiLink tuning now uses the four EpiLink configurations and selects 0.3; the six-model criterion selecting 0.2 is retained in the audit. Scottish results and their primary resolution 0.3 were not reanalysed. The thesis describes the final methods without draft correction history.",
    }
    (AUDIT / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Chapter 3 reference-membership correction",
        "",
        "All 26 synthetic runs, the early-case sweep, temporal partitions, 60 TreeCluster settings, and Boston summaries have been recalculated.",
        "",
        "The corrected reference covers exactly 4,990 cases, including root 4537061. Historical F1 and reference-independent metrics were checked against archived outputs before accepting changes. Baseline pair labels and scores reproduced; AP summaries were retained.",
        "The root has one membership and every other case has two. Input data, dated trees, EpiLink model sources, simulation settings, thresholds, resolution grids, and restart counts are unchanged. Boston's selection criterion now includes only EpiLink; its selected resolution returns to the original value 0.3.",
        "",
        "| Model | Previous F1 | Corrected F1 | Precision | Recall | Baseline resolution | Temporal resolution | Temporal Jaccard |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {r['model']} | {r['old_f1']:.6f} | {r['f1']:.6f} | {r['precision']:.6f} | {r['recall']:.6f} | {r['best_resolution']:.1f} | {r['temporal_resolution']:.1f} | {r['temporal_jaccard_mean']:.6f} |"
        for r in summary
    ]
    lines += [
        "",
        f"Boston's minimum-mean-shortfall resolution is {chosen:g} across EDD, EDS, ESD, and ESS. The earlier six-model criterion selected 0.2. The revised scope matches empirical EpiLink tuning, while logistic comparators remain in the synthetic and temporal benchmarks. Both selection criteria are recorded in boston_resolution_selection.json; the previous outputs are preserved in epilink_only_boston_20260928/before/. Regenerated memberships and descriptive checks are recorded in validation.json.",
        "",
        "| TreeCluster input | Selected method | Threshold (days) | Precision | Recall | F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {r.data_process} | {r.method} | {r.threshold_days:g} | {r.bcubed_precision:.6f} | {r.bcubed_recall:.6f} | {r.bcubed_f1:.6f} |"
        for r in tc_best.itertuples()
    ]
    lines += [
        "",
        "The largest corrected sensitivity loss occurs with increased incubation CV under mismatch: all six models lose 55.7–65.6% of baseline F1. At their best settings, precision remains 0.979–0.997 but recall falls to 0.112–0.134. The same partitions exactly reproduce the archived F1 values under the historical reference, and AP is unchanged. The F1 figure's axis now includes these losses in full.",
        "",
        "The sparse BCubed calculation is mathematically equivalent to the package definition and was tested against that independent implementation on overlapping and hard assignments. Self-pairs, multiplicities, and case weighting are retained.",
        "Scenario-summary figures keep the existing 95% bootstrap-interval method and use seed 12345 for reproducible rendering; the earlier plot did not specify this seed. Saved baseline AP confidence intervals are unchanged.",
        "",
        "## Scope",
        "",
        report["scope_note"],
        "",
        "## Reproduction",
        "",
        "See README.md for audit commands. archive_manifest.json records input hashes, archived file hashes, package versions, and EpiLink source hashes. Per-run paired resolution sweeps and reproduction checks are in runs/.",
        "",
    ]
    (AUDIT / "report.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "boston"}, indent=2
        ),
        flush=True,
    )


def legacy_reference(tree):
    """Historical bug, used only for paired verification of archived results."""
    memberships = defaultdict(set)
    for cluster_id, node in enumerate(tree):
        for member in set(node).union(tree.successors(node)):
            memberships[int(member)].add(cluster_id)
    return dict(memberships)


def run_one(run, baseline_performance=None, classifiers=None, baseline=False):
    config = load_config()
    name = f"{run.condition}__{run.scenario_name}"
    out = AUDIT / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "result.json"
    if checkpoint.exists():
        data = json.loads(checkpoint.read_text())
        return data["rows"]
    start = time.monotonic()
    print(f"START {name}", flush=True)
    tree_path = str(Path(run.tree_path).resolve())
    corrected = evaluate._reference_memberships(tree_path)
    legacy = legacy_reference(evaluate._load_tree_template(tree_path))
    original_scores = evaluate.bcubed_scores
    original_clustering = evaluate._clustering
    sweep = []
    model_index = 0
    resolution_index = 0
    model = None

    def paired_scores(predicted_memberships, reference_memberships):
        nonlocal resolution_index
        current = original_scores(predicted_memberships, reference_memberships)
        if reference_memberships is corrected:
            previous = original_scores(predicted_memberships, legacy)
            sweep.append(
                {
                    "model": model,
                    "resolution": round(0.1 * (resolution_index + 1), 1),
                    "precision": current[0],
                    "recall": current[1],
                    "f1": current[2],
                    "legacy_precision": previous[0],
                    "legacy_recall": previous[1],
                    "legacy_f1": previous[2],
                }
            )
            resolution_index += 1
        return current

    def audited_clustering(*args, **kwargs):
        nonlocal model_index, resolution_index, model
        model = MODEL_KEYS[model_index]
        model_index += 1
        resolution_index = 0
        print(f"CLUSTER {name} {model}", flush=True)
        if baseline:
            saved = pd.read_parquet(
                ARCHIVE / "results/synthetic/baseline_scores.parquet", columns=[model]
            )[model].to_numpy()
            np.testing.assert_allclose(
                args[3],
                saved,
                rtol=1e-12,
                atol=1e-14,
                err_msg=f"Baseline pair scores changed for {model}",
            )
        result = original_clustering(*args, **kwargs)
        pd.DataFrame(sweep).to_parquet(out / "resolution_sweep.parquet", index=False)
        print(
            f"SCORED {name} {model} F1={result[0]:.6f} elapsed={time.monotonic() - start:.1f}s",
            flush=True,
        )
        return result

    evaluate.bcubed_scores = paired_scores
    evaluate._clustering = audited_clustering
    kwargs = dict(config["execution"]["evaluate_kwargs"])
    kwargs.update(
        sparsification=json.loads(
            (ROOT / "results/sparsification/optimal_thresholds.json").read_text()
        ),
        rng_seed=int(config["rng_seed"]),
        return_classifier=baseline,
        return_scores=baseline,
    )
    try:
        result, fitted, scores = evaluate.evaluate_scenario(
            tree_path=tree_path,
            scenario_name=run.scenario_name,
            generation_parameters=run.generation_parameters,
            inference_parameters=run.inference_parameters,
            logistic_classifier=classifiers,
            baseline_performance=baseline_performance,
            **kwargs,
        )
    finally:
        evaluate.bcubed_scores = original_scores
        evaluate._clustering = original_clustering
    frame = pd.DataFrame(sweep)
    if len(frame) != 60:
        raise AssertionError(f"Expected 60 resolution evaluations, found {len(frame)}")
    frame.to_parquet(out / "resolution_sweep.parquet", index=False)
    rows = [_make_row(run, result, key, value) for key, value in result.models.items()]
    old = pd.read_parquet(ARCHIVE / "results/synthetic/results.parquet")
    old = old.loc[
        (old.condition == run.condition) & (old.scenario == run.scenario_name)
    ].set_index("model")
    checks = []
    for key, value in result.models.items():
        previous = old.loc[key]
        np.testing.assert_allclose(
            value.ap,
            previous.ap,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"AP changed for {name} {key}",
        )
        np.testing.assert_allclose(
            result.prevalence, previous.prevalence, rtol=0, atol=1e-15
        )
        assert result.n_pairs == previous.n_pairs
        legacy_best = float(frame.loc[frame.model == key, "legacy_f1"].max())
        np.testing.assert_allclose(
            legacy_best,
            previous.best_f1,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"Historical F1 failed to reproduce for {name} {key}",
        )
        np.testing.assert_allclose(
            [value.mean_stability, value.std_stability],
            [previous.mean_stability, previous.std_stability],
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"Reference-independent resolution stability changed for {name} {key}",
        )
        checks.append(
            {
                "model": key,
                "ap_delta": value.ap - previous.ap,
                "legacy_f1_delta": legacy_best - previous.best_f1,
                "resolution_stability_delta": value.mean_stability
                - previous.mean_stability,
            }
        )
    if baseline:
        saved_labels = pd.read_parquet(
            ARCHIVE / "results/synthetic/baseline_scores.parquet", columns=["IsRelated"]
        )
        np.testing.assert_array_equal(
            scores.IsRelated.to_numpy(), saved_labels.IsRelated.to_numpy()
        )
        with (AUDIT / "baseline_classifiers.pkl").open("wb") as handle:
            pickle.dump(fitted, handle)
        old_summary = pd.read_parquet(
            ARCHIVE / "results/synthetic/baseline_summary.parquet"
        )
        summary = old_summary.copy()
        performance = {}
        for key, value in result.models.items():
            performance[key] = {
                "n_pairs": result.n_pairs,
                "prevalence": result.prevalence,
                **{
                    k: v
                    for k, v in asdict(value).items()
                    if k not in ("ap_loss", "f1_loss")
                },
            }
            for column in ("best_f1", "mean_stability", "std_stability"):
                summary.loc[summary.model == key, column] = getattr(value, column)
        summary.to_parquet(
            ROOT / "results/synthetic/baseline_summary.parquet", index=False
        )
        (ROOT / "results/synthetic/baseline_performance.json").write_text(
            json.dumps(performance, indent=2) + "\n"
        )
    checkpoint.write_text(
        json.dumps(
            {
                "rows": rows,
                "checks": checks,
                "elapsed_seconds": time.monotonic() - start,
                "peak_rss_platform_units": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
                "baseline_scores_and_labels_reproduced": baseline,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"DONE {name} {time.monotonic() - start:.1f}s {checks}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "baseline",
            "experiments",
            "stability",
            "select-boston",
            "treecluster",
            "treecluster-figure",
            "report",
        ],
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.stage == "report":
        write_report()
        return
    if args.stage == "treecluster-figure":
        refresh_treecluster_comparison()
        return
    if args.stage == "treecluster":
        rerun_treecluster()
        return
    if args.stage == "stability":
        from . import stability

        old_reference = legacy_reference(
            evaluate._load_tree_template(str(ROOT / "results/scovmod/scovmod_tree.gml"))
        )
        original = stability.bcubed_scores
        paired = []

        def score(predicted, reference):
            current = original(predicted, reference)
            old = original(predicted, old_reference)
            index = len(paired)
            paired.append(
                {
                    "weight": MODEL_KEYS[index // 10],
                    "resolution": round((index % 10 + 1) / 10, 1),
                    "precision": current[0],
                    "recall": current[1],
                    "f1_score": current[2],
                    "legacy_f1": old[2],
                }
            )
            pd.DataFrame(paired).to_parquet(
                AUDIT / "stability_paired_sweep.parquet", index=False
            )
            return current

        stability.bcubed_scores = score
        try:
            stability.main()
        finally:
            stability.bcubed_scores = original
        return
    if args.stage == "select-boston":
        from .stability import epilink_resolution_shortfalls, select_shared_resolution
        import re

        selection = pd.read_parquet(
            ROOT / "results/stability/stability_resolution_selection.parquet"
        )
        chosen = select_shared_resolution(selection)
        previous_resolution = load_config()["workflows"]["boston"]["resolution"]
        config_path = ROOT / "config.yaml"
        source = config_path.read_text()
        source, count = re.subn(
            r"(workflows:\n  boston:\n    minimum_edge_weight: [^\n]+\n    resolution: )[^\n]+",
            lambda match: match[1] + str(chosen),
            source,
        )
        assert count == 1
        config_path.write_text(source)
        scores = selection.pivot(
            index="resolution", columns="weight", values="f1_score"
        ).sort_index()
        all_model_regret = (scores.max() - scores).mean(axis=1)
        shortfalls = epilink_resolution_shortfalls(selection)
        regret = shortfalls.mean(axis=1)
        (AUDIT / "boston_resolution_selection.json").write_text(
            json.dumps(
                {
                    "previous_resolution": previous_resolution,
                    "selected_resolution": chosen,
                    "rule": "minimum equally weighted mean F1 shortfall across four EpiLink configurations on initial weekly cases; lower resolution breaks ties",
                    "models": list(shortfalls.columns),
                    "mean_regret": {str(k): v for k, v in regret.items()},
                    "all_six_model_comparison": {
                        "selected_resolution": float(all_model_regret.idxmin()),
                        "mean_regret": {str(k): v for k, v in all_model_regret.items()},
                    },
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Boston resolution {chosen:g}", flush=True)
        return
    runs = build_run_specs(load_config())
    baseline = next(
        run
        for run in runs
        if run.condition == "matched" and run.scenario_name == "baseline"
    )
    if args.stage == "baseline":
        run_one(baseline, baseline=True)
        return
    rows = json.loads((AUDIT / "runs/matched__baseline/result.json").read_text())[
        "rows"
    ]
    performance = json.loads(
        (ROOT / "results/synthetic/baseline_performance.json").read_text()
    )
    with (AUDIT / "baseline_classifiers.pkl").open("rb") as handle:
        classifiers = pickle.load(handle)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                run_one,
                run,
                performance,
                classifiers if run.logit_training_source == "baseline" else None,
            )
            for run in runs
            if run != baseline
        ]
        for future in as_completed(futures):
            rows.extend(future.result())
    result = pd.DataFrame(rows).sort_values(["condition", "scenario", "model"])
    assert len(result) == 156
    result.to_parquet(ROOT / "results/synthetic/results.parquet", index=False)
    print("Published 156 corrected synthetic result rows", flush=True)


if __name__ == "__main__":
    main()
