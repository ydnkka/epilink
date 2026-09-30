"""TreeCluster comparators evaluated with the same M-horizon cluster summaries."""
from __future__ import annotations

import io
import shutil
import subprocess
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from evaluation.metrics import bcubed_scores, get_reference_memberships
from synthetic_exploration.clusters import PairLookup
from synthetic_exploration.common import save_table, write_json, log

from .clusters import partition_statistics


def find_treecluster_command() -> str | None:
    command = shutil.which("TreeCluster.py")
    if command is not None:
        return command
    candidate = Path("/opt/homebrew/Caskroom/miniconda/base/envs/epilik_evaluation/bin/TreeCluster.py")
    return str(candidate) if candidate.is_file() else None


def run_treecluster(command: str, tree_path: Path, threshold: float, method: str,
                    timeout: int | None = None) -> pd.DataFrame:
    output = subprocess.check_output(
        [command, "-i", str(tree_path), "-t", str(threshold), "-m", method],
        text=True,
        timeout=timeout,
    )
    frame = pd.read_csv(io.StringIO(output), sep="\t")
    frame.columns = ["SequenceName", "ClusterNumber"]
    frame["SequenceName"] = frame.SequenceName.astype(str)
    return frame


def labels_from_treecluster(partition: pd.DataFrame, lookup: PairLookup) -> np.ndarray:
    expected = lookup.cases.case_id.astype(str).tolist()
    observed = set(partition.SequenceName.astype(str))
    if observed != set(expected):
        missing = sorted(set(expected) - observed)[:5]
        extra = sorted(observed - set(expected))[:5]
        raise ValueError(f"TreeCluster sample mismatch; missing={missing}, extra={extra}")
    by_case = partition.set_index("SequenceName").loc[expected]
    labels = np.empty(len(expected), dtype=np.int32)
    cluster_map: dict[int, int] = {}
    next_label = 0
    for i, raw in enumerate(by_case.ClusterNumber.astype(int)):
        if raw == -1:
            labels[i] = next_label
            next_label += 1
            continue
        if raw not in cluster_map:
            cluster_map[raw] = next_label
            next_label += 1
        labels[i] = cluster_map[raw]
    return labels


def summarise_treecluster_partition(labels: np.ndarray, base: dict, pairs: pd.DataFrame,
                                    lookup: PairLookup, reference: dict) -> tuple[dict, list[dict], pd.DataFrame]:
    summary, _, composition = partition_statistics(labels, pairs, lookup)
    predicted = {int(case): {int(label)} for case, label in zip(lookup.cases.case_id, labels)}
    precision, recall, f1 = bcubed_scores(predicted, reference)
    summary = {**base, **summary, "bcubed_precision": precision,
               "bcubed_recall": recall, "bcubed_f1": f1}
    composition = [{**base, **row} for row in composition]
    memberships = pd.DataFrame({"case_id": lookup.cases.case_id, "cluster_id": labels, **base})
    return summary, composition, memberships


def read_existing_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def partition_key(row: dict | pd.Series) -> tuple:
    threshold_days = row.get("threshold_days", np.nan)
    if pd.isna(threshold_days):
        threshold_days = "nan"
    else:
        threshold_days = f"{float(threshold_days):.12g}"
    return (
        str(row.get("tree_kind")),
        str(row.get("data_process")),
        str(row.get("method")),
        f"{float(row.get('threshold')):.12g}",
        threshold_days,
    )


def configured_tree_specs(config: dict) -> list[tuple[str, str, Path, float, float]]:
    specs = []
    processes = sorted(set(config["raw_trees"]) | set(config["dated_trees"]))
    genetic_thresholds = list(config["genetic_thresholds"])
    threshold_days = list(config["threshold_days"])
    width = max(len(genetic_thresholds), len(threshold_days))
    for process in processes:
        for i in range(width):
            if process in config["raw_trees"] and i < len(genetic_thresholds):
                specs.append(("raw_genetic_tree", process, Path(config["raw_trees"][process]),
                              float(genetic_thresholds[i]), np.nan))
            if process in config["dated_trees"] and i < len(threshold_days):
                days = float(threshold_days[i])
                specs.append(("temporal_dated_tree", process, Path(config["dated_trees"][process]),
                              days / 365.0, days))
    return specs


def investigate_treecluster(pairs: pd.DataFrame, cases: pd.DataFrame, run,
                            directory, settings) -> None:
    log("3/3: TreeCluster raw-genetic and dated-tree summaries")
    directory.mkdir(parents=True, exist_ok=True)
    config = settings["treecluster"]
    if not config.get("enabled", True):
        write_json(directory / "treecluster_status.json", {"status": "disabled"})
        save_table(directory, "treecluster_partition_summary", [])
        save_table(directory, "treecluster_within_cluster_M", [])
        return
    command = find_treecluster_command()
    if command is None:
        write_json(directory / "treecluster_status.json", {
            "status": "skipped",
            "reason": "TreeCluster.py was not found on PATH or the known project environment path.",
        })
        save_table(directory, "treecluster_partition_summary", [])
        save_table(directory, "treecluster_within_cluster_M", [])
        return
    lookup = PairLookup(pairs, cases)
    reference = get_reference_memberships(nx.read_gml(run.tree_path))
    summary_path = directory / "treecluster_partition_summary.csv"
    composition_path = directory / "treecluster_within_cluster_M.csv"
    membership_path = directory / "treecluster_memberships.csv"
    existing_summary = read_existing_csv(summary_path) if config.get("resume_existing", True) else pd.DataFrame()
    existing_composition = read_existing_csv(composition_path) if config.get("resume_existing", True) else pd.DataFrame()
    existing_memberships = read_existing_csv(membership_path) if config.get("resume_existing", True) else pd.DataFrame()
    summaries = existing_summary.to_dict(orient="records")
    compositions = existing_composition.to_dict(orient="records")
    memberships = [existing_memberships] if not existing_memberships.empty else []
    done = {partition_key(row) for row in summaries}
    errors = []
    tree_specs = configured_tree_specs(config)
    started = time.monotonic()
    max_runtime = config.get("max_runtime_seconds")
    command_timeout = config.get("command_timeout_seconds")
    stopped_for_runtime = False
    for tree_kind, process, tree_path, threshold, threshold_days in tree_specs:
        if not tree_path.is_file():
            errors.append({"tree_kind": tree_kind, "data_process": process,
                           "tree_path": str(tree_path), "reason": "tree file missing"})
            continue
        for method in config["methods"]:
            base = {
                "tree_kind": tree_kind,
                "data_process": process,
                "method": method,
                "threshold": threshold,
                "threshold_days": threshold_days,
            }
            if partition_key(base) in done:
                continue
            if max_runtime is not None and time.monotonic() - started >= float(max_runtime):
                stopped_for_runtime = True
                break
            log(f"TreeCluster {tree_kind} {process} {method} threshold={threshold:.6g}")
            try:
                partition = run_treecluster(command, tree_path, threshold, method,
                                            None if command_timeout is None else int(command_timeout))
                labels = labels_from_treecluster(partition, lookup)
                summary, composition, member_frame = summarise_treecluster_partition(
                    labels, base, pairs, lookup, reference)
                summaries.append(summary)
                compositions.extend(composition)
                memberships.append(member_frame)
                done.add(partition_key(base))
            except Exception as exc:  # Keep the rest of the workflow inspectable.
                errors.append({**base, "tree_path": str(tree_path), "reason": repr(exc)})
        save_table(directory, "treecluster_partition_summary", summaries)
        save_table(directory, "treecluster_within_cluster_M", compositions)
        if memberships:
            save_table(directory, "treecluster_memberships", pd.concat(memberships, ignore_index=True))
        if stopped_for_runtime:
            break
    write_json(directory / "treecluster_status.json", {
        "status": "partial_runtime_limit" if stopped_for_runtime else ("complete_with_errors" if errors else "complete"),
        "command": command,
        "completed_partitions": len(summaries),
        "configured_partitions": len(tree_specs) * len(config["methods"]),
        "max_runtime_seconds": max_runtime,
        "errors": errors,
    })
