"""Empirical Boston SARS-CoV-2 clustering with frozen baseline settings.

Applies training-free EpiLink scores and frozen operating points to observed
TN93 pairs. The TN93 table is distance-censored; missing pairs are unobserved,
not zero. Coverage is explicit and reported.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from ..clusterers.graph import components, leiden
from ..graphs.construction import build_graph
from ..provenance import (
    complete_artifact,
    digest_file,
    fingerprint,
    git_revision,
    read_json,
    valid_artifact,
    write_json,
)
from ..scorers import ScoringContext
from .boston_config import load_study_config
from .boston_scoring import BOSTON_SPECS, operating_settings, score_observations, validate_scorers
from .reference import OperatingReference

LOG = logging.getLogger(__name__)


def load_boston_inputs(cases_path, pairs_path):
    cases = pd.read_parquet(cases_path)
    pairs = pd.read_parquet(pairs_path)
    if cases.empty or cases.case_id.isna().any():
        raise ValueError("Boston cases must have nonmissing identifiers")
    if pairs[["CaseID1", "CaseID2"]].isna().any().any():
        raise ValueError("Boston pair identifiers must be nonmissing")
    cases["case_id"] = cases.case_id.astype(str)
    pairs["CaseID1"] = pairs.CaseID1.astype(str)
    pairs["CaseID2"] = pairs.CaseID2.astype(str)
    if cases.case_id.duplicated().any():
        raise ValueError("Boston case identifiers must be unique")
    canonical = np.sort(pairs[["CaseID1", "CaseID2"]].to_numpy(str), axis=1)
    if (canonical[:, 0] == canonical[:, 1]).any() or pd.DataFrame(canonical).duplicated().any():
        raise ValueError("Boston pairs must be distinct unordered non-self pairs")
    missing = (set(pairs.CaseID1) | set(pairs.CaseID2)) - set(cases.case_id)
    if missing:
        raise ValueError(f"Pair IDs lack metadata: {sorted(missing)[:5]}")
    return cases, pairs


def build_observations(cases, pairs):
    case_index = pd.Series(np.arange(len(cases)), index=cases.case_id, name="idx")
    a = pairs.CaseID1.map(case_index).to_numpy(np.int32)
    b = pairs.CaseID2.map(case_index).to_numpy(np.int32)
    td = pairs.TD.to_numpy(float)
    gd = pairs.GD.to_numpy(float)
    tn93 = pairs.TN93_distance.to_numpy(float)
    observations = pd.DataFrame({"a": a, "b": b, "TD": td, "GD": gd, "tn93": tn93})
    return observations, case_index


class BostonEmpirical:
    def __init__(self, config):
        self.config = deepcopy(config)
        validate_scorers(config["scorers"])
        self.root = Path(config["output_directory"]).resolve()
        self.reference = OperatingReference(config["baseline_run"], config["implementation"])
        source = self.reference.root.resolve()
        if self.root == source or self.root.is_relative_to(source) or source.is_relative_to(self.root):
            raise ValueError("Boston output must be separate from baseline outputs")
        self.definitions, self.selection = operating_settings(self.reference, config["scorers"])
        self.cases_path = Path(config["inputs"]["cases_path"])
        self.pairs_path = Path(config["inputs"]["pairs_path"])
        if not self.cases_path.exists():
            raise FileNotFoundError(f"Boston cases not found: {self.cases_path}")
        if not self.pairs_path.exists():
            raise FileNotFoundError(f"Boston pairs not found: {self.pairs_path}")
        self.cases, self.pairs = load_boston_inputs(self.cases_path, self.pairs_path)
        self.observations, self.case_index = build_observations(self.cases, self.pairs)
        self.n_cases, self.n_observed_pairs = len(self.cases), len(self.pairs)
        self.n_all_pairs = self.n_cases * (self.n_cases - 1) // 2
        self.scoring_config = {
            key: deepcopy(self.reference.config[key]) for key in ("inference", "scorer")
        }
        self.context = ScoringContext(self.scoring_config, logistic_models={})
        self.signature = {
            "kind": "boston-training-free-v2",
            "reference": self.reference.identity,
            "boston_cases_sha256": digest_file(self.cases_path),
            "boston_pairs_sha256": digest_file(self.pairs_path),
            "implementation": config["implementation"],
            "scorers": {name: BOSTON_SPECS[name].metadata() for name in config["scorers"]},
            "scoring_config": self.scoring_config,
            "definitions": self.definitions,
            "n_cases": self.n_cases,
            "n_observed_pairs": self.n_observed_pairs,
            "n_all_pairs": self.n_all_pairs,
            "candidate_universe": "TN93-censored; missing pairs are unobserved, not zero",
        }
        self.directory = self.root / "runs" / fingerprint(self.signature)[:20]
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.directory / "reference.json", self.reference.identity)
        write_json(self.directory / "selection.json", self.selection)
        write_json(self.directory / "settings.json", self.definitions)
        write_json(self.directory / "inputs.json", {
            "cases_path": str(self.cases_path),
            "pairs_path": str(self.pairs_path),
            "n_cases": self.n_cases,
            "n_observed_pairs": self.n_observed_pairs,
            "n_all_pairs": self.n_all_pairs,
        })
        write_json(self.root / "current.json", {
            "run_directory": str(self.directory),
            "fingerprint": fingerprint(self.signature),
        })

    def score(self):
        signature = {
            "kind": "boston-scores-v2", "run": fingerprint(self.signature),
            "scorers": self.config["scorers"],
        }
        score_id = fingerprint(signature)[:20]
        score_dir = self.root / "artifacts/scores" / score_id
        if not valid_artifact(score_dir, signature):
            values = score_observations(self.observations, self.context, self.config["scorers"])
            score_frame = pd.concat([
                self.pairs[["CaseID1", "CaseID2"]].reset_index(drop=True), values.reset_index(drop=True),
            ], axis=1)
            score_dir.mkdir(parents=True, exist_ok=True)
            score_frame.to_parquet(score_dir / "scores.parquet", index=False)
            complete_artifact(score_dir, signature, ["scores.parquet"])
        else:
            score_frame = pd.read_parquet(score_dir / "scores.parquet")
        return score_frame, score_id

    def clusters(self, scores, score_id):
        cluster_definitions = {k: v for k, v in self.definitions.items()
                               if v["kind"] in ("components", "leiden")}
        (self.directory / "clusters").mkdir(parents=True, exist_ok=True)
        all_complete = True
        rows, errors = [], []
        graphs, partitions = {}, {}
        for key, definition in cluster_definitions.items():
            artifact = self.directory / "clusters" / key
            signature = {
                "run": fingerprint(self.signature),
                "score_id": score_id,
                "definition": definition,
            }
            base = {
                "setting_id": key, "pipeline": definition["pipeline"],
                "score_name": definition["score_name"], "data_process": "empirical",
                "baseline_setting_id": definition["baseline_setting_id"],
                "baseline_score_name": definition["baseline_score_name"],
            }
            if valid_artifact(artifact, signature):
                rows.append({**base, **read_json(artifact / "metrics.json")})
                continue
            artifact.mkdir(parents=True, exist_ok=True)
            try:
                spec = BOSTON_SPECS[definition["score_name"]]
                values = scores[spec.name].to_numpy(float)
                requested_graph = (spec.name, definition["threshold"], definition["weight_policy"],
                                   definition["empty"])
                if requested_graph not in graphs:
                    graphs[requested_graph] = build_graph(
                        self.observations,
                        self.n_cases,
                        values,
                        spec,
                        definition["threshold"],
                        definition["weight_policy"],
                        definition.get("empty", False),
                    )
                graph = graphs[requested_graph]
                if definition["kind"] == "components":
                    labels, metadata = components(graph)
                else:
                    labels, metadata = leiden(
                        graph,
                        definition["resolution"],
                        definition["objective"],
                        definition["restarts"],
                        definition["algorithm_seed"],
                    )
                metadata["retained_graph_edges"] = graph.ecount()
                _, labels = np.unique(labels, return_inverse=True)
                partition_id = fingerprint(labels.tolist())
                if partition_id not in partitions:
                    memberships = pd.DataFrame({
                        "case_id": self.cases.case_id,
                        "cluster_id": labels,
                    })
                    cluster_table = self._summarize_partition(memberships)
                    partitions[partition_id] = cluster_table
                cluster_table = partitions[partition_id]
                memberships = pd.DataFrame({
                    "case_id": self.cases.case_id,
                    "cluster_id": labels,
                })
                memberships.to_parquet(artifact / "memberships.parquet", index=False)
                cluster_table.to_parquet(artifact / "clusters.parquet", index=False)
                metrics_dict = {
                    "n_cases": self.n_cases,
                    "n_clusters": len(cluster_table),
                    "size_mean": float(cluster_table["n_cases"].mean()),
                    "size_std": float(cluster_table["n_cases"].std()) if len(cluster_table) > 1 else 0.0,
                    "largest_cluster": int(cluster_table["n_cases"].max()),
                }
                write_json(artifact / "metrics.json", metrics_dict)
                write_json(artifact / "algorithm.json", metadata)
                complete_artifact(
                    artifact,
                    signature,
                    ["memberships.parquet", "clusters.parquet", "metrics.json", "algorithm.json"],
                )
                rows.append({**base, **metrics_dict})
            except Exception as exc:
                LOG.error("Cluster failed setting=%s: %s", key, exc)
                error = {**base, "definition": definition, "error": repr(exc)}
                write_json(artifact / "manifest.json", {"status": "failed", **error})
                errors.append(error)
        pd.DataFrame(rows).to_csv(self.directory / "clusters" / "metrics.csv", index=False)
        write_json(
            self.directory / "clusters" / "status.json",
            {
                "status": "partial" if errors else "complete",
                "configured": len(cluster_definitions),
                "completed": len(rows),
                "errors": errors,
            },
        )
        all_complete &= not errors
        return all_complete

    def _summarize_partition(self, memberships):
        labels = memberships.cluster_id.to_numpy()
        _, sizes = np.unique(labels, return_counts=True)
        cluster_pairs = sizes.astype(np.int64) * (sizes - 1) // 2
        clusters = pd.DataFrame({
            "cluster_id": np.arange(len(sizes)),
            "n_cases": sizes,
            "within_pairs": cluster_pairs,
        })
        for col in ["Exposure", "Clade", "Mutation"]:
            if col in self.cases.columns:
                merged = memberships.merge(self.cases[["case_id", col]], on="case_id", validate="many_to_one")
                for val in merged[col].dropna().unique():
                    mask = merged[col] == val
                    positive = np.bincount(merged.loc[mask, "cluster_id"].to_numpy(int), minlength=len(sizes))
                    clusters[f"n_{col}_{val}"] = positive
        return clusters

    def run(self):
        manifest = {
            "status": "running",
            "config": self.config,
            "signature": self.signature,
            "git_revision": git_revision(),
            "run_directory": str(self.directory),
        }
        write_json(self.directory / "manifest.json", manifest)
        try:
            scores, score_id = self.score()
            manifest["score_id"] = score_id
            self.directory.mkdir(parents=True, exist_ok=True)
            complete = self.clusters(scores, score_id)
            manifest["status"] = "complete" if complete else "partial"
        except Exception as exc:
            manifest.update(status="failed", error=repr(exc))
            raise
        finally:
            write_json(self.directory / "manifest.json", manifest)
            from ..reporting.boston import render_report
            render_report(self.directory)
        LOG.info("Boston report: %s", self.directory / "report.html")
        return manifest["status"] == "complete"


def main(argv=None):
    config = load_study_config(argv)
    return 0 if BostonEmpirical(config).run() else 1
