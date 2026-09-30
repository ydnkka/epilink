"""Empirical Boston SARS-CoV-2 clustering with frozen baseline settings.

Applies matched-baseline EpiLink models and operating points to observed
TN93 pairs. The TN93 table is distance-censored; missing pairs are unobserved,
not zero. Coverage is explicit and reported.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from ..clusterers.graph import build_graph, components, leiden
from ..provenance import (
    complete_artifact,
    digest_file,
    fingerprint,
    git_revision,
    read_json,
    valid_artifact,
    write_json,
)
from ..reference import BaselineReference
from ..scorers import SCORERS, ScoringContext
from .boston_config import load_study_config

LOG = logging.getLogger(__name__)


def load_boston_inputs(cases_path, pairs_path):
    cases = pd.read_parquet(cases_path)
    pairs = pd.read_parquet(pairs_path)
    cases["case_id"] = cases.case_id.astype(str)
    pairs["CaseID1"] = pairs.CaseID1.astype(str)
    pairs["CaseID2"] = pairs.CaseID2.astype(str)
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
        self.root = Path(config["output_directory"]).resolve()
        self.reference = BaselineReference(config["baseline_run"], config["implementation"])
        fresh_seeds = config["seeds"]
        used_seeds = {seed for seeds in self.reference.config["splits"].values() for seed in seeds}
        if fresh_seeds.intersection(used_seeds):
            raise ValueError("Boston seeds must be fresh relative to every baseline split")
        self.seeds = fresh_seeds
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
        self.context = ScoringContext(self.config, deepcopy(self.reference.models))
        self.signature = {
            "kind": "boston-empirical-v1",
            "reference": self.reference.identity,
            "boston_cases_sha256": digest_file(self.cases_path),
            "boston_pairs_sha256": digest_file(self.pairs_path),
            "implementation": config["implementation"],
            "tools": self.reference.tools,
            "n_cases": self.n_cases,
            "n_observed_pairs": self.n_observed_pairs,
            "n_all_pairs": self.n_all_pairs,
            "candidate_universe": "TN93-censored; missing pairs are unobserved, not zero",
        }
        self.directory = self.root / "runs" / fingerprint(self.signature)[:20]
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.directory / "reference.json", self.reference.identity)
        write_json(self.directory / "selection.json", self.reference.frozen)
        write_json(self.directory / "settings.json", self.reference.selected)
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
        scores = {}
        for name in self.config["scorers"]:
            spec = SCORERS[name].spec
            if spec.family == "logistic":
                scores[name] = pd.Series(
                    SCORERS[name].predict(self.observations, self.context),
                    name=name,
                )
            elif spec.family == "epilink":
                scores[name] = pd.Series(
                    SCORERS[name].predict(self.observations, self.context),
                    name=name,
                )
            else:
                scores[name] = self.observations["GD" if spec.data_process == "deterministic" else "gd"]
        score_frame = pd.concat(scores, axis=1)
        score_id = fingerprint({k: v.tolist() for k, v in scores.items()})[:20]
        score_dir = self.root / "artifacts/scores" / score_id
        if not valid_artifact(score_dir, {"kind": "boston-scores-v1", "score_id": score_id}):
            score_dir.mkdir(parents=True, exist_ok=True)
            score_frame.to_parquet(score_dir / "scores.parquet", index=False)
            complete_artifact(score_dir, {"kind": "boston-scores-v1", "score_id": score_id}, ["scores.parquet"])
        return score_frame, score_id

    def clusters(self, scores, score_id):
        definitions = self.reference.selected
        cluster_definitions = {k: v for k, v in definitions.items() if v["kind"] != "pairwise"}
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
            base = {"setting_id": key, "pipeline": definition["pipeline"], "data_process": definition["data_process"]}
            if valid_artifact(artifact, signature):
                rows.append({**base, **read_json(artifact / "metrics.json")})
                continue
            artifact.mkdir(parents=True, exist_ok=True)
            try:
                spec = SCORERS[definition["score_name"]].spec
                values = scores[spec.name].to_numpy(float)
                requested_graph = (spec.name, definition["threshold"], definition["weight_policy"])
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
                    partitions[partition_id] = (cluster_table, memberships)
                cluster_table, memberships = partitions[partition_id]
                memberships.to_parquet(artifact / "memberships.parquet", index=False)
                cluster_table.to_parquet(artifact / "clusters.parquet", index=False)
                write_json(artifact / "metrics.json", cluster_table.to_dict(orient="list"))
                write_json(artifact / "algorithm.json", metadata)
                complete_artifact(
                    artifact,
                    signature,
                    ["memberships.parquet", "clusters.parquet", "metrics.json", "algorithm.json"],
                )
                rows.append({**base, **cluster_table.to_dict(orient="list")})
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
        _, labels_unique, sizes = np.unique(labels, return_inverse=True, return_counts=True)
        summary = {
            "n_cases": self.n_cases,
            "n_clusters": len(sizes),
            "n_singletons": int(np.sum(sizes == 1)),
            "singleton_fraction": float(np.sum(sizes == 1) / self.n_cases),
            "largest_cluster": int(sizes.max()),
            "largest_cluster_fraction": float(sizes.max() / self.n_cases),
            "size_mean": float(sizes.mean()),
            "size_std": float(sizes.std()) if len(sizes) > 1 else 0.0,
        }
        for col in ["Exposure", "Clade", "Mutation"]:
            if col in self.cases.columns:
                merged = memberships.merge(self.cases[["case_id", col]], on="case_id", validate="many_to_one")
                for label, count in merged.groupby("cluster_id")[col].value_counts().items():
                    pass
                counts = merged.groupby("cluster_id")[col].value_counts().unstack(fill_value=0)
                for c in counts.columns:
                    summary[f"count_{col}_{c}"] = counts[c].tolist()
        cluster_pairs = sizes.astype(np.int64) * (sizes - 1) // 2
        clusters = pd.DataFrame({
            "cluster_id": np.arange(len(sizes)),
            "n_cases": sizes,
            "within_pairs": cluster_pairs,
        })
        for col in ["Exposure", "Clade", "Mutation"]:
            if col in self.cases.columns:
                merged = memberships.merge(self.cases[["case_id", col]], on="case_id", validate="many_to_one")
                for val in merged[col].unique():
                    mask = merged[col] == val
                    positive = np.bincount(merged.loc[mask, "cluster_id"], minlength=len(sizes))
                    clusters[f"n_{col}_{val}"] = positive
        summary["clusters"] = clusters.to_dict(orient="list")
        return pd.DataFrame({k: [v] if not isinstance(v, list) else v for k, v in summary.items()})

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
    import sys
    from .boston_config import load_study_config

    config = load_study_config(argv)
    return 0 if BostonEmpirical(config).run() else 1
