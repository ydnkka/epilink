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
from ..clusterers.treecluster import treecluster
from ..graphs.construction import build_graph
from ..phylogeny.boston import dated_boston_tree, raw_boston_tree
from ..phylogeny.external import command_identity
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
from .boston_assessment import assess_partitions, load_treecluster
from .boston_config import load_study_config
from .boston_scoring import (
    BOSTON_SPECS,
    operating_settings,
    score_observations,
    validate_scorers,
)
from .reference import OperatingReference, checked_artifact

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
    if (canonical[:, 0] == canonical[:, 1]).any() or pd.DataFrame(
        canonical
    ).duplicated().any():
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
        self.reference = OperatingReference(
            config["baseline_run"], config["implementation"]
        )
        source = self.reference.root.resolve()
        if (
            self.root == source
            or self.root.is_relative_to(source)
            or source.is_relative_to(self.root)
        ):
            raise ValueError("Boston output must be separate from baseline outputs")
        self.trees_enabled = config.get("trees", {}).get("enabled", False)
        self.definitions, self.selection = operating_settings(
            self.reference,
            config["scorers"],
            include_trees=self.trees_enabled,
        )
        self.exploration = self._resolve_exploration_config(
            config.get("exploration", {})
        )
        self.cases_path = Path(config["inputs"]["cases_path"])
        self.pairs_path = Path(config["inputs"]["pairs_path"])
        prepared = self.cases_path.parent
        if (
            "data_root" in config["inputs"]
            and self.cases_path == prepared / "cases.parquet"
            and self.pairs_path == prepared / "observed_pairs.parquet"
        ):
            from ..inputs.boston import prepare_boston

            prepare_boston(config["inputs"]["data_root"], prepared)
        if not self.cases_path.exists():
            raise FileNotFoundError(f"Boston cases not found: {self.cases_path}")
        if not self.pairs_path.exists():
            raise FileNotFoundError(f"Boston pairs not found: {self.pairs_path}")
        self.cases, self.pairs = load_boston_inputs(self.cases_path, self.pairs_path)
        self.observations, self.case_index = build_observations(self.cases, self.pairs)
        self.n_cases, self.n_observed_pairs = len(self.cases), len(self.pairs)
        self.n_all_pairs = self.n_cases * (self.n_cases - 1) // 2
        self.tree_paths, self.tree_tools = {}, {}
        self.alignment_length = None
        if self.trees_enabled:
            if not any(d["kind"] == "treecluster" for d in self.definitions.values()):
                raise ValueError(
                    "Reference has no selected raw or dated TreeCluster settings"
                )
            self.alignment_path = Path(config["trees"]["alignment_path"])
            for tool in ("tn93", "fastme", "treetime", "treecluster"):
                executable = (
                    config["trees"].get("tn93_executable", "tn93")
                    if tool == "tn93"
                    else self.reference.config["treecluster"]["executables"][tool]
                )
                self.tree_tools[tool] = command_identity(executable)
            from ..phylogeny.boston import alignment_length

            self.alignment_length = alignment_length(self.alignment_path, self.cases)
        assessment = config.get("assessment", {})
        self.comparator_path = assessment.get("treecluster_path")
        self.comparator = load_treecluster(self.comparator_path, self.cases)
        self.focus_exposures = assessment.get("focus_exposures", ["Conference", "SNF"])
        self.min_cluster_size = assessment.get("min_cluster_size", 2)
        self.scoring_config = {
            key: deepcopy(self.reference.config[key]) for key in ("inference", "scorer")
        }
        logistic_models = {}
        if any(BOSTON_SPECS[name].family == "logistic" for name in config["scorers"]):
            training_id = self.reference.frozen["training_fingerprint"]
            model_dir = self.reference.root / "artifacts/models" / training_id[:20]
            model_manifest = checked_artifact(model_dir)
            if (
                model_manifest["fingerprint"] != training_id
                or model_manifest["seeds"] != self.reference.config["splits"]["train"]
            ):
                raise ValueError(
                    "Reference fitted classifiers differ from frozen training identity"
                )
            logistic_models = read_json(model_dir / "models.json")
            self.models_sha256 = digest_file(model_dir / "models.json")
        else:
            self.models_sha256 = None
        self.context = ScoringContext(
            self.scoring_config, logistic_models=logistic_models
        )
        self.signature = {
            "kind": "boston-empirical-v3",
            "reference": self.reference.identity,
            "boston_cases_sha256": digest_file(self.cases_path),
            "boston_pairs_sha256": digest_file(self.pairs_path),
            "treecluster_sha256": digest_file(self.comparator_path)
            if self.comparator_path
            else None,
            "models_sha256": self.models_sha256,
            "alignment_sha256": digest_file(self.alignment_path)
            if self.trees_enabled
            else None,
            "tree_tools": self.tree_tools,
            "tree_settings": self.reference.config["treecluster"]
            if self.trees_enabled
            else None,
            "implementation": config["implementation"],
            "scorers": {
                name: BOSTON_SPECS[name].metadata() for name in config["scorers"]
            },
            "scoring_config": self.scoring_config,
            "definitions": self.definitions,
            "exploration": self.exploration,
            "n_cases": self.n_cases,
            "n_observed_pairs": self.n_observed_pairs,
            "n_all_pairs": self.n_all_pairs,
            "candidate_universe": "TN93-censored; missing pairs are unobserved, not zero",
            "assessment": assessment,
        }
        self.directory = self.root / "runs" / fingerprint(self.signature)[:20]
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.directory / "reference.json", self.reference.identity)
        write_json(self.directory / "selection.json", self.selection)
        write_json(self.directory / "settings.json", self.definitions)
        write_json(
            self.directory / "inputs.json",
            {
                "cases_path": str(self.cases_path),
                "pairs_path": str(self.pairs_path),
                "n_cases": self.n_cases,
                "n_observed_pairs": self.n_observed_pairs,
                "n_all_pairs": self.n_all_pairs,
                "treecluster_path": str(self.comparator_path)
                if self.comparator_path
                else None,
                "treecluster_sha256": self.signature["treecluster_sha256"],
                "alignment_path": str(self.alignment_path)
                if self.trees_enabled
                else None,
                "alignment_sha256": self.signature["alignment_sha256"],
                "trees_enabled": self.trees_enabled,
            },
        )
        write_json(
            self.root / "current.json",
            {
                "run_directory": str(self.directory),
                "fingerprint": fingerprint(self.signature),
            },
        )

    def score(self):
        signature = {
            "kind": "boston-scores-v2",
            "run": fingerprint(self.signature),
            "scorers": self.config["scorers"],
        }
        score_id = fingerprint(signature)[:20]
        score_dir = self.root / "artifacts/scores" / score_id
        if not valid_artifact(score_dir, signature):
            values = score_observations(
                self.observations, self.context, self.config["scorers"]
            )
            score_frame = pd.concat(
                [
                    self.pairs[["CaseID1", "CaseID2"]].reset_index(drop=True),
                    values.reset_index(drop=True),
                ],
                axis=1,
            )
            score_dir.mkdir(parents=True, exist_ok=True)
            score_frame.to_parquet(score_dir / "scores.parquet", index=False)
            complete_artifact(score_dir, signature, ["scores.parquet"])
        else:
            score_frame = pd.read_parquet(score_dir / "scores.parquet")
        return score_frame, score_id

    def _resolve_exploration_config(self, exploration):
        exploration = deepcopy(exploration or {})
        exploration.setdefault("scorers", list(self.config["scorers"]))
        validate_scorers(exploration["scorers"])
        unknown = set(exploration["scorers"]) - set(self.config["scorers"])
        if unknown:
            raise ValueError(
                f"Boston exploration scorers must be included in scorers: {sorted(unknown)}"
            )

        graph = exploration.get("graph", {})
        if isinstance(graph, bool):
            graph = {"enabled": graph}
        elif graph is None:
            graph = {"enabled": False}
        elif not isinstance(graph, dict):
            raise ValueError("Boston exploration.graph must be a mapping or boolean")
        graph.setdefault("enabled", True)
        graph.setdefault("thresholds", deepcopy(self.reference.config["thresholds"]))
        graph.setdefault("clustering", deepcopy(self.reference.config["clustering"]))

        treecluster = exploration.get("treecluster", {})
        if isinstance(treecluster, bool):
            treecluster = {"enabled": treecluster}
        elif treecluster is None:
            treecluster = {"enabled": False}
        elif not isinstance(treecluster, dict):
            raise ValueError(
                "Boston exploration.treecluster must be a mapping or boolean"
            )
        reference_treecluster = self.reference.config["treecluster"]
        treecluster.setdefault("enabled", self.trees_enabled)
        treecluster.setdefault("methods", list(reference_treecluster["methods"]))
        treecluster.setdefault(
            "genetic_thresholds", list(reference_treecluster["genetic_thresholds"])
        )
        treecluster.setdefault(
            "threshold_days", list(reference_treecluster["threshold_days"])
        )
        return {
            "scorers": exploration["scorers"],
            "graph": graph,
            "treecluster": treecluster,
        }

    def exploration_definitions(self):
        definitions = {}

        def add(definition):
            definitions[fingerprint(definition)[:20]] = definition

        graph = self.exploration["graph"]
        if graph.get("enabled", True):
            clustering = graph["clustering"]
            thresholds = graph["thresholds"]
            for name in self.exploration["scorers"]:
                spec = BOSTON_SPECS[name]
                values = [None, *sorted(set(thresholds[spec.family]))]
                for threshold in values:
                    base = {
                        "score_name": name,
                        "data_process": "empirical",
                        "threshold": threshold,
                        "empty": threshold is None,
                        "baseline_setting_id": None,
                        "baseline_score_name": name,
                        "baseline_data_process": spec.data_process,
                        "exploration": True,
                    }
                    if "components" in clustering["algorithms"]:
                        add(
                            {
                                **base,
                                "kind": "components",
                                "weight_policy": "binary",
                                "pipeline": f"explore/components/{name}",
                            }
                        )
                    if "leiden" in clustering["algorithms"]:
                        leiden_config = clustering["leiden"]
                        for policy in leiden_config["weight_policies"]:
                            if policy == "native" and spec.family == "genetic":
                                continue
                            for resolution in leiden_config["resolutions"]:
                                add(
                                    {
                                        **base,
                                        "kind": "leiden",
                                        "weight_policy": policy,
                                        "objective": leiden_config["objective"],
                                        "resolution": float(resolution),
                                        "restarts": leiden_config["restarts"],
                                        "algorithm_seed": leiden_config["seed"],
                                        "pipeline": f"explore/leiden/{name}/{policy}",
                                    }
                                )

        treecluster = self.exploration["treecluster"]
        if treecluster.get("enabled", self.trees_enabled):
            settings = self.reference.config["treecluster"]
            sequence_length = (
                self.alignment_length
                if self.alignment_length
                else self.reference.config["simulation"]["sequence_length"]
            )
            for kind, thresholds, units in (
                ("raw", treecluster["genetic_thresholds"], "substitutions_per_site"),
                ("dated", treecluster["threshold_days"], "days"),
            ):
                for method in treecluster["methods"]:
                    for threshold in thresholds:
                        threshold_value = (
                            float(threshold) / sequence_length
                            if kind == "raw"
                            else float(threshold)
                        )
                        add(
                            {
                                "kind": "treecluster",
                                "tree_kind": kind,
                                "data_process": "empirical",
                                "method": method,
                                "threshold": threshold_value,
                                "threshold_units": units,
                                "days_per_year": settings["days_per_year"],
                                "pipeline": f"explore/treecluster/empirical/{kind}",
                                "baseline_setting_id": None,
                                "baseline_score_name": "TREE",
                                "baseline_data_process": "empirical",
                                "score_name": "TREE",
                                "exploration": True,
                            }
                        )
        return definitions

    def clusters(self, scores, score_id, definitions=None, directory=None):
        definitions = self.definitions if definitions is None else definitions
        directory = (
            self.directory / "clusters" if directory is None else Path(directory)
        )
        cluster_definitions = {
            k: v
            for k, v in definitions.items()
            if v["kind"] in ("components", "leiden")
        }
        directory.mkdir(parents=True, exist_ok=True)
        all_complete = True
        rows, errors = [], []
        graphs, partitions = {}, {}
        for key, definition in cluster_definitions.items():
            artifact = directory / key
            signature = {
                "run": fingerprint(self.signature),
                "score_id": score_id,
                "definition": definition,
            }
            base = {
                "setting_id": key,
                "pipeline": definition["pipeline"],
                "score_name": definition["score_name"],
                "data_process": "empirical",
                "baseline_setting_id": definition.get("baseline_setting_id"),
                "baseline_score_name": definition.get("baseline_score_name"),
            }
            if valid_artifact(artifact, signature):
                rows.append({**base, **read_json(artifact / "metrics.json")})
                continue
            artifact.mkdir(parents=True, exist_ok=True)
            try:
                spec = BOSTON_SPECS[definition["score_name"]]
                values = scores[spec.name].to_numpy(float)
                requested_graph = (
                    spec.name,
                    definition["threshold"],
                    definition["weight_policy"],
                    definition["empty"],
                )
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
                    memberships = pd.DataFrame(
                        {
                            "case_id": self.cases.case_id,
                            "cluster_id": labels,
                        }
                    )
                    cluster_table = self._summarize_partition(memberships)
                    partitions[partition_id] = cluster_table
                cluster_table = partitions[partition_id]
                memberships = pd.DataFrame(
                    {
                        "case_id": self.cases.case_id,
                        "cluster_id": labels,
                    }
                )
                memberships.to_parquet(artifact / "memberships.parquet", index=False)
                cluster_table.to_parquet(artifact / "clusters.parquet", index=False)
                metrics_dict = {
                    "n_cases": self.n_cases,
                    "n_clusters": len(cluster_table),
                    "size_mean": float(cluster_table["n_cases"].mean()),
                    "size_std": float(cluster_table["n_cases"].std())
                    if len(cluster_table) > 1
                    else 0.0,
                    "largest_cluster": int(cluster_table["n_cases"].max()),
                }
                write_json(artifact / "metrics.json", metrics_dict)
                write_json(artifact / "algorithm.json", metadata)
                complete_artifact(
                    artifact,
                    signature,
                    [
                        "memberships.parquet",
                        "clusters.parquet",
                        "metrics.json",
                        "algorithm.json",
                    ],
                )
                rows.append({**base, **metrics_dict})
            except Exception as exc:
                LOG.error("Cluster failed setting=%s: %s", key, exc)
                error = {**base, "definition": definition, "error": repr(exc)}
                write_json(artifact / "manifest.json", {"status": "failed", **error})
                errors.append(error)
        pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
        write_json(
            directory / "status.json",
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
        clusters = pd.DataFrame(
            {
                "cluster_id": np.arange(len(sizes)),
                "n_cases": sizes,
                "within_pairs": cluster_pairs,
            }
        )
        for col in ["Exposure", "Clade", "Mutation"]:
            if col in self.cases.columns:
                merged = memberships.merge(
                    self.cases[["case_id", col]], on="case_id", validate="many_to_one"
                )
                for val in merged[col].dropna().unique():
                    mask = merged[col] == val
                    positive = np.bincount(
                        merged.loc[mask, "cluster_id"].to_numpy(int),
                        minlength=len(sizes),
                    )
                    clusters[f"n_{col}_{val}"] = positive
        return clusters

    def trees(self, definitions=None, directory=None):
        """Apply the selected baseline raw/dated TreeCluster rules to Boston trees."""
        if not self.trees_enabled:
            raise ValueError(
                "Boston TreeCluster requires trees.enabled and an alignment_path"
            )
        definitions = self.definitions if definitions is None else definitions
        directory = self.directory / "trees" if directory is None else Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        rows, errors = [], []
        settings = self.reference.config["treecluster"]
        if "raw" not in self.tree_paths:
            self.tree_paths["raw"], length = raw_boston_tree(
                self.root,
                self.alignment_path,
                self.cases,
                self.tree_tools,
                settings,
            )
            self.tree_paths["dated"] = dated_boston_tree(
                self.root,
                self.tree_paths["raw"],
                self.cases,
                length,
                self.tree_tools,
                settings,
            )
        write_json(
            directory / "inputs.json",
            {
                kind: {"path": str(path), "sha256": digest_file(path)}
                for kind, path in self.tree_paths.items()
            },
        )
        for key, definition in definitions.items():
            if definition["kind"] != "treecluster":
                continue
            artifact = directory / key
            kind = definition["tree_kind"]
            signature = {
                "run": fingerprint(self.signature),
                "definition": definition,
                "tree_sha256": digest_file(self.tree_paths[kind]),
                "executable": self.tree_tools["treecluster"],
            }
            base = {
                "setting_id": key,
                "pipeline": definition["pipeline"],
                "score_name": "TREE",
                "baseline_setting_id": definition.get("baseline_setting_id"),
                "baseline_data_process": definition.get("baseline_data_process"),
                "tree_kind": kind,
                "method": definition["method"],
                "threshold": definition["threshold"],
                "threshold_units": definition["threshold_units"],
            }
            if valid_artifact(artifact, signature):
                rows.append({**base, **read_json(artifact / "metrics.json")})
                continue
            artifact.mkdir(parents=True, exist_ok=True)
            try:
                threshold = definition["threshold"]
                if kind == "dated":
                    threshold /= definition["days_per_year"]
                labels, metadata = treecluster(
                    self.tree_paths[kind],
                    self.cases,
                    definition["method"],
                    threshold,
                    settings,
                    artifact,
                )
                _, labels = np.unique(labels, return_inverse=True)
                memberships = pd.DataFrame(
                    {"case_id": self.cases.case_id, "cluster_id": labels}
                )
                clusters = self._summarize_partition(memberships)
                memberships.to_parquet(artifact / "memberships.parquet", index=False)
                clusters.to_parquet(artifact / "clusters.parquet", index=False)
                metrics = {
                    "n_cases": self.n_cases,
                    "n_clusters": len(clusters),
                    "size_mean": float(clusters.n_cases.mean()),
                    "size_std": float(clusters.n_cases.std())
                    if len(clusters) > 1
                    else 0.0,
                    "largest_cluster": int(clusters.n_cases.max()),
                }
                write_json(artifact / "metrics.json", metrics)
                write_json(artifact / "algorithm.json", metadata)
                complete_artifact(
                    artifact,
                    signature,
                    [
                        "memberships.parquet",
                        "clusters.parquet",
                        "metrics.json",
                        "algorithm.json",
                        "treecluster.stdout.log",
                        "treecluster.stderr.log",
                    ],
                )
                rows.append({**base, **metrics})
            except Exception as exc:
                LOG.error("TreeCluster failed setting=%s: %s", key, exc)
                error = {**base, "definition": definition, "error": repr(exc)}
                write_json(artifact / "manifest.json", {"status": "failed", **error})
                errors.append(error)
        pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
        write_json(
            directory / "status.json",
            {
                "status": "partial" if errors else "complete",
                "configured": sum(
                    d["kind"] == "treecluster" for d in definitions.values()
                ),
                "completed": len(rows),
                "errors": errors,
            },
        )
        return not errors

    def explore(self):
        """Run descriptive threshold/resolution sweeps separate from frozen transfer."""
        definitions = self.exploration_definitions()
        directory = self.directory / "exploration"
        directory.mkdir(exist_ok=True)
        write_json(directory / "settings.json", definitions)
        metadata = []
        for key, definition in definitions.items():
            metadata.append(
                {
                    "setting_id": key,
                    "kind": definition["kind"],
                    "pipeline": definition["pipeline"],
                    "score_name": definition.get("score_name"),
                    "threshold": definition.get("threshold"),
                    "weight_policy": definition.get("weight_policy"),
                    "resolution": definition.get("resolution"),
                    "tree_kind": definition.get("tree_kind"),
                    "method": definition.get("method"),
                    "threshold_units": definition.get("threshold_units"),
                }
            )
        pd.DataFrame(metadata).to_csv(directory / "setting_metadata.csv", index=False)

        complete = True
        graph_definitions = {
            k: v
            for k, v in definitions.items()
            if v["kind"] in ("components", "leiden")
        }
        if graph_definitions:
            scores, score_id = self.score()
            complete = (
                self.clusters(
                    scores,
                    score_id,
                    graph_definitions,
                    directory / "clusters",
                )
                and complete
            )
        tree_definitions = {
            k: v for k, v in definitions.items() if v["kind"] == "treecluster"
        }
        if tree_definitions:
            complete = self.trees(tree_definitions, directory / "trees") and complete

        assess_partitions(
            directory,
            self.cases,
            definitions,
            self.focus_exposures,
            self.min_cluster_size,
            self.comparator,
            self.n_observed_pairs,
            self.n_all_pairs,
        )
        status = {
            "status": "complete" if complete else "partial",
            "configured": len(definitions),
            "graph_configured": len(graph_definitions),
            "treecluster_configured": len(tree_definitions),
        }
        write_json(directory / "status.json", status)
        return complete

    def run(self, stage="all"):
        manifest = {
            "status": "running",
            "requested_stage": stage,
            "config": self.config,
            "signature": self.signature,
            "git_revision": git_revision(),
            "run_directory": str(self.directory),
        }
        write_json(self.directory / "manifest.json", manifest)
        try:
            if stage not in ("all", "trees", "explore"):
                raise ValueError(
                    "Boston supports all, trees, or explore as computational stages"
                )
            if stage == "all":
                scores, score_id = self.score()
                manifest["score_id"] = score_id
                complete = self.clusters(scores, score_id)
            elif stage == "explore":
                complete = self.explore()
            else:
                complete = True
            if stage != "explore" and self.trees_enabled:
                complete = self.trees() and complete
            if stage != "explore":
                assess_partitions(
                    self.directory,
                    self.cases,
                    self.definitions,
                    self.focus_exposures,
                    self.min_cluster_size,
                    self.comparator,
                    self.n_observed_pairs,
                    self.n_all_pairs,
                )
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
