"""Development-only feature diagnostics and known-truth partition controls."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Phylo

from ..clusterers import components, leiden, treecluster
from ..diagnostics.backbone import backbone_diagnostics, backbone_settings
from ..diagnostics.graphs import graph_summary, oracle_graph
from ..diagnostics.observations import observation_diagnostics
from ..diagnostics.trees import transmission_hop_tree
from ..inputs.experiment import prepare_experiment
from ..inputs.synthetic import load_observations, load_truth
from ..metrics.partitions import PartitionEvaluator
from ..phylogeny.external import command_identity
from ..provenance import (
    complete_artifact,
    digest_file,
    fingerprint,
    implementation_signature,
    read_json,
    valid_artifact,
    write_json,
)
from ..schemas import ENDPOINTS

LOG = logging.getLogger(__name__)


def _implementation(full, stage):
    """Only computational dependencies participate in checkpoint identities."""
    if stage == "backbone":
        paths = {
            "workflows/diagnostics.py", "provenance.py",
            "diagnostics/backbone.py", "truth/relationships.py",
        }
        packages = {"numpy", "pandas", "scipy", "networkx", "pyarrow"}
        return {
            "evaluation": {p: full["evaluation"][p] for p in sorted(paths)},
            "versions": {p: full["versions"].get(p) for p in sorted(packages)},
        }
    paths = {
        "workflows/diagnostics.py",
        "provenance.py",
        "schemas.py",
        "inputs/synthetic.py",
        "metrics/pairwise.py",
    }
    packages = {"numpy", "pandas", "pyarrow"}
    if stage == "observations":
        paths.add("diagnostics/observations.py")
    else:
        paths.update({"metrics/partitions.py", "truth/relationships.py"})
        packages.update({"scipy", "networkx"})
        if stage == "graphs":
            paths.update(
                {
                    "diagnostics/graphs.py",
                    "diagnostics/observations.py",
                    "clusterers/graph.py",
                }
            )
            packages.add("igraph")
        else:
            paths.update(
                {
                    "diagnostics/trees.py",
                    "clusterers/treecluster.py",
                    "phylogeny/external.py",
                    "phylogeny/trees.py",
                }
            )
            packages.update({"TreeCluster", "biopython"})
    return {
        "evaluation": {p: full["evaluation"][p] for p in sorted(paths)},
        "versions": {p: full["versions"].get(p) for p in sorted(packages)},
    }


def _aggregate(frame, keys):
    """Descriptive, equal-seed summaries; pairs are not independent replicates."""
    if frame.empty:
        return pd.DataFrame(columns=[*keys, "n_seeds"])
    if frame.duplicated(["seed", *keys]).any():
        raise ValueError("Diagnostics aggregation requires one row per seed/setting")
    columns = [c for c in frame if c not in {"seed", *keys}]
    # JSON nulls may make an entirely undefined metric an object column.
    # Preserve that metric and its zero defined-value count in the output.
    frame = frame.copy()
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise")
    grouped = frame.groupby(keys, dropna=False, sort=True)
    result = grouped[columns].agg(["mean", "min", "max", "count"])
    result.columns = [f"{metric}_{stat}" for metric, stat in result.columns]
    result["n_seeds"] = grouped.seed.nunique()
    return result.reset_index()


class Diagnostics:
    def __init__(self, config):
        self.config = config
        full = implementation_signature()
        self.experiment = prepare_experiment(config, full)
        self.exp = self.experiment
        self.tree = self.exp.tree
        self.truth_directory = Path(self.exp.truth_directory)
        self.seeds = list(self.exp.config["splits"]["development"])
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Diagnostics require distinct development seeds")
        self.settings = config["diagnostics"]
        self.backbone_config = backbone_settings(self.settings.get("backbone"))
        self.implementation = {
            stage: _implementation(full, stage)
            for stage in ("backbone", "observations", "graphs", "trees")
        }
        self.tools = {}
        if self.settings["treecluster"]["enabled"]:
            command = self.settings["treecluster"]["executables"]["treecluster"]
            try:
                self.tools["treecluster"] = command_identity(command)
            except (OSError, ValueError) as exc:
                self.tools["treecluster"] = {
                    "unavailable": str(exc),
                    "command": command,
                }
        self.signature = {
            "kind": "synthetic-diagnostics-v1",
            "experiment": self.exp.identity,
            "backbone": self.exp.signature["backbone"],
            "implementation": self.implementation,
            "settings": {**self.settings, "backbone": self.backbone_config},
            "tools": self.tools,
        }
        self.root = Path(config["output_directory"]).resolve()
        self.directory = self.root / "runs" / fingerprint(self.signature)
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(
            self.root / "current.json",
            {
                "run_directory": str(self.directory),
                "fingerprint": fingerprint(self.signature),
                "experiment": self.exp.identity,
            },
        )
        truth_manifest = read_json(self.truth_directory / "manifest.json")
        self.truth_identity = {
            "fingerprint": truth_manifest["fingerprint"],
            "files": truth_manifest["files"],
        }
        self.datasets = {}

    def prepare(self):
        self.exp.prepare_development()
        self.datasets = {seed: Path(self.exp.dataset(seed)) for seed in self.seeds}
        return True

    def dataset(self, seed):
        if seed not in self.seeds:
            raise ValueError("Diagnostics only access development seeds")
        if seed not in self.datasets:
            self.prepare()
        return self.datasets[seed]

    def _artifact(self, kind, signature, producer):
        directory = self.root / "artifacts" / kind / fingerprint(signature)
        if not valid_artifact(directory, signature):
            directory.mkdir(parents=True, exist_ok=True)
            write_json(
                directory / "manifest.json",
                {
                    "status": "running",
                    "signature": signature,
                },
            )
            try:
                files = producer(directory)
                complete_artifact(directory, signature, files)
            except Exception as exc:
                write_json(
                    directory / "manifest.json",
                    {
                        "status": "failed",
                        "signature": signature,
                        "error": repr(exc),
                    },
                )
                raise
        return directory

    def _stage_signature(self, stage):
        settings = {}
        if stage == "backbone":
            settings = self.backbone_config
        elif stage == "graphs":
            settings = self.settings["leiden"]
        elif stage == "trees":
            settings = self.settings["treecluster"]
        return {
            "kind": f"diagnostics-{stage}-coverage-v1",
            "experiment": self.exp.identity,
            "datasets": {} if stage == "backbone" else {str(s): p.name for s, p in self.datasets.items()},
            "implementation": self.implementation[stage],
            "settings": settings,
            "tools": self.tools if stage == "trees" else {},
        }

    def _save_stage(self, stage, records, errors, tables):
        directory = self.directory / stage
        directory.mkdir(parents=True, exist_ok=True)
        index = {
            "status": "partial" if errors else "complete",
            "records": records,
            "errors": errors,
            "datasets": {} if stage == "backbone" else {str(s): p.name for s, p in self.datasets.items()},
        }
        if stage == "backbone":
            index["scope"] = "one fixed backbone"
        write_json(directory / "index.json", index)
        for name, table in tables.items():
            table.to_csv(directory / name, index=False)
        signature = self._stage_signature(stage)
        if errors:
            write_json(
                directory / "manifest.json",
                {
                    "status": "partial",
                    "signature": signature,
                    "errors": errors,
                },
            )
        else:
            complete_artifact(directory, signature, ["index.json", *tables])
        return not errors

    def _stage_valid(self, stage):
        directory = self.directory / stage
        if not valid_artifact(directory, self._stage_signature(stage)):
            return False
        index = read_json(directory / "index.json")
        for record in index["records"]:
            for field in ("source", "artifact"):
                if field not in record:
                    continue
                artifact = Path(record[field])
                if not (artifact / "manifest.json").exists():
                    return False
                signature = read_json(artifact / "manifest.json").get("signature")
                if not valid_artifact(artifact, signature):
                    return False
        return True

    def backbone(self):
        """Describe topology without generating or accessing observations."""
        if self._stage_valid("backbone"):
            return True
        records, errors, tables = [], [], {}
        signature = {
            "kind": "diagnostic-backbone-v1",
            "backbone": self.exp.signature["backbone"],
            "implementation": self.implementation["backbone"],
            "settings": self.backbone_config,
        }
        try:
            def produce(directory):
                summary, evidence = backbone_diagnostics(self.tree, self.backbone_config)
                write_json(directory / "summary.json", summary)
                for name, table in evidence.items():
                    if name.endswith(".parquet"):
                        table.to_parquet(directory / name, index=False)
                    else:
                        table.to_csv(directory / name, index=False)
                write_json(directory / "provenance.json", {
                    "backbone": signature["backbone"],
                    "case_order": "full truth node order",
                    "offspring": "direct children in the fixed transmission backbone, including zeros",
                    "superspreading_rule": "offspring >= Poisson percentile at the backbone mean; no events if no transmissions",
                    "replication": "one backbone, not observation-seed replicates",
                })
                return ["summary.json", "provenance.json", *evidence]

            artifact = self._artifact("backbone", signature, produce)
            records.append({"backbone": signature["backbone"], "artifact": str(artifact)})
            summary = read_json(artifact / "summary.json")
            tables["summary.csv"] = pd.DataFrame([
                {k: v for k, v in summary.items() if k != "bootstrap"}
            ])
        except Exception as exc:
            errors.append({"error": repr(exc)})
            LOG.error("Backbone diagnostics failed: %s", exc)
        return self._save_stage("backbone", records, errors, tables)

    def observations(self):
        self.prepare()
        if self._stage_valid("observations"):
            return True
        records, errors = [], []
        collected = {name: [] for name in ("summary", "prevalence", "relationships")}
        for seed, dataset in self.datasets.items():
            try:
                signature = {
                    "kind": "diagnostic-feature-cells-v1",
                    "dataset": dataset.name,
                    "truth": self.truth_identity,
                    "implementation": self.implementation["observations"],
                }

                def produce(directory):
                    observations, _ = load_observations(dataset)
                    truth = load_truth(self.truth_directory, observations.pair_id)
                    cells, summary, prevalence, relationships = observation_diagnostics(
                        observations, truth
                    )
                    cells.assign(seed=seed).to_parquet(
                        directory / "cells.parquet", index=False
                    )
                    for name, table in zip(
                        collected, (summary, prevalence, relationships)
                    ):
                        table.assign(seed=seed).to_csv(
                            directory / f"{name}.csv", index=False
                        )
                    return ["cells.parquet", *[f"{name}.csv" for name in collected]]

                artifact = self._artifact("observations", signature, produce)
                records.append(
                    {"seed": seed, "dataset": dataset.name, "artifact": str(artifact)}
                )
                for name in collected:
                    collected[name].append(pd.read_csv(artifact / f"{name}.csv"))
            except Exception as exc:
                errors.append({"seed": seed, "error": repr(exc)})
                LOG.error("Observation diagnostics failed seed=%s: %s", seed, exc)
        keys = {
            "summary": ["process", "feature_set", "endpoint"],
            "prevalence": ["endpoint"],
            "relationships": ["relationship"],
        }
        tables = {}
        for name, parts in collected.items():
            frame = (
                pd.concat(parts, ignore_index=True)
                if parts
                else pd.DataFrame(columns=["seed", *keys[name]])
            )
            tables[f"{name}.csv"] = frame
            tables[f"{name}_aggregate.csv"] = _aggregate(frame, keys[name])
        return self._save_stage("observations", records, errors, tables)

    def _sample(self, dataset):
        ids = sorted(
            pd.read_parquet(
                dataset / "cases.parquet", columns=["case_id"]
            ).case_id.astype(str)
        )
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Sampled cases must be nonempty and unique")
        identity = {
            "truth": self.truth_identity,
            "sampled_cases": fingerprint(ids),
            "n_cases": len(ids),
        }
        return ids, identity

    def _control_data(self, ids):
        # Reconstruct the complete sampled universe from truth indices, never dates/GD/TD.
        nodes = pd.read_parquet(self.truth_directory / "nodes.parquet")
        lookup = dict(zip(nodes.case_id.astype(str), nodes.node_index))
        cases = pd.DataFrame({"case_id": ids, "node_index": [lookup[c] for c in ids]})
        a, b = np.triu_indices(len(cases), k=1)
        low = np.minimum(cases.node_index.to_numpy()[a], cases.node_index.to_numpy()[b])
        high = np.maximum(
            cases.node_index.to_numpy()[a], cases.node_index.to_numpy()[b]
        )
        pairs = pd.DataFrame(
            {
                "pair_id": low * (2 * len(nodes) - low - 1) // 2 + high - low - 1,
                "a": a,
                "b": b,
            }
        )
        truth = load_truth(self.truth_directory, pairs.pair_id)
        evaluator = PartitionEvaluator(pairs, cases, truth)
        return pairs, cases, truth, evaluator

    def _partition(self, kind, signature, cases, evaluator, producer, provenance):
        def produce(directory):
            labels, details = producer(directory)
            _, labels = np.unique(labels, return_inverse=True)
            metrics, clusters = evaluator.evaluate(labels)
            pd.DataFrame({"case_id": cases.case_id, "cluster_id": labels}).to_parquet(
                directory / "memberships.parquet", index=False
            )
            clusters.to_parquet(directory / "clusters.parquet", index=False)
            write_json(directory / "metrics.json", metrics)
            write_json(
                directory / "algorithm.json", {**signature["setting"], **details}
            )
            write_json(directory / "provenance.json", provenance)
            return [
                "memberships.parquet",
                "clusters.parquet",
                "metrics.json",
                "algorithm.json",
                "provenance.json",
            ]

        return self._artifact(kind, signature, produce)

    def _graph_controls(self, ids, identity):
        pairs, cases, truth, evaluator = self._control_data(ids)
        records, errors = [], []
        settings = self.settings["leiden"]
        for endpoint in ENDPOINTS:
            base = {"control_id": fingerprint(identity), "endpoint": endpoint}
            try:
                graph = oracle_graph(pairs, len(cases), truth, endpoint)
                signature = {
                    "kind": "endpoint-oracle-graph-v1",
                    "input": identity,
                    "endpoint": endpoint,
                    "implementation": self.implementation["graphs"],
                }

                def produce(directory):
                    cases.to_parquet(directory / "cases.parquet", index=False)
                    pd.DataFrame(graph.get_edgelist(), columns=["a", "b"]).assign(
                        weight=1
                    ).to_parquet(directory / "edges.parquet", index=False)
                    write_json(directory / "summary.json", graph_summary(graph))
                    write_json(
                        directory / "provenance.json",
                        {
                            **identity,
                            "endpoint": endpoint,
                            "horizon": ENDPOINTS[endpoint],
                            "edge_rule": "finite M <= horizon",
                            "weight": 1,
                            "pair_universe": "all sampled unordered pairs",
                            "case_order": "case_id lexical",
                        },
                    )
                    return [
                        "cases.parquet",
                        "edges.parquet",
                        "summary.json",
                        "provenance.json",
                    ]

                source = self._artifact("graphs", signature, produce)
            except Exception as exc:
                errors.append({**base, "error": repr(exc)})
                continue
            definitions = [{"algorithm": "components", "resolution": None}]
            definitions.extend(
                {
                    "algorithm": "leiden",
                    "resolution": resolution,
                    "objective": settings["objective"],
                    "restarts": settings["restarts"],
                    "seed": settings["seed"],
                    "restart_selection": "maximum objective",
                }
                for resolution in settings["resolution_grid"]
            )
            for setting in definitions:
                try:
                    signature = {
                        "kind": "oracle-graph-partition-v1",
                        "graph": source.name,
                        "setting": setting,
                        "implementation": self.implementation["graphs"],
                    }

                    def partition(_directory):
                        if setting["algorithm"] == "components":
                            return components(graph)
                        return leiden(
                            graph,
                            setting["resolution"],
                            setting["objective"],
                            setting["restarts"],
                            setting["seed"],
                        )

                    artifact = self._partition(
                        "graph_partitions",
                        signature,
                        cases,
                        evaluator,
                        partition,
                        {
                            "graph_directory": str(source),
                            "graph_fingerprint": source.name,
                            "evaluation": "all within-cluster sampled pairs",
                            "reference": "reconstructed transmission tree restricted to sampled cases",
                        },
                    )
                    records.append(
                        {
                            **base,
                            "algorithm": setting["algorithm"],
                            "resolution": setting["resolution"],
                            "source": str(source),
                            "artifact": str(artifact),
                        }
                    )
                except Exception as exc:
                    errors.append({**base, "setting": setting, "error": repr(exc)})
        return records, errors

    def _tree_controls(self, ids, identity):
        pairs, cases, truth, evaluator = self._control_data(ids)
        signature = {
            "kind": "sampled-transmission-hop-tree-v1",
            "input": identity,
            "implementation": self.implementation["trees"],
        }

        def produce(directory):
            hop_tree = transmission_hop_tree(
                self.tree, ids
            )  # Explicitly rejects forests.
            Phylo.write(hop_tree, directory / "transmission_hops.nwk", "newick")
            cases.to_parquet(directory / "cases.parquet", index=False)
            write_json(
                directory / "provenance.json",
                {
                    **identity,
                    "units": "transmission hops",
                    "transmission_edge_length": 1,
                    "sampled_tip_length": 0,
                    "topology": "known transmission tree",
                },
            )
            return ["transmission_hops.nwk", "cases.parquet", "provenance.json"]

        source = self._artifact("hop_trees", signature, produce)
        records, errors = [], []
        settings = self.settings["treecluster"]
        for method in settings["methods"]:
            for threshold in settings["threshold_hops"]:
                setting = {"method": method, "threshold_hops": threshold}
                try:
                    signature = {
                        "kind": "hop-tree-partition-v1",
                        "tree": source.name,
                        "setting": setting,
                        "implementation": self.implementation["trees"],
                        "tools": self.tools,
                        "timeout": settings["command_timeout_seconds"],
                    }

                    def partition(directory):
                        return treecluster(
                            source / "transmission_hops.nwk",
                            cases,
                            method,
                            threshold,
                            settings,
                            directory,
                        )

                    artifact = self._partition(
                        "tree_partitions",
                        signature,
                        cases,
                        evaluator,
                        partition,
                        {
                            "tree_directory": str(source),
                            "tree_fingerprint": source.name,
                            "units": "transmission hops",
                            "evaluation": "all within-cluster sampled pairs",
                            "reference": "reconstructed transmission tree restricted to sampled cases",
                        },
                    )
                    records.append(
                        {
                            "control_id": fingerprint(identity),
                            **setting,
                            "source": str(source),
                            "artifact": str(artifact),
                        }
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "control_id": fingerprint(identity),
                            **setting,
                            "error": repr(exc),
                        }
                    )
        return records, errors

    def _controls(self, stage):
        self.prepare()
        if self._stage_valid(stage):
            return True
        records, errors, controls = [], [], {}
        enabled = stage != "trees" or self.settings["treecluster"]["enabled"]
        if enabled:
            for seed, dataset in self.datasets.items():
                try:
                    ids, identity = self._sample(dataset)
                    key = fingerprint(identity)
                    if key not in controls:
                        producer = (
                            self._graph_controls
                            if stage == "graphs"
                            else self._tree_controls
                        )
                        controls[key] = producer(ids, identity)
                    completed, failed = controls[key]
                    records.extend(
                        {"seed": seed, "dataset": dataset.name, **r} for r in completed
                    )
                    errors.extend({"seed": seed, **r} for r in failed)
                except Exception as exc:
                    errors.append({"seed": seed, "error": repr(exc)})
        for error in errors:
            LOG.error("%s diagnostics failed: %s", stage, error)
        keys = (
            ["endpoint", "algorithm", "resolution"]
            if stage == "graphs"
            else ["method", "threshold_hops"]
        )
        rows = [
            {
                "seed": r["seed"],
                **{k: r[k] for k in keys},
                **read_json(Path(r["artifact"]) / "metrics.json"),
            }
            for r in records
        ]
        frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["seed", *keys])
        tables = {"metrics.csv": frame, "summary.csv": _aggregate(frame, keys)}
        if stage == "graphs":
            sources = {(r["seed"], r["endpoint"]): r["source"] for r in records}
            rows = [
                {
                    "seed": seed,
                    "endpoint": endpoint,
                    **read_json(Path(source) / "summary.json"),
                }
                for (seed, endpoint), source in sources.items()
            ]
            frame = (
                pd.DataFrame(rows)
                if rows
                else pd.DataFrame(columns=["seed", "endpoint"])
            )
            tables.update(
                {
                    "graph_summary.csv": frame,
                    "graph_summary_aggregate.csv": _aggregate(frame, ["endpoint"]),
                }
            )
        return self._save_stage(stage, records, errors, tables)

    def graphs(self):
        return self._controls("graphs")

    def trees(self):
        return self._controls("trees")

    def _coverage(self):
        """Validate all checkpoints and save a checksummed inventory for baseline."""
        stages = ["backbone", "observations", "graphs"]
        if self.settings["treecluster"]["enabled"]:
            stages.append("trees")
        if any(not (self.directory / stage / "index.json").exists() for stage in stages):
            return False
        if not self.datasets:
            # A standalone backbone rerun can validate saved coverage without
            # generating development observations merely to describe a tree.
            try:
                self.datasets = {s: Path(self.exp.dataset(s)) for s in self.seeds}
            except ValueError:
                return False
        datasets = {str(s): p.name for s, p in self.datasets.items()}
        if set(datasets) != set(map(str, self.seeds)):
            return False
        artifacts = set()
        for stage in stages:
            directory = self.directory / stage
            if not valid_artifact(directory, self._stage_signature(stage)):
                return False
            index = read_json(directory / "index.json")
            if index["status"] != "complete":
                return False
            if stage == "backbone":
                if (index["datasets"] or len(index["records"]) != 1
                    or index["records"][0]["backbone"] != self.exp.signature["backbone"]):
                    return False
            elif (index["datasets"] != datasets
                  or {r["seed"] for r in index["records"]} != set(self.seeds)):
                return False
            artifacts.add(directory)
            for record in index["records"]:
                for field in ("artifact", "source"):
                    if field in record:
                        artifacts.add(Path(record[field]))
        inventory = {}
        for directory in sorted(artifacts):
            manifest = read_json(directory / "manifest.json")
            if not valid_artifact(directory, manifest["signature"]):
                return False
            inventory[str(directory)] = {
                "manifest_sha256": digest_file(directory / "manifest.json"),
                "files": manifest["files"],
            }
        signature = {
            "experiment": self.exp.identity,
            "diagnostics": fingerprint(self.signature),
            "datasets": datasets,
        }
        completion = self.directory / "completion"
        coverage = {
            "status": "complete", "stages": stages, "datasets": datasets,
            "artifacts": inventory,
            "aggregation": "one backbone; other summaries use equal development-seed weights; no pair-based CI",
        }
        if not valid_artifact(completion, signature) or read_json(completion / "coverage.json") != coverage:
            write_json(
                completion / "coverage.json",
                coverage,
            )
            complete_artifact(completion, signature, ["coverage.json"])
        marker = {
            "run_directory": str(self.directory),
            "fingerprint": fingerprint(self.signature),
            "experiment": self.exp.identity,
            "datasets": datasets,
            "status": "complete",
        }
        path = Path(self.exp.directory) / "diagnostics.json"
        if not path.exists() or read_json(path) != marker:
            write_json(path, marker)
        return True

    def _invalidate_completion(self):
        path = self.directory / "completion/manifest.json"
        if path.exists():
            write_json(
                path, {"status": "partial", "diagnostics": fingerprint(self.signature)}
            )
        marker = Path(self.exp.directory) / "diagnostics.json"
        if marker.exists() and read_json(marker).get("run_directory") == str(
            self.directory
        ):
            marker.unlink()

    def run(self, stage):
        if stage not in {"prepare", "backbone", "observations", "graphs", "trees", "report", "all"}:
            raise ValueError(f"Unknown diagnostics stage: {stage}")
        if stage == "report":
            # The CLI can call render_report directly to avoid even experiment setup.
            from ..reporting.diagnostics import render_report

            render_report(self.directory)
            return True
        manifest = {
            "status": "running",
            "requested_stage": stage,
            "signature": self.signature,
            "experiment": self.exp.identity,
            "config": self.config,
            "run_directory": str(self.directory),
        }
        write_json(self.directory / "manifest.json", manifest)
        complete = True
        try:
            for requested in (
                ("backbone", "observations", "graphs", "trees") if stage == "all" else (stage,)
            ):
                complete = getattr(self, requested)() and complete
            covered = self._coverage() if complete else False
            if not covered:
                self._invalidate_completion()
            manifest.update(
                status="complete" if complete else "partial", coverage_complete=covered
            )
            if stage == "all" and not covered:
                complete = False
                manifest["status"] = "partial"
        except Exception as exc:
            complete = False
            self._invalidate_completion()
            manifest.update(status="failed", error=repr(exc))
            raise
        finally:
            write_json(self.directory / "manifest.json", manifest)
        from ..reporting.diagnostics import render_report

        render_report(self.directory)
        return complete
