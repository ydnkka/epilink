"""Paired EpiLink clustering sensitivity: inference × clustering adaptation."""

import logging
from copy import deepcopy
from pathlib import Path

import networkx as nx
import pandas as pd

from ..inputs.synthetic import prepare_observations, prepare_truth
from ..provenance import (
    complete_artifact,
    digest_file,
    fingerprint,
    git_revision,
    implementation_signature,
    read_json,
    valid_artifact,
    write_json,
)
from ..scorers import ScoringContext
from ..selection.operating import select_operating_points
from .baseline import Baseline
from .perturbation_config import MODES, scenarios
from .reference import EpiLinkClusteringReference
from .settings import settings_registry

LOG = logging.getLogger(__name__)
METADATA_NUMBERS = {"seed", "value", "baseline_value", "multiplier"}


def read_table(path):
    try:
        return pd.read_csv(path, dtype={"setting_id": str})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def numeric_metrics(frame):
    return [
        name
        for name in frame.select_dtypes(include="number")
        if name not in METADATA_NUMBERS
    ]


def paired_deltas(frame, keys, metrics):
    """Pair by analysis arm, not setting ID: updated settings can change by scenario."""
    if frame.empty:
        return frame.copy()
    metadata = (
        ["setting_id"] if "setting_id" in frame and "setting_id" not in keys else []
    )
    control = frame.loc[frame.scenario == "baseline", [*keys, *metadata, *metrics]]
    control = control.rename(
        columns={name: f"baseline_{name}" for name in [*metadata, *metrics]}
    ).copy()
    paired = (
        frame.loc[frame.scenario != "baseline"]
        .copy()
        .merge(control, on=keys, how="left", validate="many_to_one", indicator=True)
    )
    paired["control_available"] = paired.pop("_merge").eq("both")
    differences = {
        f"delta_{name}": paired[name] - paired[f"baseline_{name}"] for name in metrics
    }
    return pd.concat([paired, pd.DataFrame(differences, index=paired.index)], axis=1)


def summarize(frame, keys, metrics):
    if frame.empty:
        return pd.DataFrame()
    groups = frame.groupby(keys, sort=True, dropna=False)
    summary = groups[metrics].agg(["mean", "std", "min", "max", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_realizations"] = groups.size()
    if "control_available" in frame:
        summary["n_controls"] = groups.control_available.sum()
    return summary.reset_index()


class ClusteringReplay(Baseline):
    """Use shared scoring/partition machinery without fitting or pairwise analysis."""

    def __init__(self, study, scenario, mode):
        self.inference_mode, self.clustering_mode = MODES[mode]
        self.config = study.observation_config(scenario)
        self.config["inference"] = deepcopy(
            scenario["generation"]
            if self.inference_mode == "matched"
            else study.reference.config["inference"]
        )
        self.config["splits"] = {
            "train": [],
            "development": study.config["development_seeds"],
            "evaluation": study.config["seeds"],
        }
        self.config["scorers"] = study.config["scorers"]
        self.config["clustering"]["algorithms"] = ["leiden"]
        self.config["treecluster"]["enabled"] = False
        self.implementation, self.tools = study.implementation, {}
        self.root, self.tree, self.truth_directory = (
            study.root,
            study.tree,
            study.truth_directory,
        )
        self.directory = study.directory / "scenarios" / scenario["name"] / mode
        self.directory.mkdir(parents=True, exist_ok=True)
        self.definitions = deepcopy(study.reference.selected)
        self.points = deepcopy(study.reference.frozen["operating_points"])
        self.criterion = deepcopy(self.points[0]["rule"])
        self.training_id = None
        self.context = ScoringContext(self.config, {})
        self.datasets = {}
        self._evaluation_released = False
        self.signature = {
            "kind": "epilink-clustering-replay-v2",
            "study": fingerprint(study.signature),
            "scenario": scenario,
            "mode": mode,
            "config": self.config,
        }

    def train(self):
        return  # EpiLink scores are training-free.

    def dataset(self, seed):
        role = next(
            (role for role, seeds in self.config["splits"].items() if seed in seeds),
            None,
        )
        if role is None:
            raise ValueError("Perturbation may only access its configured fresh seeds")
        if role == "evaluation" and not self._evaluation_released:
            raise ValueError(
                "Freeze clustering settings before accessing evaluation observations"
            )
        if seed not in self.datasets:
            self.datasets[seed] = prepare_observations(
                self.config, self.tree, self.truth_directory, seed, self.implementation
            )
        return self.datasets[seed]

    def select(self):
        """Reselect resolution within baseline bounds on fresh development."""
        candidates = {
            key: d
            for key, d in settings_registry(self.config).items()
            if d["kind"] == "leiden"
        }
        self.definitions = candidates
        if not candidates or not self._adaptive_leiden([self.criterion]):
            raise ValueError("Updated clustering requires a complete development sweep")
        candidates = self.definitions.copy()
        evidence = self.collect("development")
        expected = {
            (seed, key)
            for seed in self.config["splits"]["development"]
            for key in candidates
        }
        if (
            evidence.duplicated(["seed", "setting_id"]).any()
            or set(zip(evidence.seed, evidence.setting_id)) != expected
        ):
            raise ValueError("Incomplete updated-clustering development matrix")
        self.points = select_operating_points(
            evidence, candidates, [self.criterion], self.config["splits"]["development"]
        )
        if len(self.points) != len(self.config["scorers"]) or any(
            p["status"] != "selected" for p in self.points
        ):
            raise ValueError(
                "No feasible updated clustering setting for every EpiLink pipeline"
            )
        self.definitions = {p["setting_id"]: p["definition"] for p in self.points}

    def run(self):
        manifest = {
            "status": "running",
            "signature": self.signature,
            "config": self.config,
        }
        write_json(self.directory / "manifest.json", manifest)
        try:
            if self.clustering_mode == "updated":
                self.select()
            selection = {
                "run_fingerprint": fingerprint(self.signature),
                "inference_mode": self.inference_mode,
                "clustering_mode": self.clustering_mode,
                "development_seeds": self.config["splits"]["development"]
                if self.clustering_mode == "updated"
                else [],
                "development_evidence_sha256": digest_file(
                    self.directory / "development/metrics.csv"
                )
                if self.clustering_mode == "updated"
                else None,
                "operating_points": self.points,
            }
            write_json(self.directory / "selection.json", selection)
            write_json(self.directory / "settings.json", self.definitions)
            self._evaluation_released = True
            write_json(
                self.directory / "evaluation/heldout_access.json",
                {
                    "seeds": self.config["splits"]["evaluation"],
                    "selection_fingerprint": fingerprint(selection),
                },
            )
            complete = self.clusters("evaluation", self.definitions)
            manifest["status"] = "complete" if complete else "partial"
        except Exception as exc:
            LOG.exception("Clustering replay failed: %s", self.directory)
            manifest.update(status="failed", error=repr(exc))
        finally:
            write_json(self.directory / "manifest.json", manifest)
        return manifest


class PerturbationStudy:
    def __init__(self, config):
        self.config = deepcopy(config)
        self.implementation = implementation_signature()
        self.reference = EpiLinkClusteringReference(
            config["baseline_run"],
            self.implementation,
            config["scorers"],
            config["criterion"],
        )
        used_seeds = {
            seed for seeds in self.reference.config["splits"].values() for seed in seeds
        }
        study_seeds = [*config["development_seeds"], *config["seeds"]]
        if len(set(study_seeds)) != len(study_seeds) or used_seeds.intersection(
            study_seeds
        ):
            raise ValueError(
                "Perturbation development/evaluation seeds must be distinct "
                "and fresh relative to every baseline split"
            )
        if (
            self.reference.config["inputs"].get("smoke_cases")
            and not config["smoke_mode"]
        ):
            raise ValueError(
                "A full perturbation study requires a full baseline reference"
            )
        self.scenarios = scenarios(config, self.reference.config["generation"])
        self.root = Path(config["output_directory"]).resolve()
        source = self.reference.root.resolve()
        if (
            self.root == source
            or self.root.is_relative_to(source)
            or source.is_relative_to(self.root)
        ):
            raise ValueError(
                "Perturbation output must be separate from baseline outputs"
            )
        self.tree = self.reference.tree.copy()
        if config["case_limit"] is not None:
            keep = list(nx.topological_sort(self.tree))[: config["case_limit"]]
            self.tree = self.tree.subgraph(keep).copy()
        backbone_signature = {
            "kind": "frozen-backbone-v1",
            "reference_truth": self.reference.identity["truth_fingerprint"],
            "nodes": list(self.tree),
            "edges": list(self.tree.edges()),
        }
        backbone = (
            self.root / "artifacts/backbones" / fingerprint(backbone_signature)[:20]
        )
        self.backbone_path = backbone / "transmission_tree.gml"
        if not valid_artifact(backbone, backbone_signature):
            backbone.mkdir(parents=True, exist_ok=True)
            nx.write_gml(self.tree, self.backbone_path)
            complete_artifact(backbone, backbone_signature, [self.backbone_path.name])
        self.truth_directory = prepare_truth(
            {
                "output_directory": str(self.root),
                "inputs": {"tree_path": str(self.backbone_path)},
            },
            self.tree,
            self.implementation,
        )
        self.signature = {
            "schema": 2,
            "kind": "epilink-clustering-perturbation-v2",
            "config": {
                k: v
                for k, v in config.items()
                if k not in ("config_path", "output_directory", "baseline_run")
            },
            "reference": self.reference.identity,
            "scenarios": self.scenarios,
            "implementation": self.implementation,
            "truth": self.truth_directory.name,
        }
        self.directory = self.root / "runs" / fingerprint(self.signature)[:20]
        self.directory.mkdir(parents=True, exist_ok=True)
        for name, value in (
            ("reference", self.reference.identity),
            ("selection", self.reference.frozen),
            ("settings", self.reference.selected),
            ("scenarios", self.scenarios),
        ):
            write_json(self.directory / f"{name}.json", value)
        write_json(
            self.root / "current.json",
            {
                "run_directory": str(self.directory),
                "fingerprint": fingerprint(self.signature),
            },
        )

    def observation_config(self, scenario):
        config = deepcopy(self.reference.config)
        config["generation"] = deepcopy(scenario["generation"])
        config["output_directory"] = str(self.root)
        config["inputs"]["tree_path"] = str(self.backbone_path)
        config["inputs"]["smoke_cases"] = self.config["case_limit"]
        return config

    def collect(self):
        records, coverage = [], []
        expected = len(self.config["scorers"]) * len(self.config["seeds"])
        for scenario in self.scenarios:
            for mode in self.config["modes"]:
                directory = self.directory / "scenarios" / scenario["name"] / mode
                path = directory / "manifest.json"
                saved = read_json(path) if path.exists() else {"status": "not_run"}
                inference, clustering = MODES[mode]
                base = {
                    "scenario": scenario["name"],
                    "mode": mode,
                    "inference_mode": inference,
                    "clustering_mode": clustering,
                }
                frame = (
                    read_table(directory / "evaluation/metrics.csv")
                    if saved["status"] in ("complete", "partial")
                    else pd.DataFrame()
                )
                coverage.append(
                    {
                        **base,
                        "status": saved["status"],
                        "completed": len(frame),
                        "expected": expected,
                        "error": saved.get("error"),
                    }
                )
                if frame.empty:
                    continue
                selection = read_json(directory / "selection.json")
                decisions = pd.DataFrame(
                    [
                        {"setting_id": p["setting_id"], "criterion": p["criterion"]}
                        for p in selection["operating_points"]
                        if p["status"] == "selected"
                    ]
                )
                metadata = {
                    **base,
                    **{
                        k: scenario[k]
                        for k in ("parameter", "value", "baseline_value", "multiplier")
                    },
                }
                records.append(
                    frame.merge(
                        decisions, on="setting_id", validate="many_to_one"
                    ).assign(**metadata)
                )
        coverage = pd.DataFrame(coverage)
        coverage.to_csv(self.directory / "coverage.csv", index=False)
        frame = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
        frame.to_csv(self.directory / "results.csv", index=False)
        metrics = numeric_metrics(frame)
        group = [
            "scenario",
            "mode",
            "inference_mode",
            "clustering_mode",
            "criterion",
            "pipeline",
            "setting_id",
        ]
        summarize(frame, group, metrics).to_csv(
            self.directory / "results_summary.csv", index=False
        )
        paired = paired_deltas(
            frame, ["mode", "seed", "criterion", "pipeline"], metrics
        )
        paired.to_csv(self.directory / "results_deltas.csv", index=False)
        summarize(paired, group, [f"delta_{m}" for m in metrics]).to_csv(
            self.directory / "results_delta_summary.csv", index=False
        )
        return bool(
            coverage.status.eq("complete").all()
            and coverage.completed.eq(coverage.expected).all()
        )

    def run(self, stage="all"):
        if stage != "all":
            raise ValueError(f"Unknown perturbation stage: {stage}")
        manifest = {
            "status": "running",
            "requested_stage": stage,
            "config": self.config,
            "signature": self.signature,
            "git_revision": git_revision(),
            "n_cases": len(self.tree),
            "run_directory": str(self.directory),
        }
        write_json(self.directory / "manifest.json", manifest)
        try:
            for scenario in self.scenarios:
                for mode in self.config["modes"]:
                    LOG.info(
                        "EpiLink clustering scenario=%s mode=%s", scenario["name"], mode
                    )
                    ClusteringReplay(self, scenario, mode).run()
                    self.collect()
            manifest["status"] = "complete" if self.collect() else "partial"
        except Exception as exc:
            manifest.update(status="failed", error=repr(exc))
            raise
        finally:
            write_json(self.directory / "manifest.json", manifest)
            from ..reporting.perturbation import render_report

            render_report(self.directory)
        LOG.info("Perturbation report: %s", self.directory / "report.html")
        return manifest["status"] == "complete"
