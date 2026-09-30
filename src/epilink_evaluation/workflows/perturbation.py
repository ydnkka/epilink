"""Replay frozen baseline models/settings on paired parameter perturbations."""

from copy import deepcopy
import logging
from pathlib import Path

import networkx as nx
import pandas as pd

from ..inputs.synthetic import prepare_truth
from ..provenance import (
    complete_artifact, fingerprint, git_revision, implementation_signature,
    read_json, valid_artifact, write_json,
)
from ..scorers import ScoringContext
from .baseline import Baseline
from .perturbation_config import scenarios
from .reference import BaselineReference

LOG = logging.getLogger(__name__)
METADATA_NUMBERS = {"seed", "value", "baseline_value", "multiplier"}


def read_table(path):
    try:
        return pd.read_csv(path, dtype={"setting_id": str})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def numeric_metrics(frame):
    return [name for name in frame.select_dtypes(include="number")
            if name not in METADATA_NUMBERS]


def paired_deltas(frame, keys, metrics):
    """Keep missing controls visible; differences are perturbed minus baseline."""
    if frame.empty:
        return frame.copy()
    control = frame.loc[frame.scenario == "baseline", [*keys, *metrics]]
    control = control.rename(columns={name: f"baseline_{name}" for name in metrics}).copy()
    paired = frame.loc[frame.scenario != "baseline"].copy().merge(
        control, on=keys, how="left", validate="many_to_one", indicator=True
    )
    paired["control_available"] = paired.pop("_merge").eq("both")
    differences = {f"delta_{name}": paired[name] - paired[f"baseline_{name}"] for name in metrics}
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


class FrozenReplay(Baseline):
    """Reuse baseline evaluation machinery without baseline init, fitting or selection."""

    def __init__(self, study, scenario, mode):
        self.config = deepcopy(study.reference.config)
        self.config["generation"] = deepcopy(scenario["generation"])
        self.config["inference"] = deepcopy(
            scenario["generation"] if mode == "matched" else study.reference.config["inference"]
        )
        self.config["splits"]["evaluation"] = study.config["seeds"]
        self.config["output_directory"] = str(study.root)
        self.config["inputs"]["tree_path"] = str(study.backbone_path)
        self.config["inputs"]["smoke_cases"] = study.config["case_limit"]
        self.implementation, self.tools = study.implementation, study.reference.tools
        self.root, self.tree, self.truth_directory = study.root, study.tree, study.truth_directory
        self.directory = study.directory / "scenarios" / scenario["name"] / mode
        self.directory.mkdir(parents=True, exist_ok=True)
        self.definitions = deepcopy(study.reference.selected)
        self.training_id = study.reference.training_id
        self.context = ScoringContext(self.config, deepcopy(study.reference.models))
        self.datasets = {}
        self.signature = {
            "kind": "frozen-perturbation-replay-v1", "study": fingerprint(study.signature),
            "scenario": scenario, "mode": mode, "config": self.config,
        }

    def train(self):
        """Models were loaded from the reference; this path never fits a classifier."""
        return

    def select(self):
        raise ValueError("Perturbation replays use frozen settings; retuning is a separate study")

    def run(self):
        manifest = {"status": "running", "signature": self.signature, "config": self.config}
        write_json(self.directory / "manifest.json", manifest)
        try:
            self.pairwise("evaluation", self.definitions)
            complete = self.clusters("evaluation", self.definitions)
            manifest["status"] = "complete" if complete else "partial"
        except Exception as exc:
            LOG.exception("Replay failed: %s", self.directory)
            manifest.update(status="failed", error=repr(exc))
        finally:
            write_json(self.directory / "manifest.json", manifest)
        return manifest


class PerturbationStudy:
    def __init__(self, config):
        self.config = deepcopy(config)
        self.implementation = implementation_signature()
        self.reference = BaselineReference(config["baseline_run"], self.implementation)
        used_seeds = {seed for seeds in self.reference.config["splits"].values() for seed in seeds}
        if used_seeds.intersection(config["seeds"]):
            raise ValueError("Perturbation seeds must be fresh relative to every baseline split")
        if self.reference.config["inputs"].get("smoke_cases") and not config["smoke_mode"]:
            raise ValueError("A full perturbation study requires a full baseline reference")
        self.scenarios = scenarios(config, self.reference.config["generation"])
        self.root = Path(config["output_directory"]).resolve()
        source = self.reference.root.resolve()
        if self.root == source or self.root.is_relative_to(source) or source.is_relative_to(self.root):
            raise ValueError("Perturbation output must be separate from baseline outputs")
        self.tree = self.reference.tree.copy()
        if config["case_limit"] is not None:
            keep = list(nx.topological_sort(self.tree))[:config["case_limit"]]
            self.tree = self.tree.subgraph(keep).copy()
        backbone_signature = {
            "kind": "frozen-backbone-v1", "reference_truth": self.reference.identity["truth_fingerprint"],
            "nodes": list(self.tree), "edges": list(self.tree.edges()),
        }
        backbone = self.root / "artifacts/backbones" / fingerprint(backbone_signature)[:20]
        self.backbone_path = backbone / "transmission_tree.gml"
        if not valid_artifact(backbone, backbone_signature):
            backbone.mkdir(parents=True, exist_ok=True)
            nx.write_gml(self.tree, self.backbone_path)
            complete_artifact(backbone, backbone_signature, [self.backbone_path.name])
        truth_config = {"output_directory": str(self.root), "inputs": {"tree_path": str(self.backbone_path)}}
        self.truth_directory = prepare_truth(truth_config, self.tree, self.implementation)
        self.reference.copy_models(self.root)
        self.signature = {
            "schema": 1, "kind": "perturbation-study-v1",
            "config": {k: v for k, v in config.items()
                       if k not in ("config_path", "output_directory", "baseline_run")},
            "reference": self.reference.identity, "scenarios": self.scenarios,
            "implementation": self.implementation, "tools": self.reference.tools,
            "truth": self.truth_directory.name,
        }
        self.directory = self.root / "runs" / fingerprint(self.signature)[:20]
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.directory / "reference.json", self.reference.identity)
        write_json(self.directory / "selection.json", self.reference.frozen)
        write_json(self.directory / "settings.json", self.reference.selected)
        write_json(self.directory / "scenarios.json", self.scenarios)
        write_json(self.root / "current.json", {
            "run_directory": str(self.directory), "fingerprint": fingerprint(self.signature),
        })

    def collect(self):
        records, rankings, coverage = [], [], []
        decisions = pd.DataFrame([
            {"setting_id": p["setting_id"], "criterion": p["criterion"]}
            for p in self.reference.frozen["operating_points"] if p["status"] == "selected"
        ])
        expected = len(self.reference.selected) * len(self.config["seeds"])
        for scenario in self.scenarios:
            for mode in self.config["modes"]:
                directory = self.directory / "scenarios" / scenario["name"] / mode
                path = directory / "manifest.json"
                saved = read_json(path) if path.exists() else {"status": "not_run"}
                base = {"scenario": scenario["name"], "mode": mode}
                frame = (read_table(directory / "evaluation/metrics.csv")
                         if saved["status"] in ("complete", "partial") else pd.DataFrame())
                coverage.append({**base, "status": saved["status"], "completed": len(frame),
                                 "expected": expected, "error": saved.get("error")})
                if frame.empty:
                    continue
                metadata = {**base, **{k: scenario[k] for k in ("parameter", "value", "baseline_value", "multiplier")}}
                records.append(frame.merge(decisions, on="setting_id", validate="many_to_many").assign(**metadata))
                for seed in self.config["seeds"]:
                    ranking = read_table(directory / "evaluation" / f"seed_{seed}" / "pairwise/rankings.csv")
                    if not ranking.empty:
                        rankings.append(ranking.assign(**metadata))
        coverage = pd.DataFrame(coverage)
        coverage.to_csv(self.directory / "coverage.csv", index=False)
        for name, frames, keys in (
            ("results", records, ["criterion", "pipeline", "setting_id"]),
            ("rankings", rankings, ["score_name", "data_process", "score_family"]),
        ):
            frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            frame.to_csv(self.directory / f"{name}.csv", index=False)
            metrics = numeric_metrics(frame)
            group = ["scenario", "mode", *keys]
            summary = summarize(frame, group, metrics)
            summary.to_csv(self.directory / f"{name}_summary.csv", index=False)
            paired = paired_deltas(frame, ["mode", "seed", *keys], metrics)
            paired.to_csv(self.directory / f"{name}_deltas.csv", index=False)
            summarize(paired, group, [f"delta_{m}" for m in metrics]).to_csv(
                self.directory / f"{name}_delta_summary.csv", index=False
            )
        return bool(coverage.status.eq("complete").all() and coverage.completed.eq(coverage.expected).all())

    def run(self):
        manifest = {"status": "running", "config": self.config, "signature": self.signature,
                    "git_revision": git_revision(), "n_cases": len(self.tree),
                    "run_directory": str(self.directory)}
        write_json(self.directory / "manifest.json", manifest)
        try:
            for scenario in self.scenarios:
                for mode in self.config["modes"]:
                    LOG.info("Perturbation scenario=%s mode=%s", scenario["name"], mode)
                    FrozenReplay(self, scenario, mode).run()
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
