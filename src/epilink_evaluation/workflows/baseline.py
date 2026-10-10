"""Baseline pipeline: develop, freeze, then replay on held-out realizations."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd

from ..clusterers import components, leiden, treecluster
from ..graphs import build_graph, selected_pairs
from ..inputs.experiment import load_experiment
from ..inputs.synthetic import (
    load_observations,
    load_truth,
)
from ..metrics.pairwise import (
    PairTruth,
    calibration,
    metrics_at_thresholds,
    precision_recall_curve,
)
from ..metrics.partitions import PartitionEvaluator
from ..phylogeny.external import command_identity
from ..phylogeny.trees import prepare_phylogeny
from ..provenance import (
    baseline_signature,
    complete_artifact,
    digest_file,
    fingerprint,
    git_revision,
    implementation_signature,
    read_json,
    valid_artifact,
    write_json,
)
from ..scorers import SCORERS, ScoringContext
from ..scorers.logistic import fit_logistic, training_cells
from ..selection.operating import (
    aggregate_settings,
    endpoint_frontiers,
    select_operating_points,
)
from .settings import settings_registry

LOG = logging.getLogger(__name__)


class Baseline:
    def __init__(self, config):
        self.config = deepcopy(config)
        self.implementation = baseline_signature(implementation_signature())
        self.experiment = load_experiment(config, self.implementation)
        self.diagnostics = self.experiment.require_diagnostics()
        self.tree = self.experiment.tree
        self.truth_directory = self.experiment.truth_directory
        self._evaluation_released = False
        self.tools = {}
        if config["treecluster"]["enabled"]:
            for name, command in config["treecluster"]["executables"].items():
                try:
                    self.tools[name] = command_identity(command)
                except FileNotFoundError as exc:
                    self.tools[name] = {"unavailable": str(exc)}
        scientific = {
            key: value
            for key, value in config.items()
            if key not in ("selection", "output_directory", "config_path")
        }
        self.signature = {
            "schema": 1,
            "config": scientific,
            "implementation": self.implementation,
            "tools": self.tools,
            "truth": self.truth_directory.name,
            "experiment": self.experiment.identity,
        }
        self.root = Path(config["output_directory"])
        self.directory = self.root / "runs" / fingerprint(self.signature)[:20]
        self.directory.mkdir(parents=True, exist_ok=True)
        self.definitions = settings_registry(config)
        candidates = self.directory / "development/pairwise_candidates"
        candidate_signature = self._candidate_signature()
        if candidate_signature and valid_artifact(candidates, candidate_signature):
            self.definitions.update(read_json(candidates / "definitions.json"))
        write_json(self.directory / "settings.json", self.definitions)
        tree_path = Path(self.experiment.config["inputs"]["tree_path"])
        inputs_info = {
            "experiment": self.experiment.identity,
            "diagnostics": self.diagnostics,
            "tree_path": str(tree_path),
            "tree_sha256": digest_file(tree_path),
            "source": read_json(self.experiment.backbone_directory / "source.json"),
            "n_cases": len(self.tree),
            "tree_seed": config["inputs"]["tree_seed"],
            "target_component_size": config["inputs"]["target_component_size"],
        }
        write_json(self.directory / "inputs.json", inputs_info)
        write_json(self.directory / "experiment.json", self.experiment.identity)
        write_json(self.directory / "diagnostics.json", self.diagnostics)
        write_json(
            self.root / "current.json",
            {
                "run_directory": str(self.directory),
                "fingerprint": fingerprint(self.signature),
            },
        )
        self.datasets = {}
        self.context = None
        self.training_id = None

    def dataset(self, seed):
        role = self.experiment._role(seed)
        if role == "evaluation" and not self._evaluation_released:
            raise ValueError(
                "Evaluation observations require frozen selection and explicit held-out release"
            )
        if seed not in self.datasets:
            LOG.info("Prepare observations seed=%s", seed)
            self.datasets[seed] = self.experiment.dataset(
                seed, prepare=role != "development"
            )
        return self.datasets[seed]

    def prepare(self):
        for split in ("train", "development"):
            for seed in self.config["splits"][split]:
                self.dataset(seed)

    def train(self):
        if self.context is not None:
            return
        datasets = [self.dataset(seed) for seed in self.config["splits"]["train"]]
        signature = {
            "kind": "logistic-training-v1",
            "datasets": [path.name for path in datasets],
            "C": self.config["scorer"]["logistic_C"],
            "implementation": self.implementation,
        }
        self.training_id = fingerprint(signature)
        directory = self.root / "artifacts/models" / self.training_id[:20]
        if not valid_artifact(directory, signature):
            LOG.info("Fit logistic models using training realizations only")
            directory.mkdir(parents=True, exist_ok=True)
            cells = {"deterministic": [], "stochastic": []}
            for dataset in datasets:
                observations, _ = load_observations(dataset)
                truth = load_truth(self.truth_directory, observations.pair_id)
                for process in cells:
                    cells[process].append(training_cells(observations, truth, process))
            models = {
                process: fit_logistic(
                    pd.concat(parts, ignore_index=True), signature["C"]
                )
                for process, parts in cells.items()
            }
            write_json(directory / "models.json", models)
            complete_artifact(
                directory,
                signature,
                ["models.json"],
                seeds=self.config["splits"]["train"],
            )
        self.context = ScoringContext(self.config, read_json(directory / "models.json"))

    def scores(self, seed):
        self.train()
        dataset = self.dataset(seed)
        observations, cases = load_observations(dataset)
        signature = {
            "kind": "scores-v1",
            "dataset": dataset.name,
            "training": self.training_id,
            "scorers": [
                SCORERS[name].spec.metadata() for name in self.config["scorers"]
            ],
            "inference": self.config["inference"],
            "scorer_config": self.config["scorer"],
            "implementation": self.implementation,
        }
        directory = self.root / "artifacts/scores" / fingerprint(signature)[:20]
        if not valid_artifact(directory, signature):
            directory.mkdir(parents=True, exist_ok=True)
            scores = pd.DataFrame({"pair_id": observations.pair_id})
            for name in self.config["scorers"]:
                LOG.info("Score seed=%s model=%s", seed, name)
                values = SCORERS[name].predict(observations, self.context)
                if len(values) != len(scores) or not np.all(np.isfinite(values)):
                    raise ValueError(f"Invalid score output: {name}")
                scores[name] = values
            scores.to_parquet(directory / "scores.parquet", index=False)
            complete_artifact(directory, signature, ["scores.parquet"])
        scores = pd.read_parquet(directory / "scores.parquet")
        if not observations.pair_id.equals(scores.pair_id):
            raise ValueError("Cached scores are not pair-aligned")
        truth = load_truth(self.truth_directory, observations.pair_id)
        return observations, cases, truth, scores, directory.name

    def _candidate_signature(self):
        manifests = {
            str(seed): self.directory
            / "development"
            / f"seed_{seed}"
            / "pairwise/evidence/manifest.json"
            for seed in self.config["splits"]["development"]
        }
        if not all(path.exists() for path in manifests.values()):
            return None
        return {
            "kind": "development-pairwise-candidates-v1",
            "run": fingerprint(self.signature),
            "evidence": {seed: digest_file(path) for seed, path in manifests.items()},
        }

    def _development_pairwise_evidence(self, seed):
        """Checkpoint cumulative score evidence independently of candidate cutoffs."""
        _, _, truth_frame, scores, score_id = self.scores(seed)
        directory = (
            self.directory / "development" / f"seed_{seed}" / "pairwise/evidence"
        )
        signature = {
            "kind": "development-pairwise-curves-v1",
            "run": fingerprint(self.signature),
            "score_id": score_id,
            "seed": seed,
        }
        if valid_artifact(directory, signature):
            return directory
        directory.mkdir(parents=True, exist_ok=True)
        truth = PairTruth(truth_frame)
        rankings, curves, budgets, calibrations = [], [], [], []
        for name in self.config["scorers"]:
            spec = SCORERS[name].spec
            values = scores[name].to_numpy(float)
            curve, ranking = precision_recall_curve(values, spec, truth)
            base = {
                "split": "development",
                "seed": seed,
                "score_name": name,
                "data_process": spec.data_process,
            }
            curves.append(curve.assign(**base))
            for fraction in self.config["pairwise"]["selected_fractions"]:
                row = curve.loc[curve.selected_fraction >= fraction].iloc[0].to_dict()
                budgets.append({**base, "requested_fraction": fraction, **row})
            if spec.family == "logistic":
                bins, summary = calibration(values, truth)
                calibrations.append(bins.assign(**base))
                ranking.update(summary)
            rankings.append({**base, **ranking})
        pd.concat(curves, ignore_index=True).to_parquet(
            directory / "precision_recall.parquet", index=False
        )
        pd.DataFrame(rankings).to_csv(directory / "rankings.csv", index=False)
        pd.DataFrame(budgets).to_csv(directory / "budgets.csv", index=False)
        (
            pd.concat(calibrations, ignore_index=True)
            if calibrations
            else pd.DataFrame()
        ).to_csv(directory / "calibration.csv", index=False)
        write_json(
            directory / "empty_metrics.json",
            truth.statistics(np.zeros(truth.n, dtype=bool)),
        )
        complete_artifact(
            directory,
            signature,
            [
                "precision_recall.parquet",
                "rankings.csv",
                "budgets.csv",
                "calibration.csv",
                "empty_metrics.json",
            ],
        )
        return directory

    def _development_pairwise(self):
        sources = {
            seed: self._development_pairwise_evidence(seed)
            for seed in self.config["splits"]["development"]
        }
        candidates = self.directory / "development/pairwise_candidates"
        signature = self._candidate_signature()
        if not valid_artifact(candidates, signature):
            thresholds = {name: set() for name in self.config["scorers"]}
            for source in sources.values():
                curves = pd.read_parquet(
                    source / "precision_recall.parquet",
                    columns=["score_name", "threshold"],
                )
                for name, curve in curves.groupby("score_name"):
                    thresholds[name].update(curve.threshold.tolist())
            definitions = {
                key: value
                for key, value in settings_registry(self.config, thresholds).items()
                if value["kind"] == "pairwise"
            }
            write_json(candidates / "definitions.json", definitions)
            complete_artifact(candidates, signature, ["definitions.json"])
        definitions = read_json(candidates / "definitions.json")
        self.definitions = {**settings_registry(self.config), **definitions}
        write_json(self.directory / "settings.json", self.definitions)
        for seed, source in sources.items():
            directory = source.parent
            stage_signature = {
                "run": fingerprint(self.signature),
                "seed": seed,
                "split": "development",
                "score_id": read_json(source / "manifest.json")["signature"][
                    "score_id"
                ],
                "candidates": digest_file(candidates / "manifest.json"),
                "evidence": digest_file(source / "manifest.json"),
            }
            if valid_artifact(directory, stage_signature):
                continue
            curves = pd.read_parquet(source / "precision_recall.parquet")
            empty = read_json(source / "empty_metrics.json")
            metrics = []
            for name in self.config["scorers"]:
                spec = SCORERS[name].spec
                choices = {
                    key: d for key, d in definitions.items() if d["score_name"] == name
                }
                values = metrics_at_thresholds(
                    curves.loc[curves.score_name == name],
                    [d["threshold"] for d in choices.values()],
                    spec.higher_is_better,
                    empty,
                )
                metrics.append(
                    values.assign(
                        split="development",
                        seed=seed,
                        score_name=name,
                        data_process=spec.data_process,
                        pipeline=f"pairwise/{name}",
                        setting_id=list(choices),
                    )
                )
            pd.concat(metrics, ignore_index=True).to_csv(
                directory / "metrics.csv", index=False
            )
            files = [
                "precision_recall.parquet",
                "rankings.csv",
                "budgets.csv",
                "calibration.csv",
            ]
            for filename in files:
                copyfile(source / filename, directory / filename)
            complete_artifact(directory, stage_signature, ["metrics.csv", *files])
        self.collect("development")

    def pairwise(self, split="development", selected=None):
        if split == "development":
            if selected is not None:
                raise ValueError(
                    "Development pairwise evaluation uses the complete candidate registry"
                )
            return self._development_pairwise()
        definitions = self.definitions if selected is None else selected
        definitions = {
            key: definition
            for key, definition in definitions.items()
            if definition["kind"] == "pairwise"
        }
        for seed in self.config["splits"][split]:
            _, _, truth_frame, scores, score_id = self.scores(seed)
            signature = {
                "run": fingerprint(self.signature),
                "score_id": score_id,
                "split": split,
                "seed": seed,
                "definitions": definitions,
            }
            directory = self.directory / split / f"seed_{seed}" / "pairwise"
            if valid_artifact(directory, signature):
                continue
            directory.mkdir(parents=True, exist_ok=True)
            truth = PairTruth(truth_frame)
            metrics, rankings, curves, budgets, calibrations = [], [], [], [], []
            for name in self.config["scorers"]:
                spec = SCORERS[name].spec
                values = scores[name].to_numpy(float)
                curve, ranking = precision_recall_curve(values, spec, truth)
                base = {
                    "split": split,
                    "seed": seed,
                    "score_name": name,
                    "data_process": spec.data_process,
                }
                if split == "development":
                    curves.append(curve.assign(**base))
                    for fraction in self.config["pairwise"]["selected_fractions"]:
                        row = (
                            curve.loc[curve.selected_fraction >= fraction]
                            .iloc[0]
                            .to_dict()
                        )
                        budgets.append({**base, "requested_fraction": fraction, **row})
                if spec.family == "logistic":
                    bins, calibration_summary = calibration(values, truth)
                    calibrations.append(bins.assign(**base))
                    ranking.update(calibration_summary)
                rankings.append({**base, **ranking})
                for key, definition in definitions.items():
                    if definition["score_name"] != name:
                        continue
                    mask = selected_pairs(
                        values, spec, definition["threshold"], definition["empty"]
                    )
                    metrics.append(
                        {
                            **base,
                            "pipeline": definition["pipeline"],
                            "setting_id": key,
                            **truth.statistics(mask),
                        }
                    )
            pd.DataFrame(metrics).to_csv(directory / "metrics.csv", index=False)
            pd.DataFrame(rankings).to_csv(directory / "rankings.csv", index=False)
            pd.DataFrame(budgets).to_csv(directory / "budgets.csv", index=False)
            (
                pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
            ).to_parquet(directory / "precision_recall.parquet", index=False)
            (
                pd.concat(calibrations, ignore_index=True)
                if calibrations
                else pd.DataFrame()
            ).to_csv(directory / "calibration.csv", index=False)
            complete_artifact(
                directory,
                signature,
                [
                    "metrics.csv",
                    "rankings.csv",
                    "budgets.csv",
                    "precision_recall.parquet",
                    "calibration.csv",
                ],
            )
        self.collect(split)

    def clusters(self, split="development", selected=None):
        definitions = self.definitions if selected is None else selected
        definitions = {
            key: value
            for key, value in definitions.items()
            if value["kind"] != "pairwise"
        }
        all_complete = True
        for seed in self.config["splits"][split]:
            observations, cases, truth, scores, score_id = self.scores(seed)
            evaluator = PartitionEvaluator(
                observations,
                cases,
                truth,
            )
            directory = self.directory / split / f"seed_{seed}" / "clusters"
            directory.mkdir(parents=True, exist_ok=True)
            rows, errors, trees, partitions = [], [], {}, {}
            graph_key, graph = None, None
            for key, definition in definitions.items():
                artifact = directory / key
                signature = {
                    "run": fingerprint(self.signature),
                    "score_id": score_id,
                    "definition": definition,
                    "split": split,
                    "seed": seed,
                }
                base = {
                    "split": split,
                    "seed": seed,
                    "setting_id": key,
                    "pipeline": definition["pipeline"],
                    "data_process": definition["data_process"],
                }
                if valid_artifact(artifact, signature):
                    rows.append({**base, **read_json(artifact / "metrics.json")})
                    continue
                artifact.mkdir(parents=True, exist_ok=True)
                try:
                    if definition["kind"] == "treecluster":
                        process, kind = (
                            definition["data_process"],
                            definition["tree_kind"],
                        )
                        if (process, kind) not in trees:
                            try:
                                obs_dir = self.dataset(seed)
                                trees[process, kind] = prepare_phylogeny(
                                    self.config,
                                    obs_dir,
                                    process,
                                    obs_dir.name,
                                    self.implementation,
                                )
                            except Exception as exc:
                                trees[process, kind] = exc
                        if isinstance(trees[process, kind], Exception):
                            raise trees[process, kind]
                        tree_path = trees[process, kind] / ("raw.nwk" if kind == "raw" else "dated.nwk")
                        threshold = definition["threshold"]
                        labels, metadata = treecluster(
                            tree_path,
                            cases,
                            definition["method"],
                            threshold,
                            self.config["treecluster"],
                            artifact,
                        )
                    else:
                        spec = SCORERS[definition["score_name"]].spec
                        requested_graph = (
                            spec.name,
                            definition["threshold"],
                            definition["weight_policy"],
                        )
                        if requested_graph != graph_key:
                            graph = build_graph(
                                observations,
                                len(cases),
                                scores[spec.name].to_numpy(float),
                                spec,
                                definition["threshold"],
                                definition["weight_policy"],
                                definition["empty"],
                            )
                            graph_key = requested_graph
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
                        partitions[partition_id] = evaluator.evaluate(labels)
                    summary, cluster_table = partitions[partition_id]
                    pd.DataFrame(
                        {"case_id": cases.case_id, "cluster_id": labels}
                    ).to_parquet(artifact / "memberships.parquet", index=False)
                    cluster_table.to_parquet(artifact / "clusters.parquet", index=False)
                    write_json(artifact / "metrics.json", summary)
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
                    rows.append({**base, **summary})
                except Exception as exc:
                    # A failed comparator is visible and resumable, never a silent omission.
                    LOG.error("Partition failed seed=%s setting=%s: %s", seed, key, exc)
                    error = {**base, "definition": definition, "error": repr(exc)}
                    write_json(
                        artifact / "manifest.json", {"status": "failed", **error}
                    )
                    errors.append(error)
            pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
            write_json(
                directory / "status.json",
                {
                    "status": "partial" if errors else "complete",
                    "configured": len(definitions),
                    "completed": len(rows),
                    "errors": errors,
                    "treecluster_enabled": self.config["treecluster"]["enabled"],
                },
            )
            all_complete &= not errors
        self.collect(split)
        return all_complete

    def collect(self, split):
        parent = self.directory / split
        parent.mkdir(parents=True, exist_ok=True)
        tables = []
        for seed in self.config["splits"][split]:
            for stage in ("pairwise", "clusters"):
                path = parent / f"seed_{seed}" / stage / "metrics.csv"
                if path.exists():
                    try:
                        frame = pd.read_csv(path)
                    except pd.errors.EmptyDataError:
                        continue
                    if not frame.empty:
                        tables.append(frame)
        if not tables:
            return pd.DataFrame()
        frame = pd.concat(tables, ignore_index=True)
        frame.to_csv(parent / "metrics.csv", index=False)
        aggregated = aggregate_settings(frame)
        aggregated.to_csv(parent / "summary.csv", index=False)
        endpoint_frontiers(aggregated).to_csv(parent / "frontier.csv", index=False)
        return frame

    def select(self):
        # Complete the declared development grid before freezing comparisons.
        self.pairwise()
        if not self.clusters():
            raise RuntimeError(
                "Development sweep is partial; inspect status.json before freezing operating points"
            )
        evidence = self.collect("development")
        expected = {
            (seed, key)
            for seed in self.config["splits"]["development"]
            for key in self.definitions
        }
        if set(zip(evidence.seed, evidence.setting_id)) != expected:
            raise ValueError(
                "Cannot select from an incomplete development comparison matrix"
            )
        selected = select_operating_points(
            evidence,
            self.definitions,
            self.config["selection"]["criteria"],
            self.config["splits"]["development"],
        )
        frozen = {
            "run_fingerprint": fingerprint(self.signature),
            "training_fingerprint": self.training_id,
            "development_evidence_sha256": digest_file(
                self.directory / "development/metrics.csv"
            ),
            "criteria": self.config["selection"]["criteria"],
            "operating_points": selected,
        }
        self.experiment.assert_selection(frozen)
        path = self.directory / "selection/operating_points.json"
        if path.exists() and read_json(path) != frozen:
            if (self.directory / "evaluation/heldout_access.json").exists():
                raise ValueError(
                    "These held-out seeds have already been accessed. Revised criteria require fresh evaluation seeds."
                )
            # Keep earlier decisions alongside the replacement; evaluation artifacts
            # validate their own exact definitions and are replayed when changed.
            old = read_json(path)
            write_json(
                path.parent / f"operating_points_{fingerprint(old)[:20]}.json", old
            )
        write_json(path, frozen)
        return frozen

    def evaluate(self):
        path = self.directory / "selection/operating_points.json"
        if not path.exists():
            raise ValueError(
                "Run the select stage to freeze development settings before evaluation"
            )
        frozen = read_json(path)
        self.train()
        if (
            frozen["run_fingerprint"] != fingerprint(self.signature)
            or frozen["training_fingerprint"] != self.training_id
        ):
            raise ValueError(
                "Frozen operating settings do not match the experiment/training context"
            )
        if frozen["criteria"] != self.config["selection"]["criteria"]:
            raise ValueError(
                "Selection criteria changed; rerun select using development data"
            )
        if frozen["development_evidence_sha256"] != digest_file(
            self.directory / "development/metrics.csv"
        ):
            raise ValueError("Development evidence changed after freezing")
        selected = {
            point["setting_id"]: point["definition"]
            for point in frozen["operating_points"]
            if point["status"] == "selected"
        }
        if not selected:
            raise ValueError("No operating criterion is feasible")
        if any(
            self.definitions.get(key) != definition
            for key, definition in selected.items()
        ):
            raise ValueError(
                "Frozen settings differ from the validated development candidate registry"
            )
        self.experiment.release_evaluation(frozen)
        self._evaluation_released = True
        write_json(
            self.directory / "evaluation/heldout_access.json",
            {
                "seeds": self.config["splits"]["evaluation"],
                "selection_fingerprint": fingerprint(frozen),
            },
        )
        self.pairwise("evaluation", selected)
        complete = self.clusters("evaluation", selected)
        frame = self.collect("evaluation")
        decisions = pd.DataFrame(
            [
                {"setting_id": point["setting_id"], "criterion": point["criterion"]}
                for point in frozen["operating_points"]
                if point["status"] == "selected"
            ]
        )
        frame.merge(decisions, on="setting_id", validate="many_to_many").to_csv(
            self.directory / "evaluation/operating_results.csv", index=False
        )
        write_json(self.directory / "evaluation/selection_used.json", frozen)
        return complete

    def run(self, stage):
        manifest = {
            "status": "running",
            "requested_stage": stage,
            "signature": self.signature,
            "git_revision": git_revision(),
            "config": self.config,
            "run_directory": str(self.directory),
        }
        write_json(self.directory / "manifest.json", manifest)
        complete = True
        try:
            if stage == "prepare":
                self.prepare()
            elif stage == "pairwise":
                self.pairwise()
            elif stage == "clusters":
                complete = self.clusters()
            elif stage in ("develop", "all"):
                self.pairwise()
                complete = self.clusters()
                if stage == "all" and complete:
                    self.select()
                    complete = self.evaluate()
            elif stage == "select":
                self.select()
            elif stage == "evaluate":
                complete = self.evaluate()
            elif stage != "report":
                raise ValueError(f"Unknown stage: {stage}")
            manifest["status"] = "complete" if complete else "partial"
        except Exception as exc:
            manifest.update(status="failed", error=repr(exc))
            raise
        finally:
            write_json(self.directory / "manifest.json", manifest)
            from ..reporting.report import render_report

            render_report(self.directory)
        LOG.info("Report: %s", self.directory / "report.html")
        return complete
