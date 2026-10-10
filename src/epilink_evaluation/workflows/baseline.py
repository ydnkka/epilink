"""Baseline pipeline: develop, freeze, then replay on held-out realizations."""

from __future__ import annotations

import logging
import subprocess
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
from ..metrics.component_sweep import ComponentSweep
from ..metrics.pairwise import (
    PairTruth,
    calibration,
    metrics_at_thresholds,
    precision_recall_curve,
)
from ..metrics.partitions import PartitionEvaluator
from ..phylogeny.external import command_identity, treecluster_executable
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
from ..selection.search import (
    cutoff_settings,
    next_cutoffs,
    next_resolutions,
    resolution_settings,
)
from .settings import (
    TREE_CUTOFF_FIELDS,
    leiden_definition,
    settings_registry,
    treecluster_definition,
)

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
            try:
                self.tools["treecluster"] = command_identity(
                    treecluster_executable(config["treecluster"])
                )
            except FileNotFoundError as exc:
                self.tools["treecluster"] = {"unavailable": str(exc)}
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
        candidates = self.directory / "development/cutoff_candidates"
        candidate_signature = self._candidate_signature()
        if candidate_signature and valid_artifact(candidates, candidate_signature):
            self.definitions.update(read_json(candidates / "definitions.json"))
        search = self.directory / "development/resolution_search"
        if valid_artifact(search, self._resolution_signature()):
            self.definitions.update(read_json(search / "definitions.json"))
        if config["treecluster"]["enabled"]:
            for kind in TREE_CUTOFF_FIELDS:
                search = self.directory / "development/treecluster_search" / kind
                if valid_artifact(search, self._treecluster_signature(kind)):
                    self.definitions.update(read_json(search / "definitions.json"))
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
                for process, parts in cells.items():
                    parts.append(training_cells(observations, truth, process))
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
        if self.context is None:
            raise RuntimeError("Scoring context was not initialized")
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
            "kind": "development-cutoff-candidates-v1",
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
        rankings, curves, calibrations = [], [], []
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
            if spec.family == "logistic":
                bins, summary = calibration(values, truth)
                calibrations.append(bins.assign(**base))
                ranking.update(summary)
            rankings.append({**base, **ranking})
        pd.concat(curves, ignore_index=True).to_parquet(
            directory / "precision_recall.parquet", index=False
        )
        pd.DataFrame(rankings).to_csv(directory / "rankings.csv", index=False)
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
        candidates = self.directory / "development/cutoff_candidates"
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
                if value["kind"] in {"pairwise", "components"}
            }
            write_json(candidates / "definitions.json", definitions)
            complete_artifact(candidates, signature, ["definitions.json"])
        definitions = read_json(candidates / "definitions.json")
        searches = {
            k: d
            for k, d in self.definitions.items()
            if d["kind"] in {"leiden", "treecluster"}
        }
        self.definitions = {**settings_registry(self.config), **searches, **definitions}
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
                    key: d
                    for key, d in definitions.items()
                    if d["score_name"] == name and d["kind"] == "pairwise"
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
            metrics, rankings, calibrations = [], [], []
            for name in self.config["scorers"]:
                spec = SCORERS[name].spec
                values = scores[name].to_numpy(float)
                _, ranking = precision_recall_curve(values, spec, truth)
                base = {
                    "split": split,
                    "seed": seed,
                    "score_name": name,
                    "data_process": spec.data_process,
                }
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
            pd.DataFrame().to_parquet(
                directory / "precision_recall.parquet", index=False
            )
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
                    "precision_recall.parquet",
                    "calibration.csv",
                ],
            )
        self.collect(split)

    def _component_sweep(
        self,
        directory,
        seed,
        name,
        choices,
        observations,
        cases,
        evaluator,
        values,
        score_id,
    ):
        source = directory / "component_sweeps" / name
        signature = {
            "kind": "development-component-sweep-v1",
            "run": fingerprint(self.signature),
            "seed": seed,
            "score_id": score_id,
            "definitions": choices,
        }
        if not valid_artifact(source, signature):
            source.mkdir(parents=True, exist_ok=True)
            spec = SCORERS[name].spec
            sweep = ComponentSweep(evaluator, values, spec.higher_is_better)
            ordered = sorted(
                choices,
                key=lambda key: (
                    choices[key]["threshold"] is not None,
                    (
                        -choices[key]["threshold"]
                        if spec.higher_is_better
                        else choices[key]["threshold"]
                    )
                    if choices[key]["threshold"] is not None
                    else 0,
                ),
            )
            rows, files, summary, partition_id = [], [], None, None
            for key in ordered:
                changed = sweep.advance(choices[key]["threshold"])
                if changed or summary is None:
                    labels, summary, clusters = sweep.evaluate()
                    partition_id = fingerprint(labels.tolist())[:20]
                    partition = source / "partitions" / partition_id
                    partition.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(
                        {"case_id": cases.case_id, "cluster_id": labels}
                    ).to_parquet(partition / "memberships.parquet", index=False)
                    clusters.to_parquet(partition / "clusters.parquet", index=False)
                    write_json(partition / "metrics.json", summary)
                    files.extend(
                        f"partitions/{partition_id}/{file}"
                        for file in (
                            "memberships.parquet",
                            "clusters.parquet",
                            "metrics.json",
                        )
                    )
                rows.append(
                    {
                        "split": "development",
                        "seed": seed,
                        "score_name": name,
                        "data_process": spec.data_process,
                        "pipeline": f"components/{name}",
                        "setting_id": key,
                        "partition_id": partition_id,
                        "retained_graph_edges": sweep.position,
                        **summary,
                    }
                )
            pd.DataFrame(rows).to_parquet(source / "metrics.parquet", index=False)
            write_json(
                source / "algorithm.json",
                {
                    "algorithm": "incremental connected components",
                    "whole_ties": True,
                    "n_candidates": len(rows),
                    "n_partitions": len(files) // 3,
                    "pair_counts": "each cross-component pair counted at its first merge",
                },
            )
            complete_artifact(
                source, signature, ["metrics.parquet", "algorithm.json", *files]
            )
        return pd.read_parquet(source / "metrics.parquet")

    def _materialize_selected_components(self, points):
        selected = {
            p["setting_id"]: p["definition"]
            for p in points
            if p["status"] == "selected" and p["definition"]["kind"] == "components"
        }
        for seed in self.config["splits"]["development"]:
            directory = self.directory / "development" / f"seed_{seed}" / "clusters"
            for key, definition in selected.items():
                source = directory / "component_sweeps" / definition["score_name"]
                row = (
                    pd.read_parquet(source / "metrics.parquet")
                    .set_index("setting_id")
                    .loc[key]
                )
                state = source / "partitions" / row.partition_id
                artifact = directory / key
                artifact.mkdir(parents=True, exist_ok=True)
                signature = {
                    "run": fingerprint(self.signature),
                    "score_id": read_json(source / "manifest.json")["signature"][
                        "score_id"
                    ],
                    "definition": definition,
                    "split": "development",
                    "seed": seed,
                }
                if valid_artifact(artifact, signature):
                    continue
                for file in ("memberships.parquet", "clusters.parquet", "metrics.json"):
                    copyfile(state / file, artifact / file)
                write_json(
                    artifact / "algorithm.json",
                    {
                        **read_json(source / "algorithm.json"),
                        "retained_graph_edges": int(row.retained_graph_edges),
                        "source": str(source),
                        "partition_id": row.partition_id,
                    },
                )
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

    def _resolution_signature(self, criteria=None):
        return {
            "kind": "development-resolution-search-v1",
            "run": fingerprint(self.signature),
            "settings": self.config["clustering"]["leiden"]["resolutions"],
            "criteria": self.config["selection"]["criteria"]
            if criteria is None
            else criteria,
            "seeds": self.config["splits"]["development"],
        }

    def _adaptive_leiden(self, criteria=None):
        criteria = (
            self.config["selection"]["criteria"] if criteria is None else criteria
        )
        settings = self.config["clustering"]["leiden"]["resolutions"]
        initial = {
            k: d
            for k, d in settings_registry(self.config).items()
            if d["kind"] == "leiden"
        }
        search = self.directory / "development/resolution_search"
        signature = self._resolution_signature(criteria)
        if valid_artifact(search, signature):
            initial.update(read_json(search / "definitions.json"))
        self.definitions.update(initial)
        batch = {k: d for k, d in self.definitions.items() if d["kind"] == "leiden"}
        while batch:
            self.definitions.update(batch)
            write_json(self.directory / "settings.json", self.definitions)
            if not self._clusters("development", batch):
                return False
            evidence = pd.read_csv(
                self.directory / "development/metrics.csv", float_precision="round_trip"
            )
            trials = {
                k: d for k, d in self.definitions.items() if d["kind"] == "leiden"
            }
            values = next_resolutions(
                settings,
                evidence,
                trials,
                criteria,
                self.config["splits"]["development"],
            )
            write_json(search / "definitions.json", trials)
            write_json(
                search / "search.json",
                {
                    "status": "running" if values else "complete",
                    "settings": resolution_settings(settings),
                    "evaluated_resolutions": sorted(
                        {d["resolution"] for d in trials.values()}
                    ),
                    "criteria": criteria,
                    "development_seeds": self.config["splits"]["development"],
                },
            )
            complete_artifact(search, signature, ["definitions.json", "search.json"])
            batch = {}
            for value in values:
                for name in self.config["scorers"]:
                    definition = leiden_definition(self.config, name, value)
                    batch[fingerprint(definition)[:20]] = definition
        return True

    def clusters(self, split="development", selected=None):
        if split != "development" or selected is not None:
            return self._clusters(split, selected)
        if "components" in self.config["clustering"]["algorithms"]:
            self._development_pairwise()
        base = {k: d for k, d in self.definitions.items() if d["kind"] == "components"}
        if base and not self._clusters(split, base):
            return False
        if (
            "leiden" in self.config["clustering"]["algorithms"]
            and not self._adaptive_leiden()
        ):
            return False
        if self.config["treecluster"]["enabled"]:
            return self._adaptive_treecluster()
        return True

    def _treecluster_signature(self, kind):
        return {
            "kind": "development-treecluster-cutoff-search-v1",
            "run": fingerprint(self.signature),
            "tree_kind": kind,
            "settings": self.config["treecluster"][TREE_CUTOFF_FIELDS[kind]],
            "methods": self.config["treecluster"]["methods"],
            "criteria": self.config["selection"]["criteria"],
            "seeds": self.config["splits"]["development"],
        }

    def _adaptive_treecluster(self):
        for kind, field in TREE_CUTOFF_FIELDS.items():
            settings = self.config["treecluster"][field]
            integer = kind == "raw"
            initial = {
                key: d
                for key, d in settings_registry(self.config).items()
                if d["kind"] == "treecluster" and d["tree_kind"] == kind
            }
            search = self.directory / "development/treecluster_search" / kind
            signature = self._treecluster_signature(kind)
            if valid_artifact(search, signature):
                initial.update(read_json(search / "definitions.json"))
            self.definitions.update(initial)
            batch = {
                key: d
                for key, d in self.definitions.items()
                if d["kind"] == "treecluster" and d["tree_kind"] == kind
            }
            while batch:
                self.definitions.update(batch)
                write_json(self.directory / "settings.json", self.definitions)
                if not self._clusters("development", batch):
                    return False
                evidence = pd.read_csv(
                    self.directory / "development/metrics.csv",
                    float_precision="round_trip",
                )
                trials = {
                    key: d
                    for key, d in self.definitions.items()
                    if d["kind"] == "treecluster" and d["tree_kind"] == kind
                }
                values = next_cutoffs(
                    settings,
                    evidence,
                    trials,
                    self.config["selection"]["criteria"],
                    self.config["splits"]["development"],
                    integer=integer,
                )
                write_json(search / "definitions.json", trials)
                write_json(
                    search / "search.json",
                    {
                        "status": "running" if values else "complete",
                        "tree_kind": kind,
                        "settings": cutoff_settings(settings, integer=integer),
                        "evaluated_cutoffs": sorted(
                            {d["threshold_input"] for d in trials.values()}
                        ),
                        "units": "snps" if integer else "days",
                        "integer_domain": integer,
                        "methods": self.config["treecluster"]["methods"],
                        "criteria": self.config["selection"]["criteria"],
                        "development_seeds": self.config["splits"]["development"],
                    },
                )
                complete_artifact(
                    search, signature, ["definitions.json", "search.json"]
                )
                batch = {}
                processes = sorted({d["data_process"] for d in trials.values()})
                for value in values:
                    for process in processes:
                        for method in self.config["treecluster"]["methods"]:
                            definition = treecluster_definition(
                                self.config, process, kind, method, value
                            )
                            batch[fingerprint(definition)[:20]] = definition
        return True

    def _clusters(self, split="development", selected=None):
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
            if split == "development":
                for name in self.config["scorers"]:
                    choices = {
                        k: d
                        for k, d in definitions.items()
                        if d["kind"] == "components" and d["score_name"] == name
                    }
                    if choices:
                        result = self._component_sweep(
                            directory,
                            seed,
                            name,
                            choices,
                            observations,
                            cases,
                            evaluator,
                            scores[name].to_numpy(float),
                            score_id,
                        )
                        rows.extend(
                            result.drop(
                                columns=["partition_id", "retained_graph_edges"]
                            ).to_dict("records")
                        )
            graph_key, graph = None, None
            for key, definition in sorted(
                definitions.items(),
                key=lambda item: (
                    item[1]["pipeline"],
                    str(item[1].get("threshold")),
                    item[1].get("resolution") or 0,
                ),
            ):
                if split == "development" and definition["kind"] == "components":
                    continue
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
                            except (
                                OSError,
                                subprocess.SubprocessError,
                                ValueError,
                            ) as exc:
                                trees[process, kind] = exc
                        if isinstance(trees[process, kind], Exception):
                            raise trees[process, kind]
                        tree_path = trees[process, kind] / (
                            "raw.nwk" if kind == "raw" else "dated.nwk"
                        )
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
                            definition["empty"],
                            definition.get("graph_mode"),
                        )
                        if requested_graph != graph_key:
                            graph = None
                            graph = build_graph(
                                observations,
                                len(cases),
                                scores[spec.name].to_numpy(float),
                                spec,
                                definition["threshold"],
                                definition["empty"],
                                full=definition.get("graph_mode") == "full",
                            )
                            graph_key = requested_graph
                        assert graph is not None
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
                    if labels is None:
                        raise ValueError("Clustering did not return labels")
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
                except (
                    OSError,
                    subprocess.SubprocessError,
                    ValueError,
                ) as exc:
                    # A failed comparator is visible and resumable, never a silent omission.
                    LOG.error("Partition failed seed=%s setting=%s: %s", seed, key, exc)
                    error = {**base, "definition": definition, "error": repr(exc)}
                    write_json(
                        artifact / "manifest.json", {"status": "failed", **error}
                    )
                    errors.append(error)
            frame = pd.DataFrame(rows)
            previous = directory / "metrics.csv"
            if split == "development" and previous.exists():
                try:
                    old = pd.read_csv(previous, float_precision="round_trip")
                except pd.errors.EmptyDataError:
                    old = pd.DataFrame()
                if not old.empty:
                    old = old.loc[
                        ~old.setting_id.isin(definitions)
                        & old.setting_id.isin(self.definitions)
                    ]
                    frame = pd.concat([old, frame], ignore_index=True)
            if not frame.empty:
                frame = frame.sort_values(
                    ["pipeline", "setting_id", "seed"]
                ).reset_index(drop=True)
            frame.to_csv(previous, index=False)
            write_json(
                directory / "status.json",
                {
                    "status": "partial" if errors else "complete",
                    "configured": sum(
                        d["kind"] != "pairwise" for d in self.definitions.values()
                    )
                    if split == "development"
                    else len(definitions),
                    "completed": len(frame),
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
                        frame = pd.read_csv(path, float_precision="round_trip")
                    except pd.errors.EmptyDataError:
                        continue
                    if not frame.empty:
                        tables.append(frame)
        if not tables:
            return pd.DataFrame()
        frame = (
            pd.concat(tables, ignore_index=True)
            .sort_values(["seed", "pipeline", "setting_id"])
            .reset_index(drop=True)
        )
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
        self._materialize_selected_components(selected)
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
