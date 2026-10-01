"""Shared synthetic experiments: immutable inputs and explicitly released holdouts."""

from copy import deepcopy
from pathlib import Path

import networkx as nx

from ..config import validate_experiment
from ..provenance import (
    complete_artifact,
    digest_file,
    fingerprint,
    generation_signature,
    implementation_signature,
    read_json,
    valid_artifact,
    write_json,
)
from .synthetic import load_backbone, prepare_observations, prepare_truth


def specification(config):
    """Data design, excluding method settings, study paths and presentation."""
    return {
        k: deepcopy(config[k]) for k in ("inputs", "generation", "simulation", "splits")
    }


def checked(directory, expected=None):
    directory = Path(directory)
    path = directory / "manifest.json"
    if not path.exists():
        raise ValueError(
            f"Missing prepared artifact: {directory}; run diagnostics preparation"
        )
    saved = read_json(path)
    if not valid_artifact(
        directory, saved.get("signature") if expected is None else expected
    ):
        raise ValueError(
            f"Incomplete or changed prepared artifact: {directory}; rerun preparation"
        )
    return saved


class SyntheticExperiment:
    def __init__(self, directory, implementation=None):
        self.directory = Path(directory).resolve()
        self.root = self.directory.parent.parent
        self.manifest = checked(self.directory)
        self.signature = self.manifest["signature"]
        if self.directory.name != fingerprint(self.signature)[:20]:
            raise ValueError("Synthetic experiment identity differs from its directory")
        self.identity = {
            "experiment_directory": str(self.directory),
            "fingerprint": self.manifest["fingerprint"],
        }
        self.config = read_json(self.directory / "experiment.json")
        self.implementation = implementation or implementation_signature()
        self.truth_directory = self.root / "artifacts/truth" / self.signature["truth"]
        truth = checked(self.truth_directory)
        self.backbone_directory = (
            self.root / "artifacts/backbones" / self.signature["backbone"]
        )
        checked(self.backbone_directory)
        self.tree = nx.DiGraph()
        self.tree.add_nodes_from(truth["signature"]["nodes"])
        self.tree.add_edges_from(truth["signature"]["edges"])
        self._selection = None

    def _role(self, seed):
        roles = [role for role, seeds in self.config["splits"].items() if seed in seeds]
        if len(roles) != 1:
            raise ValueError(
                "Seed must belong to exactly one configured experiment split"
            )
        return roles[0]

    def _access_path(self, seed):
        # Root-level records survive replacing study runs and changing experiments.
        return self.root / "heldout_access" / f"seed_{seed}.json"

    def assert_selection(self, frozen):
        if self.config["inputs"].get("smoke_cases") is not None:
            # Pipeline validation is repeatable across code revisions; it is not
            # held-out scientific evidence. Its accesses are recorded separately.
            return
        identity = fingerprint(frozen)
        for seed in self.config["splits"]["evaluation"]:
            path = self._access_path(seed)
            if path.exists() and read_json(path)["selection_fingerprint"] != identity:
                raise ValueError(
                    "Held-out seeds already accessed; revised analyses require fresh evaluation seeds"
                )

    def release_evaluation(self, frozen):
        self.assert_selection(frozen)
        identity = fingerprint(frozen)
        for seed in self.config["splits"]["evaluation"]:
            path = (
                self.root / "validation_access" / identity / f"seed_{seed}.json"
                if self.config["inputs"].get("smoke_cases") is not None
                else self._access_path(seed)
            )
            if not path.exists():
                write_json(
                    path,
                    {
                        "seed": seed,
                        "experiment": self.identity,
                        "selection_fingerprint": identity,
                    },
                )
        self._selection = identity

    def dataset(self, seed, prepare=False):
        role = self._role(seed)
        if role == "evaluation":
            if self._selection is None:
                raise ValueError(
                    "Evaluation observations require frozen selection and explicit held-out release"
                )
        elif self._access_path(seed).exists():
            raise ValueError(
                "Previously accessed held-out seeds cannot become training/development data; use fresh evaluation seeds"
            )
        link = self.directory / "observations" / f"seed_{seed}.json"
        if link.exists():
            record = read_json(link)
            directory = self.root / "artifacts/observations" / record["dataset"]
            if record["role"] != role:
                raise ValueError("Observation split differs from the pinned experiment")
            try:
                saved = checked(directory)
                if (
                    saved["fingerprint"] != record["fingerprint"]
                    or saved["signature"]["truth"] != self.truth_directory.name
                    or saved["signature"]["seed"] != seed
                    or saved["signature"]["generation"] != self.config["generation"]
                    or saved["signature"]["simulation"] != self.config["simulation"]
                    or saved["signature"]["implementation"] != self.signature["producer"]
                ):
                    raise ValueError(
                        "Observation provenance differs from the experiment"
                    )
                return directory
            except ValueError:
                if not prepare:
                    raise
        elif not prepare:
            raise ValueError(
                f"Development seed {seed} is not prepared; run diagnostics first"
            )
        if generation_signature(self.implementation) != self.signature["producer"]:
            raise ValueError(
                "Generation implementation changed; prepare a new synthetic experiment"
            )
        directory = prepare_observations(
            self.config, self.tree, self.truth_directory, seed, self.implementation
        )
        saved = checked(directory)
        write_json(
            link,
            {
                "seed": seed,
                "role": role,
                "dataset": directory.name,
                "fingerprint": saved["fingerprint"],
            },
        )
        return directory

    def prepare_development(self):
        for seed in self.config["splits"]["development"]:
            self.dataset(seed, prepare=True)

    def require_diagnostics(self):
        marker_path = self.directory / "diagnostics.json"
        if not marker_path.exists():
            raise ValueError("Run diagnostics --stage all before baseline development")
        marker = read_json(marker_path)
        datasets = {
            str(s): self.dataset(s).name for s in self.config["splits"]["development"]
        }
        if (
            marker.get("status") != "complete"
            or marker["experiment"] != self.identity
            or marker["datasets"] != datasets
        ):
            raise ValueError(
                "Completed diagnostics do not match this experiment's development observations"
            )
        directory = Path(marker["run_directory"])
        signature = {
            "experiment": self.identity,
            "diagnostics": marker["fingerprint"],
            "datasets": datasets,
        }
        checked(directory / "completion", signature)
        coverage = read_json(directory / "completion/coverage.json")
        if coverage["status"] != "complete" or coverage["datasets"] != datasets:
            raise ValueError("Diagnostic coverage does not match the experiment")
        for path, inventory in coverage["artifacts"].items():
            path = Path(path)
            manifest = checked(path)
            if (
                digest_file(path / "manifest.json") != inventory["manifest_sha256"]
                or manifest["files"] != inventory["files"]
            ):
                raise ValueError(f"Diagnostic evidence changed: {path}")
        return marker


def prepare_experiment(config, implementation=None):
    validate_experiment(config)
    implementation = implementation or implementation_signature()
    if not config.get("experiment_root"):
        raise ValueError(
            "Synthetic preparation requires experiment_root from a shared experiment config"
        )
    root = Path(config["experiment_root"]).resolve()
    tree = load_backbone(config)
    source_path = Path(config["inputs"]["tree_path"])
    backbone_signature = {
        "kind": "synthetic-backbone-v1",
        "source_sha256": digest_file(source_path),
        "nodes": list(tree),
        "edges": list(tree.edges()),
    }
    backbone = root / "artifacts/backbones" / fingerprint(backbone_signature)[:20]
    if not valid_artifact(backbone, backbone_signature):
        backbone.mkdir(parents=True, exist_ok=True)
        nx.write_gml(tree, backbone / "transmission_tree.gml")
        source = Path(
            config["inputs"].get("tree_source_path")
            or source_path.with_suffix(".source.json")
        )
        write_json(
            backbone / "source.json",
            {
                "tree_path": str(source_path),
                "tree_sha256": digest_file(source_path),
                "provenance": read_json(source) if source.exists() else None,
            },
        )
        complete_artifact(
            backbone, backbone_signature, ["transmission_tree.gml", "source.json"]
        )
    resolved = specification(config)
    resolved.update(schema_version=1, output_directory=str(root))
    resolved["inputs"]["tree_path"] = str(backbone / "transmission_tree.gml")
    truth = prepare_truth(resolved, tree, implementation)
    signature = {
        "kind": "synthetic-experiment-v1",
        "specification": specification(config),
        "producer": generation_signature(implementation),
        "backbone": backbone.name,
        "truth": truth.name,
    }
    directory = root / "experiments" / fingerprint(signature)[:20]
    if not valid_artifact(directory, signature):
        write_json(directory / "experiment.json", resolved)
        complete_artifact(directory, signature, ["experiment.json"])
    experiment = SyntheticExperiment(directory, implementation)
    write_json(root / "current.json", experiment.identity)
    return experiment


def load_experiment(config, implementation=None):
    root = Path(config["experiment_root"])
    pointer = root / "current.json"
    if not pointer.exists():
        raise ValueError(
            "No prepared synthetic experiment; run diagnostics --stage all first"
        )
    identity = read_json(pointer)
    experiment = SyntheticExperiment(identity["experiment_directory"], implementation)
    if experiment.identity != identity or experiment.signature[
        "specification"
    ] != specification(config):
        raise ValueError(
            "Shared experiment differs from configuration; run diagnostics for this design"
        )
    if experiment.signature["producer"] != generation_signature(
        experiment.implementation
    ):
        raise ValueError(
            "Generation implementation changed; rerun diagnostics preparation"
        )
    return experiment
