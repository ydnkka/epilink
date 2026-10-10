"""Read-only, integrity-checked access to an evaluated baseline experiment."""

from copy import deepcopy
from pathlib import Path

import networkx as nx

from ..inputs.experiment import SyntheticExperiment
from ..provenance import digest_file, fingerprint, read_json, valid_artifact
from ..truth import TreeIndex


def checked_artifact(directory, expected=None):
    directory = Path(directory)
    saved = read_json(directory / "manifest.json")
    signature = saved.get("signature") if expected is None else expected
    if signature is None or not valid_artifact(directory, signature):
        raise ValueError(f"Incomplete or changed reference artifact: {directory}")
    return saved


class OperatingReference:
    """Frozen operating rules and inference parameters, without fitted models.

    Validate run identity, selection evidence and held-out selection provenance.
    Consumers using training-free scores do not need training artifacts, the
    synthetic truth topology, or local phylogenetic executables.
    """

    def __init__(self, path, implementation):
        path = Path(path).resolve()
        if path.is_file():
            path = Path(read_json(path)["run_directory"]).resolve()
        if path.parent.name != "runs":
            raise ValueError(
                "baseline_run must identify a baseline runs/<id> directory or current.json"
            )
        self.directory, self.root = path, path.parent.parent
        self.manifest = read_json(path / "manifest.json")
        self.config = deepcopy(self.manifest["config"])
        self.frozen = read_json(path / "selection/operating_points.json")
        signature = self.manifest["signature"]
        run_id = fingerprint(signature)
        if (
            self.frozen["run_fingerprint"] != run_id
            or path.name != run_id[:20]
            or self.config["generation"] != self.config["inference"]
        ):
            raise ValueError(
                "Reference run identity or matched baseline parameters differ"
            )
        scientific = {
            k: v
            for k, v in self.config.items()
            if k not in ("selection", "output_directory", "config_path")
        }
        if (
            scientific != signature["config"]
            or self.frozen["criteria"] != self.config["selection"]["criteria"]
        ):
            raise ValueError(
                "Reference configuration differs from its frozen experiment"
            )
        if (
            digest_file(path / "development/metrics.csv")
            != self.frozen["development_evidence_sha256"]
        ):
            raise ValueError("Reference development evidence changed after selection")
        if read_json(path / "evaluation/selection_used.json") != self.frozen:
            raise ValueError(
                "Reference evaluation did not use these frozen operating settings"
            )
        access = read_json(path / "evaluation/heldout_access.json")
        if (
            access["selection_fingerprint"] != fingerprint(self.frozen)
            or access["seeds"] != self.config["splits"]["evaluation"]
        ):
            raise ValueError(
                "Reference held-out access record differs from frozen settings"
            )

        # New workflow/CLI/reporting code can consume an old baseline. Scientific
        # producers and dependencies must still implement the same definitions.
        old = signature["implementation"]
        prefixes = (
            "inputs/",
            "truth/",
            "scorers/",
            "graphs/",
            "clusterers/",
            "metrics/",
            "phylogeny/",
            "selection/",
        )
        modules = {
            "schemas.py",
            "natural_history.py",
            "config.py",
            "workflows/baseline.py",
            "workflows/settings.py",
        }
        # The Boston adapter prepares the *consumer's* empirical inputs. It
        # cannot change the synthetic observations underlying this reference.
        changed = [
            name
            for name, checksum in old["evaluation"].items()
            if name != "inputs/boston.py"
            and (name.startswith(prefixes) or name in modules)
            and implementation["evaluation"].get(name) != checksum
        ]
        if (
            changed
            or old["epilink"] != implementation["epilink"]
            or old["versions"] != implementation["versions"]
        ):
            raise ValueError(
                f"Scientific implementation/dependencies differ from baseline: {changed}"
            )
        definitions = read_json(path / "settings.json")
        self.selected = {}
        for point in self.frozen["operating_points"]:
            if point["status"] == "selected":
                key, definition = point["setting_id"], point["definition"]
                if (
                    definitions.get(key) != definition
                    or key != fingerprint(definition)[:20]
                ):
                    raise ValueError(
                        "Frozen method definition differs from reference settings"
                    )
                self.selected[key] = definition
        if not self.selected:
            raise ValueError("Reference has no selected operating points")
        self.identity = {
            "run_directory": str(path),
            "run_fingerprint": run_id,
            "selection_fingerprint": fingerprint(self.frozen),
            "baseline_implementation": old,
        }


class EpiLinkClusteringReference(OperatingReference):
    """Training-free reference for the requested score-weighted Leiden pipelines."""

    def __init__(self, path, implementation, scorers, criterion):
        super().__init__(path, implementation)
        pipelines = {f"leiden/{name}" for name in scorers}
        points = [
            p for p in self.frozen["operating_points"]
            if p["criterion"] == criterion and p["pipeline"] in pipelines
        ]
        if (
            len(points) != len(pipelines)
            or {p["pipeline"] for p in points} != pipelines
            or any(p["status"] != "selected" for p in points)
        ):
            raise ValueError(
                "Reference requires a feasible EpiLink setting "
                "for every requested pipeline"
            )
        self.frozen = {**self.frozen, "operating_points": points}
        self.selected = {p["setting_id"]: p["definition"] for p in points}
        if any(d.get("graph_mode") != "full" for d in self.selected.values()):
            raise ValueError(
                "Perturbation requires baseline full-graph Leiden settings; "
                "regenerate the baseline reference"
            )
        experiment_identity = self.manifest["signature"]["experiment"]
        self.experiment = SyntheticExperiment(
            experiment_identity["experiment_directory"], implementation
        )
        if self.experiment.identity != experiment_identity:
            raise ValueError("Reference synthetic experiment identity differs")
        self.truth_directory = self.experiment.truth_directory
        if self.truth_directory.name != self.manifest["signature"]["truth"]:
            raise ValueError("Reference truth differs from the pinned experiment")
        truth_manifest = checked_artifact(self.truth_directory)
        topology = truth_manifest["signature"]
        self.tree = nx.DiGraph()
        self.tree.add_nodes_from(topology["nodes"])
        self.tree.add_edges_from(topology["edges"])
        TreeIndex(self.tree)
        if len(self.tree) != truth_manifest["n_cases"]:
            raise ValueError("Reference truth topology size differs from its manifest")
        for seed in self.config["splits"]["evaluation"]:
            for key, definition in self.selected.items():
                artifact = (
                    self.directory / "evaluation" / f"seed_{seed}" / "clusters" / key
                )
                saved = checked_artifact(artifact)
                signature = saved["signature"]
                checked_artifact(
                    artifact,
                    {
                        "run": self.identity["run_fingerprint"],
                        "score_id": signature["score_id"],
                        "definition": definition, "split": "evaluation", "seed": seed,
                    },
                )
                checked_artifact(self.root / "artifacts/scores" / signature["score_id"])
        self.identity.update(
            truth_fingerprint=truth_manifest["fingerprint"],
            n_cases=len(self.tree),
        )
