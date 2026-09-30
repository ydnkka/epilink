"""Read-only, integrity-checked access to an evaluated baseline experiment."""

import shutil
from copy import deepcopy
from pathlib import Path

import networkx as nx

from ..phylogeny.external import command_identity
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


class BaselineReference(OperatingReference):
    """Full reference, including fitted models and completed held-out artifacts."""

    def __init__(self, path, implementation):
        super().__init__(path, implementation)
        path = self.directory
        signature = self.manifest["signature"]
        run_id = self.identity["run_fingerprint"]
        old = signature["implementation"]
        self.tools = {}
        if self.config["treecluster"]["enabled"]:
            for name, executable in self.config["treecluster"]["executables"].items():
                self.tools[name] = command_identity(executable)
                if self.tools[name]["sha256"] != signature["tools"][name].get("sha256"):
                    raise ValueError(f"Reference executable changed: {name}")
        self.training_id = self.frozen["training_fingerprint"]
        self.model_directory = self.root / "artifacts/models" / self.training_id[:20]
        model_manifest = checked_artifact(self.model_directory)
        if (
            model_manifest["fingerprint"] != self.training_id
            or model_manifest["seeds"] != self.config["splits"]["train"]
        ):
            raise ValueError(
                "Reference training identity differs from frozen selection"
            )
        self.models = read_json(self.model_directory / "models.json")
        self.truth_directory = self.root / "artifacts/truth" / signature["truth"]
        truth_manifest = checked_artifact(self.truth_directory)
        topology = truth_manifest["signature"]
        self.tree = nx.DiGraph()
        self.tree.add_nodes_from(topology["nodes"])
        self.tree.add_edges_from(topology["edges"])
        TreeIndex(self.tree)
        if len(self.tree) != truth_manifest["n_cases"]:
            raise ValueError("Reference truth topology size differs from its manifest")

        # Check completed held-out artifacts, not just the latest command status
        # (which may have been overwritten by a later development/report action).
        pairs = {k: v for k, v in self.selected.items() if v["kind"] == "pairwise"}
        clusters = {k: v for k, v in self.selected.items() if v["kind"] != "pairwise"}
        for seed in self.config["splits"]["evaluation"]:
            directory = path / "evaluation" / f"seed_{seed}"
            pair_manifest = checked_artifact(directory / "pairwise")
            score_id = pair_manifest["signature"]["score_id"]
            common = {
                "run": run_id,
                "score_id": score_id,
                "split": "evaluation",
                "seed": seed,
            }
            checked_artifact(directory / "pairwise", {**common, "definitions": pairs})
            status = read_json(directory / "clusters/status.json")
            if (
                status["status"] != "complete"
                or status["errors"]
                or status["completed"] != len(clusters)
                or status["configured"] != len(clusters)
            ):
                raise ValueError(f"Reference evaluation is incomplete for seed {seed}")
            for key, definition in clusters.items():
                checked_artifact(
                    directory / "clusters" / key, {**common, "definition": definition}
                )
        self.identity = {
            "run_directory": str(path),
            "run_fingerprint": run_id,
            "selection_fingerprint": fingerprint(self.frozen),
            "training_fingerprint": self.training_id,
            "truth_fingerprint": truth_manifest["fingerprint"],
            "n_cases": len(self.tree),
            "model_sha256": digest_file(self.model_directory / "models.json"),
            "baseline_implementation": old,
        }

    def copy_models(self, root):
        destination = Path(root) / "artifacts/models" / self.training_id[:20]
        manifest = read_json(self.model_directory / "manifest.json")
        if not valid_artifact(destination, manifest["signature"]):
            destination.mkdir(parents=True, exist_ok=True)
            for name in (*manifest["files"], "manifest.json"):
                shutil.copyfile(self.model_directory / name, destination / name)
