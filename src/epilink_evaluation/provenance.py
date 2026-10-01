"""Content-addressed artifacts, atomic metadata, and explicit stage status."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(clean_json(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def versions():
    result = {}
    for name in (
        "epilink",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "igraph",
        "networkx",
        "pyarrow",
        "TreeCluster",
        "phylo-treetime",
        "biopython",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def implementation_signature():
    import epilink

    package = Path(__file__).parent
    epilink_root = Path(epilink.__file__).parent
    return {
        "evaluation": {
            str(p.relative_to(package)): digest_file(p)
            for p in sorted(package.rglob("*.py"))
        },
        "epilink": {
            str(p.relative_to(epilink_root)): digest_file(p)
            for p in sorted(epilink_root.rglob("*.py"))
        },
        "versions": versions(),
    }


def generation_signature(implementation):
    """Observation producers only; analysis and presentation cannot change data IDs."""
    paths = (
        "inputs/synthetic.py",
        "truth/relationships.py",
        "schemas.py",
        "natural_history.py",
    )
    return {
        "evaluation": {p: implementation["evaluation"][p] for p in paths},
        "epilink": implementation["epilink"],
        "versions": {
            p: implementation["versions"].get(p)
            for p in (
                "epilink",
                "numpy",
                "pandas",
                "networkx",
                "pyarrow",
                "scipy",
            )
        },
    }


def baseline_signature(implementation):
    """Comparison code identity, excluding independent studies and presentation."""
    excluded = ("reporting/", "diagnostics/")
    independent = {
        "cli.py",
        "__main__.py",
        "workflows/diagnostics.py",
        "workflows/boston.py",
        "workflows/boston_config.py",
        "workflows/boston_scoring.py",
        "workflows/boston_assessment.py",
        "workflows/perturbation.py",
        "workflows/perturbation_config.py",
        "workflows/reference.py",
        "inputs/boston.py",
        "phylogeny/boston.py",
    }
    return {
        **implementation,
        "evaluation": {
            p: h
            for p, h in implementation["evaluation"].items()
            if not p.startswith(excluded) and p not in independent
        },
    }


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def complete_artifact(directory, signature, files, **metadata):
    directory = Path(directory)
    write_json(
        directory / "manifest.json",
        {
            "status": "complete",
            "signature": signature,
            "fingerprint": fingerprint(signature),
            "files": {name: digest_file(directory / name) for name in files},
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        },
    )


def valid_artifact(directory, signature):
    directory = Path(directory)
    path = directory / "manifest.json"
    if not path.exists():
        return False
    saved = read_json(path)
    return (
        saved.get("status") == "complete"
        and saved.get("fingerprint") == fingerprint(signature)
        and bool(saved.get("files"))
        and all(
            (directory / name).is_file() and digest_file(directory / name) == expected
            for name, expected in saved["files"].items()
        )
    )
