"""Provenance, paths, and table output shared by all four investigations."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import ROOT


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, (Path, np.ndarray)):
            return str(x) if isinstance(x, Path) else x.tolist()
        raise TypeError(type(x).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=convert) + "\n")


def save_table(directory, name, rows):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(directory / f"{name}.csv", index=False)
    return frame


def versions():
    names = ["epilink", "numpy", "pandas", "scipy", "igraph", "scikit-learn", "pyarrow"]
    return {name: importlib.metadata.version(name) for name in names}


def source_hashes():
    import epilink
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += list((ROOT / "src/evaluation").glob("*.py"))
    paths += list(Path(epilink.__file__).parent.rglob("*.py"))
    return {str(p.resolve()): digest_file(p) for p in sorted(paths)}


def log(message):
    from datetime import datetime
    print(f"{datetime.now():%H:%M:%S} {message}", flush=True)
