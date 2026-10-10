from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..provenance import digest_file


def executable(name):
    candidates = ("iqtree3", "iqtree2", "iqtree") if str(name) == "iqtree" else (name,)
    for candidate in candidates:
        found = shutil.which(str(candidate))
        if found:
            return str(Path(found).resolve())
        sibling = Path(sys.executable).parent / candidate
        if sibling.is_file() and os.access(sibling, os.X_OK):
            return str(sibling.resolve())
    raise FileNotFoundError(
        f"Executable not found: {name}; configure a path or activate its environment"
    )


def command_identity(name):
    path = executable(name)
    return {"path": path, "sha256": digest_file(path)}


def treecluster_executable(config):
    """Use the canonical executable field, retaining legacy tool-map support."""
    return config.get("executable") or config.get("executables", {}).get(
        "treecluster", "TreeCluster.py"
    )


def run_command(argv, directory, name, timeout):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PATH"] = (
        str(Path(argv[0]).parent) + os.pathsep + environment.get("PATH", "")
    )
    try:
        result = subprocess.run(
            [str(value) for value in argv],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        (directory / f"{name}.stderr.log").write_text(str(exc))
        raise
    (directory / f"{name}.stdout.log").write_text(result.stdout)
    (directory / f"{name}.stderr.log").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(
            f"{name} exited {result.returncode}; see {directory / (name + '.stderr.log')}"
        )
    return result.stdout
