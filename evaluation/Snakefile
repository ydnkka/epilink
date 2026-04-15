from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


WORKFLOW_ROOT = Path(workflow.basedir).resolve()
SOURCE_ROOT = WORKFLOW_ROOT / "src"

# Both src/ (package root) and src/evaluation/ (bare-import root) must be
# on sys.path so that Snakemake itself and the subprocesses it spawns can
# resolve all module imports correctly.
for _extra in (str(SOURCE_ROOT / "evaluation"), str(SOURCE_ROOT)):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from evaluation.config import (
    load_config,
    resolve_configured_output_path,
    resolve_configured_path,
)


CONFIG_PATH = Path(config.get("evaluation_config", WORKFLOW_ROOT / "config.yaml")).expanduser()
if not CONFIG_PATH.is_absolute():
    CONFIG_PATH = (WORKFLOW_ROOT / CONFIG_PATH).resolve()

EVALUATION_CONFIG = load_config(CONFIG_PATH)
CONFIG_INPUT = str(CONFIG_PATH)
TREE_INPUT = str(resolve_configured_output_path(EVALUATION_CONFIG, "outputs.scovmod.tree_path"))

SCOVMOD_DIR = resolve_configured_output_path(EVALUATION_CONFIG, "outputs.scovmod.directory")
SPARSIFICATION_DIR = resolve_configured_output_path(
    EVALUATION_CONFIG,
    "outputs.sparsification.directory",
)
STABILITY_DIR = resolve_configured_output_path(EVALUATION_CONFIG, "outputs.stability.directory")
SYNTHETIC_DIR = resolve_configured_output_path(EVALUATION_CONFIG, "outputs.synthetic.directory")
BOSTON_DIR = resolve_configured_output_path(EVALUATION_CONFIG, "outputs.boston.directory")
LOG_DIR = resolve_configured_path(EVALUATION_CONFIG, "outputs.directory") / "logs"

SCOVMOD_INPUTS = [
    CONFIG_INPUT,
    str(resolve_configured_path(EVALUATION_CONFIG, "paths.scovmod.infection_path")),
    str(resolve_configured_path(EVALUATION_CONFIG, "paths.scovmod.transmission_path")),
]
SCOVMOD_OUTPUTS = [
    str(resolve_configured_output_path(EVALUATION_CONFIG, "outputs.scovmod.tree_path")),
    str(resolve_configured_output_path(EVALUATION_CONFIG, "outputs.scovmod.heterogeneity_path")),
    str(SCOVMOD_DIR / "tree_summary.parquet"),
    str(SCOVMOD_DIR / "component_sizes.parquet"),
    str(SCOVMOD_DIR / "degree_distributions.parquet"),
]

SPARSIFICATION_OUTPUTS = [
    str(SPARSIFICATION_DIR / "score_surfaces.parquet"),
    str(SPARSIFICATION_DIR / "sparsify_edge_retention.parquet"),
    str(resolve_configured_output_path(EVALUATION_CONFIG, "outputs.sparsification.optimal_thresholds_path")),
]

STABILITY_OUTPUTS = [
    str(STABILITY_DIR / "case_counts_over_time.parquet"),
    str(STABILITY_DIR / "stability_resolution_selection.parquet"),
]

BASELINE_OUTPUTS = [
    str(SYNTHETIC_DIR / "baseline_scores.parquet"),
    str(SYNTHETIC_DIR / "baseline_perfomance.json"),
    str(SYNTHETIC_DIR / "baseline_summary.parquet"),
]

SYNTHETIC_OUTPUTS = [str(SYNTHETIC_DIR / "results.parquet")]

BOSTON_INPUTS = [
    str(resolve_configured_path(EVALUATION_CONFIG, "paths.boston.metadata_path")),
    str(resolve_configured_path(EVALUATION_CONFIG, "paths.boston.pairwise_path")),
]
BOSTON_OUTPUTS = [
    str(BOSTON_DIR / "cluster_summary.parquet"),
    str(BOSTON_DIR / "cluster_composition.parquet"),
    str(BOSTON_DIR / "cluster_sizes.parquet"),
]


def _module_env() -> dict[str, str]:
    """Return a copy of the environment with PYTHONPATH set for module execution.

    Both ``src/`` (for package-style imports, e.g. ``evaluation.config``) and
    ``src/evaluation/`` (for the bare intra-package imports used by the
    entry-point modules, e.g. ``from config import ...``) are prepended so that
    the subprocess can resolve all imports regardless of the caller's environment.
    """
    environment = os.environ.copy()
    extra_paths = [str(SOURCE_ROOT / "evaluation"), str(SOURCE_ROOT)]
    current_pythonpath = environment.get("PYTHONPATH")
    if current_pythonpath:
        environment["PYTHONPATH"] = os.pathsep.join(extra_paths + [current_pythonpath])
    else:
        environment["PYTHONPATH"] = os.pathsep.join(extra_paths)
    return environment


def run_module(module_name: str, log_path: str) -> None:
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{module_name}] log -> {log_file}", file=sys.stderr)

    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"from evaluation.{module_name} import main; main({str(CONFIG_PATH)!r})",
        ],
        cwd=WORKFLOW_ROOT,
        env=_module_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    with log_file.open("w", encoding="utf-8") as handle:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stderr.write(line)
            handle.write(line)

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, process.args)


localrules: all


rule all:
    input:
        SCOVMOD_OUTPUTS,
        SPARSIFICATION_OUTPUTS,
        BASELINE_OUTPUTS,
        STABILITY_OUTPUTS,
        SYNTHETIC_OUTPUTS,
        BOSTON_OUTPUTS,


rule scovmod:
    input:
        SCOVMOD_INPUTS
    output:
        SCOVMOD_OUTPUTS
    log:
        str(LOG_DIR / "scovmod.log")
    message:
        "Running scovmod.py"
    run:
        run_module("scovmod", log[0])


rule sparsification:
    input:
        SCOVMOD_OUTPUTS + [CONFIG_INPUT, TREE_INPUT]
    output:
        SPARSIFICATION_OUTPUTS
    log:
        str(LOG_DIR / "sparsification.log")
    message:
        "Running sparsification.py"
    run:
        run_module("sparsification", log[0])


rule baseline:
    input:
        SPARSIFICATION_OUTPUTS + [CONFIG_INPUT, TREE_INPUT]
    output:
        BASELINE_OUTPUTS
    log:
        str(LOG_DIR / "baseline.log")
    message:
        "Running baseline.py"
    run:
        run_module("baseline", log[0])


rule stability:
    input:
        SPARSIFICATION_OUTPUTS + [CONFIG_INPUT, TREE_INPUT]
    output:
        STABILITY_OUTPUTS
    log:
        str(LOG_DIR / "stability.log")
    message:
        "Running stability.py"
    run:
        run_module("stability", log[0])


rule experiments:
    input:
        STABILITY_OUTPUTS + SPARSIFICATION_OUTPUTS + [CONFIG_INPUT, TREE_INPUT]
    output:
        SYNTHETIC_OUTPUTS
    log:
        str(LOG_DIR / "experiments.log")
    message:
        "Running experiments.py"
    run:
        run_module("experiments", log[0])


rule boston:
    input:
        SYNTHETIC_OUTPUTS + [CONFIG_INPUT] + BOSTON_INPUTS
    output:
        BOSTON_OUTPUTS
    log:
        str(LOG_DIR / "boston.log")
    message:
        "Running boston.py"
    run:
        run_module("boston", log[0])
