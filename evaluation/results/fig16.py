"""Paired observation ambiguity changes across perturbation scenarios (fig16)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from epilink_evaluation.inputs.synthetic import load_observations, load_truth
from epilink_evaluation.utils import style

from ._perturbation.common import (
    PROCESS_LABELS,
    PROCESSES,
    Study,
    add_arguments,
    scenario_axis,
)
from ._perturbation.plotting import symmetric_bound

ENDPOINTS = ("M0", "Mle1", "Mle2")
ENDPOINT_LABELS = {"M0": "$M=0$", "Mle1": "$M\\leq1$", "Mle2": "$M\\leq2$"}
FEATURE_SET = "GD_TD"
PROCESS_COLORS = {"deterministic": "#0072B2", "stochastic": "#D55E00"}


def _observation_artifacts(study: Study) -> dict[tuple[str, int], Path]:
    """Find saved observation artifacts for each scenario and evaluation seed."""
    artifact_root = study.run.parent.parent / "artifacts" / "observations"
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"Observation artifacts not found: {artifact_root}")

    manifests = []
    for path in artifact_root.glob("*/manifest.json"):
        with path.open() as handle:
            manifest = json.load(handle)
        if manifest.get("signature", {}).get("kind") == "observations-v3":
            manifests.append((path.parent, manifest))

    artifacts = {}
    truth_name = None
    for scenario in study.scenarios:
        for seed in study.seeds:
            matches = [
                (directory, manifest)
                for directory, manifest in manifests
                if manifest["signature"].get("seed") == seed
                and manifest["signature"].get("generation") == scenario["generation"]
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Expected one saved observation artifact for "
                    f"{scenario['name']}/seed_{seed}; found {len(matches)}"
                )
            directory, manifest = matches[0]
            if manifest.get("status") != "complete":
                raise ValueError(
                    f"Incomplete observation artifact: {scenario['name']}/seed_{seed}"
                )
            current_truth = manifest["signature"]["truth"]
            if truth_name is None:
                truth_name = current_truth
            elif current_truth != truth_name:
                raise ValueError("Observation artifacts use different truth references")
            artifacts[scenario["name"], seed] = directory
    return artifacts


def _feature_ambiguity(observations, truth) -> dict[tuple[str, str], float]:
    """Calculate mixed-cell fractions for exact (GD, TD) cells efficiently."""
    td_values = observations.TD.to_numpy(dtype=float)
    if not np.all(np.isfinite(td_values) & (td_values >= 0)) or not np.all(
        td_values == np.floor(td_values)
    ):
        raise ValueError("Sampling-time distances must be nonnegative whole days")
    td = td_values.astype(np.int64)
    scale = int(td.max()) + 1

    m = truth.M.to_numpy(dtype=float, na_value=np.nan)
    targets = {
        endpoint: np.isfinite(m) & (m <= horizon)
        for endpoint, horizon in zip(ENDPOINTS, (0, 1, 2))
    }
    values = {}
    for process in PROCESSES:
        gd_values = observations[f"GD_{process}"].to_numpy(dtype=float)
        if not np.all(np.isfinite(gd_values) & (gd_values >= 0)) or not np.all(
            gd_values == np.floor(gd_values)
        ):
            raise ValueError("Genetic distances must be nonnegative integers")
        gd = gd_values.astype(np.int64)
        if gd.max() > (np.iinfo(np.int64).max - int(td.max())) // scale:
            raise ValueError("Feature-cell identifiers exceed the supported range")
        _, groups = np.unique(gd * scale + td, return_inverse=True)
        cell_count = int(groups.max()) + 1
        cell_sizes = np.bincount(groups, minlength=cell_count)
        for endpoint, target in targets.items():
            target_sizes = np.bincount(groups[target], minlength=cell_count)
            mixed_count = np.count_nonzero(
                (target_sizes > 0) & (target_sizes < cell_sizes)
            )
            values[endpoint, process] = mixed_count / cell_count
    return values


def _seed_ambiguity(artifacts) -> dict[tuple[str, int, str, str], float]:
    """Calculate GD + sampling-time ambiguity from saved pairs for each seed."""
    values = {}
    for (scenario, seed), directory in artifacts.items():
        observations, _ = load_observations(directory)
        with (directory / "manifest.json").open() as handle:
            truth_name = json.load(handle)["signature"]["truth"]
        truth_directory = directory.parent.parent / "truth" / truth_name
        truth = load_truth(truth_directory, observations.pair_id)
        summary = _feature_ambiguity(observations, truth)
        if len(summary) != len(ENDPOINTS) * len(PROCESSES):
            raise ValueError(f"Incomplete ambiguity summary: {scenario}/seed_{seed}")
        for (endpoint, process), value in summary.items():
            values[scenario, seed, endpoint, process] = float(value)
    return values


def ambiguity_changes(study: Study) -> pd.DataFrame:
    """Summarize paired scenario-minus-control ambiguity in percentage points."""
    values = _seed_ambiguity(_observation_artifacts(study))
    records = []
    for scenario in study.scenario_names:
        for endpoint in ENDPOINTS:
            for process in PROCESSES:
                controls, perturbed, differences = [], [], []
                for seed in study.seeds:
                    control = values["baseline", seed, endpoint, process]
                    scenario_value = values[scenario, seed, endpoint, process]
                    if np.isfinite(control) and np.isfinite(scenario_value):
                        controls.append(control)
                        perturbed.append(scenario_value)
                        differences.append((scenario_value - control) * 100)
                records.append(
                    {
                        "scenario": scenario,
                        "process": process,
                        "feature_set": FEATURE_SET,
                        "endpoint": endpoint,
                        "control_mean_fraction": float(np.mean(controls))
                        if controls
                        else np.nan,
                        "scenario_mean_fraction": float(np.mean(perturbed))
                        if perturbed
                        else np.nan,
                        "change_mean_pp": float(np.mean(differences))
                        if differences
                        else np.nan,
                        "change_min_pp": float(np.min(differences))
                        if differences
                        else np.nan,
                        "change_max_pp": float(np.max(differences))
                        if differences
                        else np.nan,
                        "defined_pairs": len(differences),
                    }
                )
    return pd.DataFrame(records)


def create_figure(study: Study, output, *, fmt="both") -> None:
    stem = "fig16_observation_ambiguity"
    csv_path = output / f"{stem}.csv"
    cached = csv_path.is_file()
    frame = pd.read_csv(csv_path) if cached else ambiguity_changes(study)
    bound = symmetric_bound(frame[["change_min_pp", "change_max_pp"]].to_numpy())
    fig, axes = style.new_figure(
        width="double",
        height_in=5,
        nrows=1,
        ncols=len(ENDPOINTS),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    offsets = {"deterministic": -0.12, "stochastic": 0.12}
    for col, endpoint in enumerate(ENDPOINTS):
        ax = axes[col]
        for process in PROCESSES:
            group = (
                frame.loc[frame.endpoint.eq(endpoint) & frame.process.eq(process)]
                .set_index("scenario")
                .loc[study.scenario_names]
            )
            mean = group.change_mean_pp.to_numpy(float)
            low = group.change_min_pp.to_numpy(float)
            high = group.change_max_pp.to_numpy(float)
            valid = np.isfinite(mean)
            ax.errorbar(
                mean[valid],
                np.flatnonzero(valid) + offsets[process],
                xerr=np.vstack(
                    (
                        np.maximum(0, mean[valid] - low[valid]),
                        np.maximum(0, high[valid] - mean[valid]),
                    )
                ),
                fmt="o",
                color=PROCESS_COLORS[process],
                markersize=3.8,
                capsize=2,
                elinewidth=0.9,
            )
            for position, count in enumerate(group.defined_pairs):
                if count < len(study.seeds):
                    ax.text(
                        bound * 0.96,
                        position + offsets[process],
                        f"n={int(count)}",
                        ha="right",
                        va="center",
                        fontsize=6.5,
                        color=PROCESS_COLORS[process],
                    )
        scenario_axis(ax, study, show_labels=col == 0)
        ax.axvline(0, color="0.4", linestyle="--", lw=0.8)
        ax.set_xlim(-bound, bound)
        ax.grid(axis="x", color="0.9")
        ax.set_title(ENDPOINT_LABELS[endpoint], fontweight="bold")

    fig.supxlabel(
        "Change in fraction of mixed observed "
        "combinations\n(scenario $minus$ baseline; percentage points)",
        x=0.6,
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=PROCESS_COLORS[process],
                marker="o",
                linestyle="none",
                label=PROCESS_LABELS[process],
            )
            for process in PROCESSES
        ],
        title="Genetic observations",
        loc="lower center",
        bbox_to_anchor=(0.6, 1.0),
        ncol=len(PROCESSES),
    )
    style.add_panel_labels(axes)
    paths = style.save_figure(
        fig,
        output / stem,
        width="double",
        save_pdf=fmt in ("pdf", "both"),
        save_png=fmt in ("png", "both"),
    )
    if not cached:
        frame.to_csv(csv_path, index=False)
    for path in paths.values():
        print(f"Figure saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, figure=True)
    args = parser.parse_args()
    study = Study.load(args.run_dir)
    create_figure(study, study.output(args.output_dir), fmt=args.format)


if __name__ == "__main__":
    main()
