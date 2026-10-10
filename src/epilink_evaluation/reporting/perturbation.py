"""Saved-table reports for frozen-setting perturbation studies."""

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..provenance import read_json
from .report import markdown_table, read_table


def render_report(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    reference = read_json(directory / "reference.json")
    config = manifest["config"]
    title = "Parameter perturbation" + (
        " — smoke validation" if config["smoke_mode"] else ""
    )
    paragraphs = [
        f"Status: {manifest['status']}. Reference baseline: {reference['run_directory']}.",
        f"Cases: {manifest['n_cases']} (reference backbone: {reference['n_cases']}). Observation seeds: {config['seeds']}.",
        "All operating settings and both logistic models are frozen from the reference. Matched inference updates EpiLink parameters; baseline_fixed keeps baseline inference parameters. No scenario-specific fitting or retuning occurs.",
        "Deltas are perturbed minus an unperturbed control on the same seed and in the same inference mode. Negative F1/AP deltas indicate lower recovery; positive contamination deltas indicate more distant pairs. These are fresh controls, not differences from the earlier baseline evaluation seeds.",
        "Seeds are paired across scenarios; changed distributions can consume random draws differently. Realizations share a fixed backbone. SD/range describe conditional observation variation, not independent epidemics.",
        "Feature ambiguity uses the synthetic diagnostics' exact saved GD or (GD, TD) cells, without extra rounding or binning. Mixed cells contain both target and non-target pairs. It is calculated once per scenario and seed because inference modes share observations. Ambiguity deltas match the fresh control by seed, process, feature set and endpoint. Minimum feature-only error is empirical, not a universal performance ceiling.",
    ]
    if manifest.get("requested_stage") == "observations":
        paragraphs.append(
            "Requested stage: observations. Status describes feature-ambiguity coverage; frozen-method replay coverage is reported separately."
        )
    if config["smoke_mode"]:
        paragraphs.append(
            "This reduced-backbone study validates the pipeline. Its results do not replace full-backbone sensitivity evidence."
        )
    text = [f"# {title}", *paragraphs]
    body = [
        f"<h1>{html.escape(title)}</h1>",
        *[f"<p>{html.escape(p)}</p>" for p in paragraphs],
    ]

    def section(heading, table):
        text.extend([f"## {heading}", markdown_table(table)])
        body.extend(
            [
                f"<h2>{html.escape(heading)}</h2>",
                table.to_html(index=False, float_format=lambda v: f"{v:.4g}")
                if not table.empty
                else "<p>No completed results.</p>",
            ]
        )

    section("Comparison coverage", read_table(directory / "coverage.csv"))
    ambiguity_coverage = read_table(directory / "ambiguity_coverage.csv")
    if not ambiguity_coverage.empty:
        section("Feature-ambiguity coverage", ambiguity_coverage)
    ambiguity_metrics = [
        "mixed_cell_fraction",
        "pair_fraction_in_mixed_cells",
        "target_fraction_in_mixed_cells",
        "target_prevalence_in_mixed_cells",
        "class_conditional_overlap",
        "minimum_feature_only_misclassification_rate",
    ]
    ambiguity = read_table(directory / "ambiguity_summary.csv")
    if not ambiguity.empty:
        section(
            "Exact feature-cell ambiguity",
            ambiguity[
                [
                    "scenario",
                    "process",
                    "feature_set",
                    "endpoint",
                    "n_realizations",
                    *[f"{metric}_mean" for metric in ambiguity_metrics],
                ]
            ],
        )
    ambiguity_deltas = read_table(directory / "ambiguity_delta_summary.csv")
    if not ambiguity_deltas.empty:
        section(
            "Paired feature-ambiguity changes",
            ambiguity_deltas[
                [
                    "scenario",
                    "process",
                    "feature_set",
                    "endpoint",
                    "n_controls",
                    *[f"delta_{metric}_mean" for metric in ambiguity_metrics],
                ]
            ],
        )
    scenarios = read_json(directory / "scenarios.json")
    section(
        "Perturbations",
        pd.DataFrame(
            [{k: v for k, v in row.items() if k != "generation"} for row in scenarios]
        ),
    )
    selection = read_json(directory / "selection.json")
    section(
        "Frozen reference decisions",
        pd.DataFrame(
            [
                {k: p.get(k) for k in ("pipeline", "criterion", "status", "setting_id")}
                for p in selection["operating_points"]
            ]
        ),
    )
    rankings = read_table(directory / "rankings_delta_summary.csv")
    if not rankings.empty:
        section(
            "Paired ranking changes",
            rankings[
                [
                    "scenario",
                    "mode",
                    "score_name",
                    "delta_M0_AP_mean",
                    "delta_M0_AP_std",
                    "delta_M0_AP_count",
                    "n_controls",
                ]
            ],
        )
    results = read_table(directory / "results_delta_summary.csv")
    if not results.empty:
        columns = [
            "scenario",
            "mode",
            "criterion",
            "pipeline",
            "delta_M0_f1_mean",
            "delta_M0_f1_std",
            "delta_M0_f1_count",
            "delta_Mge3_contamination_mean",
            "n_controls",
        ]
        section("Paired operating-point changes", results[columns])
        output = directory / "figures"
        output.mkdir(exist_ok=True)
        for i, (criterion, group) in enumerate(results.groupby("criterion", sort=True)):
            modes = config["modes"]
            fig, axes = plt.subplots(
                1, len(modes), figsize=(7 * len(modes), 10), squeeze=False
            )
            finite = group.delta_M0_f1_mean.dropna()
            bound = max(0.01, float(finite.abs().max())) if len(finite) else 1.0
            for axis, mode in zip(axes.flat, modes):
                subset = group.loc[group["mode"] == mode]
                if subset.empty:
                    axis.set_title(f"{mode}: no results")
                    axis.set_axis_off()
                    continue
                table = subset.pivot(
                    index="pipeline", columns="scenario", values="delta_M0_f1_mean"
                )
                image = axis.imshow(
                    np.ma.masked_invalid(table.to_numpy(float)),
                    aspect="auto",
                    cmap="coolwarm",
                    vmin=-bound,
                    vmax=bound,
                )
                axis.set_xticks(
                    range(len(table.columns)),
                    table.columns,
                    rotation=60,
                    ha="right",
                    fontsize=7,
                )
                axis.set_yticks(range(len(table.index)), table.index, fontsize=7)
                axis.set_title(f"{criterion}: {mode}")
                fig.colorbar(image, ax=axis, label="Mean paired M=0 F1 change")
            fig.tight_layout()
            relative = f"figures/paired_f1_{i}.png"
            fig.savefig(directory / relative, dpi=150)
            plt.close(fig)
            text.append(f"![Paired F1 changes]({relative})")
            body.append(f'<img src="{relative}" alt="Paired F1 changes">')
    (directory / "report.md").write_text("\n\n".join(text) + "\n")
    (directory / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>'
        + html.escape(title)
        + "</title><style>body{font:15px system-ui;margin:2em}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}td,th{padding:.4em;border:1px solid #ccc}img{max-width:100%}</style><body>"
        + "\n".join(body)
        + "</body></html>\n"
    )
