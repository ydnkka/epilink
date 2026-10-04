"""Reporting for Boston empirical clustering results."""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..provenance import read_json
from .report import markdown_table, read_table


def render_report(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    reference = read_json(directory / "reference.json")
    inputs = read_json(directory / "inputs.json")
    selection = read_json(directory / "selection.json")

    title = "Boston empirical clustering"
    requested = manifest.get("requested_stage")
    status = f"Status: {manifest['status']}"
    if requested:
        status += f" (requested stage: {requested})"
    status += "."
    paragraphs = [
        status,
        f"Reference baseline: {reference['run_directory']}.",
        f"Cases: {inputs['n_cases']}, Observed pairs: {inputs['n_observed_pairs']:,} / {inputs['n_all_pairs']:,} possible.",
        "The TN93 table is distance-censored at 0.0005/site; missing pairs are unobserved, not zero.",
        "All scorers use the same observed Boston GD and TD. EpiLink inference, Monte Carlo settings, and graph operating points are frozen from the synthetic baseline. LOGIT_S/LOGIT_D, when selected, reuse its fitted classifiers without retraining.",
        "Synthetic D/S labels identify the source operating rules or fitted classifiers, not separate Boston measurements. Raw and dated trees are rebuilt from the full Boston alignment; their TreeCluster partitions and exposure flags are descriptive comparators, not transmission truth.",
    ]
    if requested == "trees":
        paragraphs.append(
            "This was a tree-only run; graph scoring and clustering were not requested."
        )
    if inputs.get("treecluster_path"):
        paragraphs.append(
            f"External TreeCluster comparator: {inputs['treecluster_path']} (SHA-256 {inputs['treecluster_sha256']}). Each unclustered -1 is a separate singleton."
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
                else "<p>No results.</p>",
            ]
        )

    section("Input summary", pd.DataFrame([inputs]))
    section(
        "Frozen reference decisions",
        pd.DataFrame(
            [
                {
                    k: p.get(k)
                    for k in (
                        "pipeline",
                        "criterion",
                        "status",
                        "setting_id",
                        "baseline_setting_id",
                    )
                }
                for p in selection["operating_points"]
            ]
        ),
    )

    status_path = directory / "clusters" / "status.json"
    cluster_status = (
        read_json(status_path)
        if status_path.exists()
        else {"status": "not_run", "configured": 0, "completed": 0}
    )
    section("Clustering status", pd.DataFrame([cluster_status]))
    tree_status_path = directory / "trees/status.json"
    tree_status = (
        read_json(tree_status_path)
        if tree_status_path.exists()
        else {"status": "not_run", "configured": 0, "completed": 0}
    )
    if inputs.get("trees_enabled") or tree_status_path.exists():
        section("Raw and dated TreeCluster status", pd.DataFrame([tree_status]))
        tree_metrics = read_table(directory / "trees/metrics.csv")
        if not tree_metrics.empty:
            section("Frozen TreeCluster settings on Boston trees", tree_metrics)

    metrics_path = directory / "clusters" / "metrics.csv"
    if metrics_path.exists():
        metrics = read_table(metrics_path)
        if not metrics.empty:
            section("Cluster metrics", metrics)

            output = directory / "figures"
            output.mkdir(exist_ok=True)

            if "size_mean" in metrics.columns:
                size_metrics = ["size_mean", "size_std", "n_clusters"]
            else:
                size_metrics = (
                    ["size", "n_clusters"]
                    if "n_clusters" in metrics.columns
                    else ["size"]
                )

            size_data = metrics[
                ["setting_id", "pipeline"]
                + [c for c in size_metrics if c in metrics.columns]
            ].copy()
            if not size_data.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(size_data))
                width = 0.8 / max(1, len(size_data))

                if "size_mean" in size_data.columns:
                    ax.bar(
                        x,
                        size_data["size_mean"],
                        width,
                        yerr=size_data.get("size_std", 0),
                        capsize=3,
                    )
                    ax.set_ylabel("Mean cluster size")
                elif "size" in size_data.columns:
                    ax.bar(x, size_data["size"], width)
                    ax.set_ylabel("Cluster size")

                ax.set_xticks(x)
                ax.set_xticklabels(
                    size_data["pipeline"], rotation=45, ha="right", fontsize=8
                )
                ax.set_title("Cluster sizes by operating setting")
                fig.tight_layout()
                fig.savefig(directory / "figures" / "cluster_sizes.png", dpi=150)
                plt.close(fig)
                text.append("![Cluster sizes](figures/cluster_sizes.png)")
                body.append('<img src="figures/cluster_sizes.png" alt="Cluster sizes">')

    assessment = directory / "assessment"
    summary = read_table(assessment / "summary.csv")
    if not summary.empty:
        section("Empirical partition summary", summary)
    named = read_table(assessment / "named_cluster_overlaps.csv")
    if not named.empty:
        section("Named exposure groups", named)
        if "jaccard" in named.columns and named.jaccard.notna().any():
            fig, ax = plt.subplots(figsize=(max(9, len(named) * 0.65), 5))
            ax.bar(np.arange(len(named)), named.jaccard.fillna(0))
            ax.set_xticks(np.arange(len(named)))
            ax.set_xticklabels(
                named.pipeline + " / " + named.exposure,
                rotation=55,
                ha="right",
                fontsize=8,
            )
            ax.set_ylabel("Jaccard overlap with external TreeCluster group")
            ax.set_ylim(0, 1)
            fig.tight_layout()
            (directory / "figures").mkdir(exist_ok=True)
            fig.savefig(directory / "figures" / "named_overlaps.png", dpi=150)
            plt.close(fig)
            text.append("![Named group overlaps](figures/named_overlaps.png)")
            body.append(
                '<img src="figures/named_overlaps.png" alt="Named group overlaps">'
            )
    agreement = read_table(assessment / "tree_agreement.csv")
    if not agreement.empty:
        section("Graph vs raw/dated TreeCluster partition agreement", agreement)
    named_tree = read_table(assessment / "named_tree_overlaps.csv")
    if not named_tree.empty:
        section("Named exposure overlap with raw/dated TreeCluster", named_tree)
    if (assessment / "cluster_composition.csv").exists():
        paragraphs.append(
            "Full exposure, clade and mutation composition and all focused-cluster overlaps are in assessment/*.csv; fractions of named exposure groups and partition agreement describe external evidence, not precision or recall against transmission links."
        )
        text.append(paragraphs[-1])
        body.append(f"<p>{html.escape(paragraphs[-1])}</p>")

    coverage = pd.DataFrame(
        [
            {
                "artifact": "clusters",
                "configured": cluster_status.get("configured", 0),
                "completed": cluster_status.get("completed", 0),
                "status": cluster_status.get("status", "unknown"),
            },
            {
                "artifact": "trees",
                "configured": tree_status.get("configured", 0),
                "completed": tree_status.get("completed", 0),
                "status": tree_status.get("status", "not_run"),
            },
        ]
    )
    section("Coverage", coverage)

    (directory / "report.md").write_text("\n\n".join(text) + "\n")
    (directory / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>'
        + html.escape(title)
        + "</title><style>body{font:15px system-ui;margin:2em}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}td,th{padding:.4em;border:1px solid #ccc}img{max-width:100%}</style><body>"
        + "\n".join(body)
        + "</body></html>\n"
    )
