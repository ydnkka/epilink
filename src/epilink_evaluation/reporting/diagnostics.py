"""Diagnostics reports rendered exclusively from saved tables and checkpoints."""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from ..provenance import read_json
from ..schemas import ENDPOINTS
from .report import markdown_table, read_table


def render_report(directory):
    """Render a complete or partial saved run, without loading an experiment."""
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    title = (
        "Synthetic diagnostics — smoke validation"
        if manifest["config"]["inputs"].get("smoke_cases") is not None
        else "Synthetic diagnostics"
    )
    notes = [
        f"Status: {manifest['status']}; requested stage: {manifest['requested_stage']}.",
        f"Experiment: {manifest['experiment']['fingerprint']}. Development seeds: "
        f"{manifest['config']['splits']['development']}.",
        "Feature ambiguity uses exact saved GD or (GD, TD) cells, without binning. "
        "The minimum feature-only classification error is empirical for decisions constant "
        "within these cells; it is not a population ceiling or a bound on partition recovery.",
        "Summaries give each development seed equal weight. Min/max describe observation "
        "variation on one fixed backbone. Dependent pairs are not independent replicates; "
        "no pair-based confidence intervals are constructed. Undefined ratios remain undefined.",
        "Oracle graphs connect exactly the target pairs for each endpoint. Their edges need "
        "not be transitive: partitions are evaluated on every within-cluster pair, including "
        "nonedges. These controls are not an absolute performance ceiling. Leiden restarts "
        "are selected by the algorithm objective, never by truth-based performance.",
        "The sampled-tip tree preserves known transmission topology, with unit transmission "
        "edges and zero-length sampled tips. It is a transmission-hop control, not a molecular "
        "genealogy. M is a relationship horizon and differs from total transmission hops. "
        "Reference memberships are reconstructed from the full transmission tree and restricted "
        "to observed cases; unsampled intermediates are retained.",
        "Oracle controls are deduplicated by truth and canonical sampled-case set across "
        "observation seeds and both distance processes. Repeated full-sampling controls "
        "therefore represent the same graph/tree, not independent oracle replicates.",
    ]
    if "error" in manifest:
        notes.append(f"Run error: {manifest['error']}")
    text = [f"# {title}", *notes]
    body = [f"<h1>{title}</h1>", *[f"<p>{html.escape(n)}</p>" for n in notes]]

    def section(heading, table):
        text.extend([f"## {heading}", markdown_table(table)])
        body.extend([f"<h2>{html.escape(heading)}</h2>",
                     table.to_html(index=False, float_format=lambda x: f"{x:.4g}")
                     if not table.empty else "<p>No completed results.</p>"])

    def figure(fig, name, caption):
        output = directory / "figures"
        output.mkdir(exist_ok=True)
        fig.tight_layout()
        relative = f"figures/{name}.png"
        fig.savefig(directory / relative, dpi=140)
        plt.close(fig)
        text.extend([f"![{caption}]({relative})", caption])
        body.append(f'<figure><img src="{relative}" alt="{html.escape(caption)}">'
                    f"<figcaption>{html.escape(caption)}</figcaption></figure>")

    coverage, errors, indices = [], [], {}
    for stage in ("observations", "graphs", "trees"):
        path = directory / stage / "index.json"
        if path.exists():
            index = indices[stage] = read_json(path)
            coverage.append({"stage": stage, "status": index["status"],
                             "seed_setting_rows": len(index["records"]),
                             "unique_artifacts": len({r["artifact"] for r in index["records"]})})
            errors.extend({"stage": stage, **error} for error in index["errors"])
        else:
            coverage.append({"stage": stage, "status": "not run"})
    if not manifest["config"]["diagnostics"]["treecluster"]["enabled"]:
        coverage[-1]["status"] = "disabled by configuration"
    section("Coverage", pd.DataFrame(coverage))
    if errors:
        section("Visible failures — incomplete coverage", pd.DataFrame(errors))

    summary = read_table(directory / "observations/summary_aggregate.csv")
    if not summary.empty:
        columns = ["process", "feature_set", "endpoint", "n_seeds", "mixed_cell_fraction_mean",
                   "pair_fraction_in_mixed_cells_mean", "target_fraction_in_mixed_cells_mean",
                   "target_prevalence_in_mixed_cells_mean", "class_conditional_overlap_mean",
                   "minimum_feature_only_misclassification_rate_mean",
                   "minimum_feature_only_misclassification_rate_min",
                   "minimum_feature_only_misclassification_rate_max"]
        section("Exact feature-cell ambiguity across seeds", summary[columns])
        section("Endpoint prevalence across seeds", read_table(directory / "observations/prevalence_aggregate.csv"))
        section("Relationship composition across seeds", read_table(directory / "observations/relationships_aggregate.csv"))
    examples = indices.get("observations", {}).get("records", [])
    if examples:
        example = min(examples, key=lambda r: r["seed"])
        cells = pd.read_parquet(Path(example["artifact"]) / "cells.parquet")
        cells = cells.loc[cells.feature_set.eq("GD_TD")]
        for process, group in cells.groupby("process", sort=True):
            fig, axes = plt.subplots(2, len(ENDPOINTS), figsize=(15, 8), squeeze=False)
            for col, endpoint in enumerate(ENDPOINTS):
                subset = group.loc[group.endpoint.eq(endpoint)]
                for row, value in enumerate(("target_fraction", "n_pairs")):
                    axis = axes[row, col]
                    # Sparse cell heatmaps preserve exact coordinates without allocating
                    # the potentially enormous Cartesian product of GD and TD values.
                    options = {"vmin": 0, "vmax": 1, "cmap": "viridis"} if row == 0 else {
                        "norm": LogNorm(vmin=1, vmax=max(2, subset.n_pairs.max())), "cmap": "magma",
                    }
                    image = axis.scatter(subset.GD, subset.TD, c=subset[value], marker="s", s=22, **options)
                    axis.set(xlabel="Exact GD (substitutions)", ylabel="Exact TD (days)", title=endpoint)
                    fig.colorbar(image, ax=axis, label="Target fraction" if row == 0 else "Cell pair occupancy (log color)")
            caption = (f"{process}, development seed {example['seed']} — labelled single-realization "
                       "GD_TD example. Squares show occupied exact cells; unoccupied cells are blank. "
                       "Target fraction (top) and occupancy (bottom); across-seed summaries are above.")
            figure(fig, f"feature_cells_{process}", caption)

    graphs = read_table(directory / "graphs/summary.csv")
    if not graphs.empty:
        section("Oracle graph structure across seeds", read_table(directory / "graphs/graph_summary_aggregate.csv"))
        fig, axes = plt.subplots(1, len(ENDPOINTS), figsize=(15, 4.5), squeeze=False)
        for axis, endpoint in zip(axes.flat, ENDPOINTS):
            group = graphs.loc[graphs.endpoint.eq(endpoint)]
            for algorithm, points in group.groupby("algorithm", sort=True):
                points = points.sort_values("resolution")
                axis.plot(points[f"{endpoint}_recall_mean"], points[f"{endpoint}_precision_mean"],
                          "o-" if algorithm == "leiden" else "s", label=algorithm)
                if algorithm == "leiden":
                    for _, point in points.iterrows():
                        x, y = point[f"{endpoint}_recall_mean"], point[f"{endpoint}_precision_mean"]
                        if np.isfinite(x) and np.isfinite(y):
                            axis.annotate(f"{point.resolution:g}", (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
            axis.set(title=endpoint, xlabel="Mean within-pair recall", ylabel="Mean within-pair precision",
                     xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
            axis.legend()
        figure(fig, "oracle_graph_precision_recall", "Endpoint-oracle partition trade-offs, equal seed weights. Leiden points are labelled by resolution and connected in resolution order.")

    trees = read_table(directory / "trees/summary.csv")
    if not trees.empty:
        fig, axes = plt.subplots(2, len(ENDPOINTS), figsize=(15, 8), squeeze=False)
        for col, endpoint in enumerate(ENDPOINTS):
            for row, metric in enumerate(("precision", "recall")):
                axis = axes[row, col]
                for method, points in trees.groupby("method", sort=True):
                    points = points.sort_values("threshold_hops")
                    axis.plot(points.threshold_hops, points[f"{endpoint}_{metric}_mean"], "o-", label=method)
                axis.set(title=endpoint, xlabel="TreeCluster threshold (transmission hops)",
                         ylabel=f"Mean within-pair {metric}", ylim=(-0.02, 1.02))
                axis.legend(fontsize=8)
        figure(fig, "transmission_hop_thresholds", "Known-transmission-hop tree controls by endpoint and method; equal development-seed weights, all configured thresholds.")

    links = [f"[{stage}/{name}]({stage}/{name})"
             for stage, names in {
                 "observations": ("summary.csv", "summary_aggregate.csv", "prevalence.csv", "relationships.csv"),
                 "graphs": ("metrics.csv", "summary.csv", "graph_summary.csv"),
                 "trees": ("metrics.csv", "summary.csv"),
             }.items() for name in names if (directory / stage / name).exists()]
    text.extend(["## Saved tables", "\n\n".join(links)])
    body.append("<h2>Saved tables</h2><ul>" + "".join(
        f'<li><a href="{stage}/{name}">{stage}/{name}</a></li>'
        for stage in ("observations", "graphs", "trees")
        for name in ("index.json", "metrics.csv", "summary.csv")
        if (directory / stage / name).exists()
    ) + "</ul>")
    (directory / "report.md").write_text("\n\n".join(text) + "\n")
    (directory / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>' + title
        + '</title><style>body{font:15px system-ui;margin:2em}table{border-collapse:collapse;'
        'font-size:12px;display:block;overflow:auto}td,th{padding:.4em;border:1px solid #ccc}'
        'img{max-width:100%}</style><body>' + "\n".join(body) + "</body></html>\n"
    )
