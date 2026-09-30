"""Static report for matched-baseline informativeness analyses."""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synthetic_exploration.common import log


FAMILY_COLORS = {
    "compatibility": "#007f86",
    "genetic_only": "#8b939d",
    "logistic_probability": "#e8ae49",
}


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def export(fig, directory: Path, name: str) -> None:
    figure_dir = directory / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def md_table(frame: pd.DataFrame, max_rows: int = 12) -> str:
    if frame.empty:
        return "No rows."
    frame = frame.head(max_rows).copy()
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "NA" if np.isnan(value) else f"{value:.4g}"
        return str(value)
    return "\n".join([
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        *["| " + " | ".join(cell(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)],
    ])


def make_figures(directory: Path, settings: dict) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    ranking = read_csv(directory / "01_pairwise" / "ranking_summary.csv")
    operating = read_csv(directory / "01_pairwise" / "operating_points.csv")
    clusters = read_csv(directory / "02_clusters" / "cluster_frontier.csv")
    treecluster = read_csv(directory / "03_treecluster" / "treecluster_partition_summary.csv")

    if not ranking.empty:
        endpoints = ["M0", "Mle1", "Mle2"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True, sharex=True)
        for ax, endpoint in zip(axes, endpoints):
            group = ranking.loc[ranking.endpoint == endpoint].sort_values("ap")
            labels = group.score_name.str.replace("compatibility_", "CS ", regex=False).str.replace("genetic_only_", "GD ", regex=False).str.replace("logistic_", "Logit ", regex=False)
            colors = [FAMILY_COLORS.get(f, "#b5bfcb") for f in group.score_family]
            ax.barh(np.arange(len(group)), group.ap, color=colors)
            ax.set_yticks(np.arange(len(group)), labels, fontsize=8)
            ax.set_title(endpoint)
            ax.set_xlabel("Average precision")
            ax.set_xlim(0, min(1, max(.05, ranking.ap.max() * 1.15)))
        fig.suptitle("Pairwise ranking informativeness by target horizon")
        export(fig, directory, "01_pairwise_ap")

    if not operating.empty:
        fraction = float(settings["figures"]["primary_selection_fraction"])
        group = operating.loc[(operating.selection == "top_fraction") & np.isclose(operating.requested, fraction)]
        if not group.empty:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True, sharey=True)
            for ax, endpoint in zip(axes, ["M0", "Mle1", "Mle2"]):
                selected = group.loc[group.endpoint == endpoint]
                for family, rows in selected.groupby("score_family"):
                    ax.scatter(rows.Mge3_contamination_fraction, rows.precision,
                               label=family.replace("_", " "), s=55,
                               color=FAMILY_COLORS.get(family, "#b5bfcb"), alpha=.85)
                ax.set_title(endpoint)
                ax.set_xlabel("Selected M>=3 contamination fraction")
                ax.set_xlim(left=0)
                ax.set_ylim(0, 1.02)
            axes[0].set_ylabel("Target precision")
            axes[0].legend(frameon=False, fontsize=8)
            fig.suptitle(f"Pairwise precision versus contamination at top {fraction:.2%}")
            export(fig, directory, "01_pairwise_contamination")

    if not clusters.empty:
        fraction = float(settings["figures"]["primary_cluster_fraction"])
        group = clusters.loc[np.isclose(clusters.requested, fraction)]
        if not group.empty:
            fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
            for family, rows in group.groupby("score_family"):
                ax.scatter(rows.Mge3_contamination_fraction, rows.Mle2_pair_recall,
                           s=70, alpha=.75, label=family.replace("_", " "),
                           color=FAMILY_COLORS.get(family, "#b5bfcb"))
            ax.set(xlabel="Within-cluster M>=3 contamination fraction",
                   ylabel="Recall of all M<=2 pairs",
                   title=f"Graph clustering frontier at top {fraction:.2%} retained edges")
            ax.legend(frameon=False)
            export(fig, directory, "02_cluster_frontier")

    if not treecluster.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
        for kind, rows in treecluster.groupby("tree_kind"):
            ax.scatter(rows.Mge3_contamination_fraction, rows.Mle2_pair_recall,
                       s=55, alpha=.75, label=kind.replace("_", " "))
        ax.set(xlabel="Within-cluster M>=3 contamination fraction",
               ylabel="Recall of all M<=2 pairs",
               title="TreeCluster frontier")
        ax.legend(frameon=False)
        export(fig, directory, "03_treecluster_frontier")


def make_report(directory, pairs, cases, settings) -> None:
    log("Rendering informativeness report")
    directory = Path(directory)
    make_figures(directory, settings)
    ranking = read_csv(directory / "01_pairwise" / "ranking_summary.csv")
    operating = read_csv(directory / "01_pairwise" / "operating_points.csv")
    clusters = read_csv(directory / "02_clusters" / "cluster_frontier.csv")
    treecluster = read_csv(directory / "03_treecluster" / "treecluster_partition_summary.csv")
    tc_status_path = directory / "03_treecluster" / "treecluster_status.json"
    tc_status = json.loads(tc_status_path.read_text()) if tc_status_path.exists() else {"status": "not_run"}

    pairwise_best = pd.DataFrame()
    if not ranking.empty:
        pairwise_best = ranking.sort_values("ap", ascending=False).groupby(
            ["endpoint", "score_family"], as_index=False).head(1)[[
                "endpoint", "score_family", "score_name", "data_process", "ap", "target_prevalence"]]
    primary_operating = pd.DataFrame()
    if not operating.empty:
        primary_operating = operating.loc[
            (operating.selection == "target_recall") & np.isclose(operating.requested, 0.7)
        ].sort_values(["endpoint", "precision"], ascending=[True, False])[[
            "endpoint", "score_name", "precision", "recall", "selected_pairs",
            "Mge3_contamination_fraction", "median_M_connected", "p90_M_connected"]]
    primary_clusters = pd.DataFrame()
    oracle_row = pd.DataFrame()
    if not clusters.empty:
        fraction = float(settings["figures"]["primary_cluster_fraction"])
        primary_clusters = clusters.loc[np.isclose(clusters.requested, fraction)].sort_values(
            ["Mge3_contamination_fraction", "Mle2_pair_recall"], ascending=[True, False])[[
                "score_name", "algorithm", "Mle2_pair_recall", "Mle2_pair_precision",
                "Mge3_contamination_fraction", "n_clusters", "n_singletons", "bcubed_f1"]]
        oracle_row = clusters.loc[clusters.score_name == "oracle_target_edges"]
        if not oracle_row.empty:
            oracle_row = oracle_row[[
                "score_name", "algorithm", "Mle2_pair_recall", "Mle2_pair_precision",
                "Mge3_contamination_fraction", "n_clusters", "n_singletons", "bcubed_f1"]]
    treecluster_best = pd.DataFrame()
    if not treecluster.empty:
        treecluster_best = treecluster.sort_values(
            ["Mge3_contamination_fraction", "Mle2_pair_recall"], ascending=[True, False])[[
                "tree_kind", "data_process", "method", "threshold", "threshold_days",
                "Mle2_pair_recall", "Mle2_pair_precision", "Mge3_contamination_fraction", "bcubed_f1"]]

    sections = [
        ("Run And Scope",
         f"This fresh analysis evaluates {len(pairs):,} unordered pairs among {int(cases.sampled.sum()):,} sampled cases under the matched baseline seed {settings['seed']}. M>=3 is treated as negative contamination. The positive endpoints are M==0, M<=1, and M<=2. Compatibility scores remain raw summed EpiLink scores, not probabilities.",
         None, ()),
        ("Pairwise Ranking",
         "Compatibility, genetic-only distance, and logistic probabilities are compared separately for each near-transmission endpoint. Logistic probabilities are trained on a separate date/genome realization on the same transmission tree.",
         pairwise_best, ("01_pairwise_ap", "01_pairwise_contamination")),
        ("Operating Points",
         "The table shows the 70% target-recall operating point where available. M>=3 contamination is the selected fraction with at least three total intermediates.",
         primary_operating, ()),
        ("Graph Clusters",
         "Connected components and Leiden communities are built from top-score pairwise graphs. Every pair sharing a cluster is evaluated, not only retained graph edges.",
         primary_clusters, ("02_cluster_frontier",)),
        ("Oracle Target-Edge Graph",
         "Leiden clustering on the graph containing only true M=0 edges reveals the structural ceiling for any method using this partition-based approach. Perfect pairwise information cannot overcome the non-transitivity of direct/shared-infector relationships.",
         oracle_row if not oracle_row.empty else None, ()),
        ("TreeCluster",
         f"TreeCluster status: {tc_status.get('status')}. Raw genetic FastME trees and temporal dated TreeTime trees are evaluated when TreeCluster.py is available.",
         treecluster_best, ("03_treecluster_frontier",) if not treecluster.empty else ()),
        ("Interpretation",
         "A score is informative for this purpose when it recovers M==0, M<=1, or M<=2 pairs with less M>=3 contamination than genetic distance alone. Cluster outputs should be read as compactness-versus-recall trade-offs, with BCubed retained as a secondary reference metric.",
         None, ()),
    ]

    lines = ["# Synthetic Informativeness", ""]
    html_parts = ["<h1>Synthetic Informativeness</h1><p class='subtitle'>Matched baseline · M>=3 contamination analysis</p>"]
    for title, text, table, figures in sections:
        lines.extend([f"## {title}", "", text, ""])
        html_parts.append(f"<section><h2>{html.escape(title)}</h2><p>{html.escape(text)}</p>")
        if table is not None:
            lines.extend([md_table(table), ""])
            html_parts.append("<div class='table'>" + table.head(12).to_html(index=False, float_format=lambda x: f"{x:.4g}", border=0) + "</div>")
        for figure in figures:
            lines.extend([f"![{figure}](figures/{figure}.png)", ""])
            html_parts.append(f"<img src='figures/{figure}.png' alt='{html.escape(figure)}'>")
        html_parts.append("</section>")
    (directory / "report.md").write_text("\n".join(lines))
    css = """body{margin:0;background:#f4f6f8;color:#243346;font:16px/1.65 system-ui,sans-serif}
    main{max-width:1180px;margin:48px auto;padding:34px 46px;background:white;border-radius:14px}
    h1{font-size:36px;line-height:1.15;color:#006b73}h2{font-size:24px;margin-top:40px}
    .subtitle{color:#697888}img{width:100%;margin:22px 0}.table{overflow:auto}
    table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:8px;border-bottom:1px solid #dde4e9;text-align:left}
    th{background:#edf5f5}section{border-top:1px solid #e7ecef;margin-top:30px}
    @media(max-width:700px){main{margin:0;padding:20px}h1{font-size:28px}}"""
    (directory / "report.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Synthetic Informativeness</title><style>{css}</style><main>" + "\n".join(html_parts) + "</main></html>")
