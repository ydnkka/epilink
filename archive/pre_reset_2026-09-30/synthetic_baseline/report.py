"""Unified report generator for comprehensive baseline assessment."""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from synthetic_exploration.common import log


FAMILY_COLORS = {
    "compatibility": "#007f86",
    "genetic_only": "#8b939d",
    "logistic_probability": "#e8ae49",
    "oracle": "#006b73",
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


def md_table(frame: pd.DataFrame, max_rows: int = 15) -> str:
    if frame.empty:
        return "No data."
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
    
    # Figure 1: Pairwise AP comparison
    ranking = read_csv(directory / "02_pairwise" / "ranking_summary.csv")
    if not ranking.empty:
        endpoints = ["M0", "Mle1", "Mle2"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True, sharex=True)
        for ax, endpoint in zip(axes, endpoints):
            group = ranking.loc[ranking.endpoint == endpoint].sort_values("ap")
            labels = group.score_name.str.replace("compatibility_", "CS ", regex=False).str.replace("genetic_only_", "GD ", regex=False).str.replace("logistic_", "Logit ", regex=False)
            colors = [FAMILY_COLORS.get(f, "#b5bfcb") for f in group.score_family]
            ax.barh(np.arange(len(group)), group.ap, color=colors)
            ax.set_yticks(np.arange(len(group)), labels, fontsize=8)
            ax.set_title(f"Endpoint: {endpoint}")
            ax.set_xlabel("Average Precision")
            ax.set_xlim(0, min(1, max(0.05, ranking.ap.max() * 1.15)))
        fig.suptitle("Pairwise Ranking Informativeness by Target Horizon")
        export(fig, directory, "01_pairwise_ap")
    
    # Figure 2: Contamination frontier
    operating = read_csv(directory / "02_pairwise" / "operating_points.csv")
    if not operating.empty:
        fraction = float(settings["figures"]["primary_selection_fraction"])
        group = operating.loc[(operating.selection == "top_fraction") & np.isclose(operating.requested, fraction)]
        if not group.empty:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True, sharey=True)
            for ax, endpoint in zip(axes, ["M0", "Mle1", "Mle2"]):
                selected = group.loc[group.endpoint == endpoint]
                for family, rows in selected.groupby("score_family"):
                    ax.scatter(rows.Mge3_contamination_fraction, rows.precision,
                               s=55, alpha=.85, label=family.replace("_", " "),
                               color=FAMILY_COLORS.get(family, "#b5bfcb"))
                ax.set_title(f"Endpoint: {endpoint}")
                ax.set_xlabel("Selected M>=3 Contamination Fraction")
                ax.set_xlim(left=0)
                ax.set_ylim(0, 1.02)
            axes[0].set_ylabel("Target Precision")
            axes[0].legend(frameon=False, fontsize=8)
            fig.suptitle(f"Pairwise Precision vs Contamination at Top {fraction:.2%}")
            export(fig, directory, "02_pairwise_contamination")
    
    # Figure 3: Cluster frontier
    clusters = read_csv(directory / "03_clusters" / "cluster_frontier.csv")
    if not clusters.empty:
        fraction = float(settings["figures"]["primary_cluster_fraction"])
        group = clusters.loc[np.isclose(clusters.requested, fraction)]
        if not group.empty:
            fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
            for family, rows in group.groupby("score_family"):
                ax.scatter(rows.Mge3_contamination_fraction, rows.Mle2_pair_recall,
                           s=70, alpha=.75, label=family.replace("_", " "),
                           color=FAMILY_COLORS.get(family, "#b5bfcb"))
            ax.set(xlabel="Within-Cluster M>=3 Contamination Fraction",
                   ylabel="Recall of All M<=2 Pairs",
                   title=f"Graph Clustering Frontier at Top {fraction:.2%} Retained Edges")
            ax.legend(frameon=False)
            export(fig, directory, "03_cluster_frontier")
    
    # Figure 4: Oracle comparison
    if not clusters.empty and "oracle_target_edges" in clusters.score_name.values:
        oracle = clusters.loc[clusters.score_name == "oracle_target_edges"]
        others = clusters.loc[clusters.score_name != "oracle_target_edges"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        
        # BCubed comparison
        ax = axes[0]
        ax.barh(["Oracle"], [oracle.bcubed_f1.iloc[0]] if len(oracle) else [0], color=FAMILY_COLORS["oracle"])
        ax.set_xlabel("BCubed F1")
        ax.set_title("Structural Ceiling for Cluster Recovery")
        ax.set_xlim(0, 1)
        
        # Recall comparison
        ax = axes[1]
        ax.barh(["Oracle"], [oracle.Mle2_pair_recall.iloc[0]] if len(oracle) else [0], color=FAMILY_COLORS["oracle"])
        ax.set_xlabel("M<=2 Pair Recall")
        ax.set_title("Maximum Achievable Recall (Perfect Pairwise)")
        ax.set_xlim(0, 1)
        
        fig.suptitle("Oracle Target-Edge Graph Benchmark")
        export(fig, directory, "04_oracle_ceiling")
    
    # Figure 5: Observation overlap (from stage 01)
    ambiguity = read_csv(directory / "01_observations" / "ambiguity_summary.csv")
    if not ambiguity.empty:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        processes = ambiguity.data_process.tolist()
        fractions = ambiguity.mixed_cell_fraction.tolist()
        ax.bar(processes, fractions, color=[FAMILY_COLORS["genetic_only"], FAMILY_COLORS["compatibility"]])
        ax.set_ylabel("Fraction of Mixed Cells")
        ax.set_title("Observable Ambiguity: Cells with Both Target and Non-Target Pairs")
        ax.set_ylim(0, max(fractions) * 1.5 if max(fractions) < 0.1 else 1)
        export(fig, directory, "05_observation_ambiguity")
    
    # Figure 6: Benchmark comparison (from stage 04)
    benchmark = read_csv(directory / "04_validation" / "benchmark_ranking.csv")
    epilink = read_csv(directory / "02_pairwise" / "ranking_summary.csv")
    if not benchmark.empty and not epilink.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True, sharex=True)
        for ax, process in zip(axes, ["deterministic", "stochastic"]):
            bench_selected = benchmark.loc[benchmark.data_process == process]
            epi_selected = epilink.loc[(epilink.endpoint == "M0") & (epilink.data_process == process)]
            
            names = epi_selected.score_name.tolist() + bench_selected.model.tolist()
            aps = epi_selected.ap.tolist() + bench_selected.ap.tolist()
            colors = [FAMILY_COLORS.get("compatibility", "#007f86")] * len(epi_selected) + \
                     [FAMILY_COLORS.get("logistic_probability", "#e8ae49")] * len(bench_selected)
            
            ax.barh(np.arange(len(names)), aps, color=colors)
            ax.set_yticks(np.arange(len(names)), [n.replace("_", " ") for n in names], fontsize=8)
            ax.set_xlabel("Average Precision (M=0)")
            ax.set_title(f"{process.capitalize()} Genetics")
            ax.set_xlim(0, 1)
        fig.suptitle("Benchmark Comparison: EpiLink vs Independent Logistic")
        export(fig, directory, "06_benchmark_comparison")


def make_report(directory, pairs, cases, settings) -> None:
    log("Rendering comprehensive baseline report")
    directory = Path(directory)
    make_figures(directory, settings)
    
    # Load all tables
    rel_prev = read_csv(directory / "01_observations" / "relationship_prevalence.csv")
    ambig = read_csv(directory / "01_observations" / "ambiguity_summary.csv")
    ranking = read_csv(directory / "02_pairwise" / "ranking_summary.csv")
    operating = read_csv(directory / "02_pairwise" / "operating_points.csv")
    clusters = read_csv(directory / "03_clusters" / "cluster_frontier.csv")
    oracle = clusters.loc[clusters.score_name == "oracle_target_edges"] if not clusters.empty else pd.DataFrame()
    nulls = read_csv(directory / "03_clusters" / "null_replicates.csv")
    validation = read_csv(directory / "04_validation" / "benchmark_ranking.csv")
    
    integrity_path = directory / "04_validation" / "integrity_checks.json"
    integrity = json.loads(integrity_path.read_text()) if integrity_path.exists() else {}
    
    sections = []
    
    # Section 1: Run and Scope
    sections.append(("Run and Scope",
        f"This comprehensive baseline assessment evaluates {len(pairs):,} unordered pairs among {int(cases.sampled.sum()):,} sampled cases "
        f"under the matched baseline (seed {settings['seed']}, training seed {settings['training_seed']}). "
        f"There are {cases.root_index.nunique()} transmission component(s). "
        f"The immediate target (M=0) prevalence is {pairs.IsRelated.mean():.4%}. "
        f"Historical reproduction status: {integrity.get('historical_comparison', 'not evaluated')}.",
        None, ()))
    
    # Section 2: Observable Ambiguity
    sections.append(("1. Observable Ambiguity",
        "Pairs with identical observed (genetic distance, time difference) cannot be distinguished by any scorer using only those inputs. "
        "Mixed cells contain both target (M=0) and non-target pairs, establishing an identifiability ceiling independent of scoring method.",
        ambig[["data_process", "n_cells", "mixed_cells", "mixed_cell_fraction", "target_fraction_in_mixed_cells"]] if not ambig.empty else None,
        ("05_observation_ambiguity",)))
    
    # Section 3: Relationship Prevalence
    sections.append(("2. Relationship Prevalence (8 Categories)",
        "The 8 relationship categories distinguish AD(0), AD(1), AD(2), AD(>=3), CA(0,0), CA(0,1), CA(1,1), and CA(m1+m2>=3). "
        "This fine-grained breakdown reveals cluster composition beyond binary M-horizon endpoints.",
        rel_prev, ()))
    
    # Section 4: Pairwise Ranking
    sections.append(("3. Pairwise Ranking Informativeness",
        "Compatibility scores (CS), genetic-only distance (-GD), and logistic probabilities are compared for each near-transmission endpoint. "
        "Logistic probabilities are trained on a separate date/genome realization on the same transmission tree.",
        ranking.sort_values("ap", ascending=False).groupby(["endpoint", "score_family"], as_index=False).head(1)[[
            "endpoint", "score_family", "score_name", "data_process", "ap", "target_prevalence"]] if not ranking.empty else None,
        ("01_pairwise_ap", "02_pairwise_contamination")))
    
    # Section 5: Operating Points
    primary_operating = pd.DataFrame()
    if not operating.empty:
        primary_operating = operating.loc[
            (operating.selection == "target_recall") & np.isclose(operating.requested, 0.7)
        ].sort_values(["endpoint", "precision"], ascending=[True, False])[[
            "endpoint", "score_name", "precision", "recall", "selected_pairs",
            "Mge3_contamination_fraction", "median_M_connected", "p90_M_connected"]]
    sections.append(("4. Operating Points (70% Target Recall)",
        "The table shows the 70% target-recall operating point where available. M>=3 contamination is the fraction of selected pairs with >=3 intermediates.",
        primary_operating, ()))
    
    # Section 6: Cluster Structure
    primary_clusters = pd.DataFrame()
    if not clusters.empty:
        fraction = float(settings["figures"]["primary_cluster_fraction"])
        primary_clusters = clusters.loc[np.isclose(clusters.requested, fraction)].sort_values(
            ["Mge3_contamination_fraction", "Mle2_pair_recall"], ascending=[True, False])[[
                "score_name", "clustering_method", "algorithm", "Mle2_pair_recall", "Mle2_pair_precision",
                "Mge3_contamination_fraction", "n_clusters", "bcubed_f1"]]
    sections.append(("5. Cluster Structure",
        "Connected components and Leiden communities are built from top-score pairwise graphs. "
        "Every pair sharing a cluster is evaluated, not only retained graph edges.",
        primary_clusters, ("03_cluster_frontier",)))
    
    # Section 7: Oracle Ceiling
    oracle_table = pd.DataFrame()
    if not oracle.empty:
        oracle_table = oracle[["resolution", "n_clusters", "n_singletons", "Mle2_pair_precision", "Mle2_pair_recall", "bcubed_f1"]]
    sections.append(("6. Oracle Target-Edge Graph (Structural Ceiling)",
        "Leiden clustering on the graph containing only true M=0 edges reveals the structural ceiling for any partition-based method. "
        "Perfect pairwise information cannot overcome the non-transitivity of direct/shared-infector relationships.",
        oracle_table, ("04_oracle_ceiling",)))
    
    # Section 8: Null Baselines
    null_summary = pd.DataFrame()
    if not nulls.empty:
        null_summary = nulls.groupby("null_kind")[["target_edge_retention", "Mle2_pair_precision", "bcubed_f1"]].agg(["mean", "std"]).round(4)
    sections.append(("7. Null Baselines",
        "Size-preserving and size+time-preserving randomizations test whether observed cluster composition exceeds chance structure.",
        null_summary, ()))
    
    # Section 9: Validation
    sections.append(("8. Validation and Independent Benchmarks",
        "Historical score reproduction confirms evaluation machinery integrity. "
        "Independent-observation logistic/lookup benchmarks test whether joint (GD, TD) information exceeds raw compatibility.",
        validation[["data_process", "model", "ap", "training"]] if not validation.empty else None,
        ("06_benchmark_comparison",)))
    
    # Section 10: Interpretation
    sections.append(("Interpretation Guidelines",
        "A score is informative when it recovers near-transmission pairs with less M>=3 contamination than genetic distance alone. "
        "High enrichment can coexist with modest absolute precision. "
        "The oracle ceiling quantifies the maximum achievable cluster recovery given the non-transitive nature of transmission relationships.",
        None, ()))
    
    # Generate Markdown
    lines = ["# Synthetic Baseline Assessment", "", f"*Matched baseline · seed {settings['seed']} · generated automatically*"]
    html_parts = [f"<h1>Synthetic Baseline Assessment</h1><p class='subtitle'>Comprehensive evaluation · seed {settings['seed']}</p>"]
    
    for title, text, table, figures in sections:
        lines.extend([f"\n## {title}", "", text, ""])
        html_parts.append(f"<section><h2>{html.escape(title)}</h2><p>{html.escape(text)}</p>")
        if table is not None and len(table):
            lines.extend([md_table(table), ""])
            html_parts.append("<div class='table'>" + table.head(15).to_html(index=False, float_format=lambda x: f"{x:.4g}", border=0) + "</div>")
        for figure in figures:
            lines.extend([f"![{figure}](figures/{figure}.png)", ""])
            html_parts.append(f"<img src='figures/{figure}.png' alt='{html.escape(figure)}'>")
        html_parts.append("</section>")
    
    (directory / "report.md").write_text("\n".join(lines))
    
    css = """body{margin:0;background:#f4f6f8;color:#243346;font:16px/1.65 system-ui,sans-serif}
    main{max-width:1200px;margin:50px auto;padding:36px 48px;background:white;border-radius:14px}
    h1{font-size:36px;line-height:1.15;color:#006b73}h2{font-size:24px;margin-top:42px}
    .subtitle{color:#697888}img{width:100%;margin:24px 0}.table{overflow:auto}
    table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:9px;border-bottom:1px solid #dde4e9;text-align:left}
    th{background:#edf5f5}section{border-top:1px solid #e7ecef;margin-top:32px}
    @media(max-width:700px){main{margin:0;padding:20px}h1{font-size:28px}}"""
    
    (directory / "report.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Synthetic Baseline Assessment</title><style>{css}</style><main>" + "\n".join(html_parts) + "</main></html>")
