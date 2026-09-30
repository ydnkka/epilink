"""Static scientific figures and a readable report generated from saved tables."""
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

from .common import log
from .truth import RELATIONSHIPS
from .score_distribution import make_M_score_distributions

COLORS = ["#007f86", "#70b6ae", "#e8ae49", "#aa7899", "#8b939d"]
LABELS = ["AD(0): direct", "CA(0,0): shared infector", "AD(m>0)", "Other CA(m1,m2)", "Separate trees"]


def read(directory, stage, name):
    return pd.read_csv(directory / stage / f"{name}.csv")


def export(fig, directory, name):
    path = directory / "figures"
    path.mkdir(exist_ok=True)
    fig.savefig(path / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def composition(ax, frame, groups, column, value):
    bottom = np.zeros(len(groups))
    for relation, label, color in zip(RELATIONSHIPS, LABELS, COLORS):
        values = frame.loc[frame.relationship == relation].set_index(column)[value].reindex(groups, fill_value=0).to_numpy()
        ax.bar(np.arange(len(groups)), values, bottom=bottom, label=label, color=color, width=.72)
        bottom += values
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of pairs")


def make_figures(directory, settings):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.dpi": 110})
    models = settings["models"]
    max_t, max_g = settings["figures"]["max_time_days"], settings["figures"]["max_genetic_distance"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), constrained_layout=True)
    for row, process in enumerate(["deterministic", "stochastic"]):
        cells = read(directory, "01_observations", f"{process}_feature_cells")
        for col, (variable, title) in enumerate([
            ("n_target", "Distribution among target pairs"),
            ("n_other", "Distribution among other pairs"),
            ("target_fraction", "Target fraction at the same inputs"),
        ]):
            temp = cells.copy()
            if col < 2:
                temp[variable] = temp[variable] / temp[variable].sum()
            grid = temp.pivot(index="genetic_distance", columns="time_days", values=variable).reindex(
                index=np.arange(max_g + 1), columns=np.arange(max_t + 1))
            image = axes[row, col].imshow(grid, origin="lower", aspect="auto", extent=[-.5, max_t+.5, -.5, max_g+.5],
                cmap="viridis", norm=LogNorm(vmin=1e-6, vmax=.1) if col < 2 else None,
                vmin=0 if col == 2 else None, vmax=1 if col == 2 else None)
            axes[row, col].set(title=f"{process.capitalize()} genetics\n{title}", xlabel="Absolute sample-time difference (days)", ylabel="Genetic distance")
            fig.colorbar(image, ax=axes[row, col], shrink=.75)
    fig.suptitle("1 · Different relationships can produce identical observed inputs", fontsize=15)
    export(fig, directory, "01_observation_overlap")

    bands = read(directory, "02_scores", "score_bands")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for model, ax in zip(models, axes.flat):
        group = bands.loc[bands.model == model]
        ids = sorted(group.bin_id.unique())
        composition(ax, group, ids, "bin_id", "proportion")
        labels = [f"{group.loc[group.bin_id==i, 'lower_inclusive'].iloc[0]:g}–\n{group.loc[group.bin_id==i, 'upper_exclusive'].iloc[0]:g}" for i in ids]
        ax.set_xticks(np.arange(len(ids)), labels, fontsize=8)
        ax.set(title=model, xlabel="Raw compatibility score band")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    fig.suptitle("2 · True relationships across compatibility score bands", fontsize=15)
    export(fig, directory, "02_score_bands")

    selections = read(directory, "02_scores", "selection_relationships")
    group = selections.loc[(selections.selection == "target_recall") & np.isclose(selections.requested, .7)]
    fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    composition(ax, group, models, "model", "proportion")
    ax.set_title("Relationships among pairs selected at ≥70% target recall")
    ax.legend(loc="outside lower center" if False else "upper left", bbox_to_anchor=(1, 1), frameon=False)
    export(fig, directory, "02_selected_relationships")
    make_M_score_distributions(directory, settings)

    partitions = read(directory, "03_clusters", "partition_summary")
    cluster_comp = read(directory, "03_clusters", "relationship_composition")
    primary = settings["clusters"]["primary_resolution"]
    selected = cluster_comp.loc[np.isclose(cluster_comp.resolution, primary)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    names = models + ["true_target_graph"]
    composition(axes[0], selected, names, "model", "pair_weighted_proportion")
    axes[0].set_xticks(np.arange(len(names)), models + ["True-edge\ngraph"])
    axes[0].set_title(f"All within-cluster pairs · resolution {primary}")
    for name in names:
        rows = partitions.loc[partitions.model == name].sort_values("resolution")
        axes[1].plot(rows.direct_edge_retention, rows.target_pair_precision, "o-", label=name.replace("true_target_graph", "True-edge graph"))
        p = rows.loc[np.isclose(rows.resolution, primary)].iloc[0]
        axes[1].scatter([p.direct_edge_retention], [p.target_pair_precision], s=120, facecolors="none", edgecolors="black")
    axes[1].set(xlabel="Fraction of true direct edges kept within clusters", ylabel="Target fraction among all within-cluster pairs",
                xlim=(0, 1.02), ylim=(0, 1.02), title="Compactness–fragmentation trade-off\nOutlined points: fixed primary resolution")
    axes[1].legend(fontsize=8, frameon=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    export(fig, directory, "03_cluster_composition")

    nulls = read(directory, "03_clusters", "null_replicates")
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for i, name in enumerate(names):
        actual = partitions.loc[(partitions.model == name) & np.isclose(partitions.resolution, primary), "target_pair_precision"].iloc[0]
        ax.scatter(i-.18, actual, marker="D", color="#007f86", label="Observed" if i == 0 else None)
        for offset, kind, color, label in [(.02, "size_preserving", "#aa7899", "Size-preserving null"),
                                         (.20, "size_and_time_preserving", "#e8ae49", "Size + time-preserving null")]:
            values = nulls.loc[(nulls.model == name) & (nulls.null_kind == kind), "target_pair_precision"]
            lo, mid, hi = values.quantile([.025, .5, .975])
            ax.errorbar(i+offset, mid, yerr=[[mid-lo], [hi-mid]], fmt="o", color=color, capsize=3, label=label if i == 0 else None)
    ax.set_xticks(np.arange(len(names)), models + ["True-edge graph"])
    ax.set(ylabel="Target fraction among within-cluster pairs", title="Cluster composition versus conditional random assignments\nRanges describe null randomisations, not epidemic uncertainty", ylim=(0, 1))
    ax.legend(frameon=False)
    export(fig, directory, "03_cluster_nulls")

    baseline_m = read(directory, "01_observations", "M_prevalence")
    score_m = read(directory, "02_scores", "selection_M")
    cluster_m = read(directory, "03_clusters", "within_cluster_M")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    def m_cdf(ax, rows, label, **kwargs):
        counts = rows.loc[rows.M.notna()].groupby("M").n_pairs.sum().sort_index()
        if counts.sum():
            cumulative = counts.cumsum() / counts.sum()
            if cumulative.index.max() < 15:
                cumulative.loc[15] = 1.0
            ax.step(cumulative.index, cumulative, where="post", label=label, **kwargs)
    for ax in axes:
        m_cdf(ax, baseline_m, "All baseline pairs", color="#6e7781", linestyle="--")
        ax.set(xlabel="M · total intermediates", ylabel="Fraction of connected pairs with M ≤ x",
               xlim=(0, 15), ylim=(0, 1.02))
    for model in models:
        m_cdf(axes[0], score_m.loc[(score_m.model == model) & (score_m.selection == "target_recall") & np.isclose(score_m.requested, .7)], model)
        m_cdf(axes[1], cluster_m.loc[(cluster_m.model == model) & np.isclose(cluster_m.resolution, primary)], model)
    m_cdf(axes[1], cluster_m.loc[(cluster_m.model == "true_target_graph") & np.isclose(cluster_m.resolution, primary)], "True-edge graph")
    axes[0].set_title("Selected pairs at ≥70% target recall")
    axes[1].set_title(f"All within-cluster pairs · resolution {primary}")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=6, fontsize=8, frameon=False)
    fig.suptitle("M = m for AD; M = m1 + m2 for CA · immediate targets have M = 0", fontsize=13)
    export(fig, directory, "03_total_intermediates")

    joint = read(directory, "04_validation", "target_joint_distributions")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for row, relation in enumerate(["direct", "shared_infector"]):
        group = joint.loc[(joint.relationship == relation) & (joint.data_process == "stochastic") & (joint.inference_process == "stochastic")]
        for source, label, color in [("observed_target_pairs", "Actual target pairs", "#007f86"),
                                     ("raw_scorer_draws_rounded", "Cached scorer draws", "#aa7899"),
                                     ("absolute_time_draws_rounded_diagnostic", "Absolute-time diagnostic", "#e8ae49")]:
            selected = group.loc[group.source == source]
            for col, variable in enumerate(["time_days", "genetic_distance"]):
                if col == 1 and source.startswith("absolute_time"):
                    continue
                mass = selected.groupby(variable).n.sum()
                axes[row, col].plot(mass.index, mass / mass.sum(), "o-", ms=3, label=label, color=color)
        axes[row, 0].set(title=f"{relation.replace('_', ' ').title()} · sampling time", xlim=(-20, 30), xlabel="Time difference (days)", ylabel="Probability mass")
        axes[row, 1].set(title=f"{relation.replace('_', ' ').title()} · stochastic genetics", xlim=(-.5, 15), xlabel="Genetic distance", ylabel="Probability mass")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("4 · Matching parameter values does not guarantee matching distributions", fontsize=15)
    export(fig, directory, "04_generator_checks")

    benchmark = read(directory, "04_validation", "benchmark_ranking")
    epilink = read(directory, "02_scores", "ranking_summary")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, process, ep_names in zip(axes, ["deterministic", "stochastic"], [["EDD", "ESD"], ["EDS", "ESS"]]):
        selected = benchmark.loc[benchmark.data_process == process]
        names = ep_names + selected.model.tolist()
        vals = [float(epilink.loc[epilink.model == m, "ap"].iloc[0]) for m in ep_names] + selected.ap.tolist()
        ax.barh(np.arange(len(names)), vals, color=["#007f86"]*2 + ["#b5bfcb"]*2 + ["#e8ae49"]*2)
        ax.set_yticks(np.arange(len(names)), [n.replace("_", " ") for n in names])
        ax.invert_yaxis()
        ax.set(xlabel="Average precision for AD(0) / CA(0,0)", title=f"{process.capitalize()} genetics", xlim=(0, 1))
        for y, value in enumerate(vals):
            ax.text(value+.01, y, f"{value:.3f}", va="center", fontsize=9)
    fig.suptitle("Joint benchmarks train on separate dates/genomes on the same tree", fontsize=14)
    export(fig, directory, "04_benchmarks")


def md_table(frame):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "—" if np.isnan(value) else f"{value:.4g}"
        return str(value)
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
                      "| " + " | ".join(["---"] * len(frame.columns)) + " |",
                      *["| " + " | ".join(cell(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)]])


def make_report(directory, pairs, cases, settings):
    log("Rendering figures and baseline interpretation report")
    make_figures(directory, settings)
    ambiguity = read(directory, "01_observations", "ambiguity_summary")
    selection = read(directory, "02_scores", "selection_summary")
    recall = selection.loc[(selection.selection == "target_recall") & np.isclose(selection.requested, .7)]
    partitions = read(directory, "03_clusters", "partition_summary")
    primary = partitions.loc[np.isclose(partitions.resolution, settings["clusters"]["primary_resolution"])]
    integrity = json.loads((directory / "04_validation/integrity_checks.json").read_text())
    checks = read(directory, "04_validation", "scorer_generator_checks")
    benchmark = read(directory, "04_validation", "benchmark_ranking")
    sections = []

    def section(title, text, table=None, figures=()):
        sections.append((title, text, table, figures))

    section("Run and scope", (
        f"This run evaluates {len(pairs):,} unordered pairs among {int(cases.sampled.sum()):,} sampled cases. "
        f"There are {cases.root_index.nunique()} transmission components. The immediate target prevalence is {pairs.IsRelated.mean():.4%}. "
        "The analysis is exploratory and conditional on this transmission tree. EDD/EDS use deterministic inference with deterministic/stochastic observed genetics; "
        "ESD/ESS use stochastic inference with deterministic/stochastic observed genetics. Raw compatibility scores are not transmission probabilities. "
        f"Historical reproduction status: {integrity['historical_comparison']}."))
    section("1. Observable ambiguity and EpiLink relationships", (
        "CaseID1 and CaseID2 identify the unordered pair in stable tree input order. AD and CA are binary. "
        "For AD, m is the number of intermediate cases; for CA, m1 and m2 are the intermediates from the shared ancestor to CaseID1 and CaseID2. "
        "Inactive counts are null. Direct transmission is AD(0), and a shared infector is CA(0,0). An AD ancestor may be either case. "
        "M combines the active class: M=m for AD and M=m1+m2 for CA. It counts total intermediates and excludes the shared ancestor for CA. "
        "Both immediate targets have M=0. The transmission-edge distance is M+1 for AD and M+2 for CA; those edge counts remain available as tree_hops. "
        "AD, CA, m, m1, and m2 remain available because equal M need not imply the same branch geometry. "
        "CA(a,b) and CA(b,a) are the same relationship: count summaries combine them using m1<=m2. "
        "The pair table retains branch depths aligned with the two case IDs. M is null for pairs from separate trees. "
        "The shared tree table records every pair once, independently of simulation seed and sampling. "
        "Mixed feature cells contain both target and other relationships with exactly the same observed inputs. Their members cannot all be separated by a deterministic rule using only these inputs. "
        "The heatmaps show a limited viewing window; the tables contain the full range. Empirical overlap is conditional on this realization, not a population identifiability theorem."),
        ambiguity[["data_process", "n_cells", "mixed_cells", "target_fraction_in_mixed_cells", "target_non_target_distribution_overlap"]],
        ("01_observation_overlap",))
    section("2. Information in high scores", (
        "Selections include every pair tied at the cutoff; actual selected fractions can exceed the requested budget. "
        "At the operating points below, target recall is at least 70%. Enrichment divides target precision by its prevalence among all sampled pairs. "
        "The broad relationship display is supplemented by exact AD/CA/m/m1/m2/M counts in selection_epilink_encodings.csv, score distributions in score_by_M.csv, and selected counts in selection_M.csv. "
        "The M-versus-score heatmaps show P(M | score band): each occupied column sums to one, weighted by the number of pairs. "
        "Zero scores have a separate column; positive-score bands have width 0.05 unless configured otherwise. "
        "Colour uses a common logarithmic fraction scale, with zero-probability cells white and empty score bands grey. "
        "Lines give the median and 10th/90th percentiles of M, not uncertainty intervals. Separate AD and CA views retain the same scales. "
        "Distance summaries include connected pairs only; separate-introduction fractions are reported explicitly. "
        "A high enrichment and a modest target precision can both be true. Other AD or CA pairs are not silently relabelled as true target pairs."),
        recall[["model", "selected_pairs", "precision", "recall", "target_enrichment", "median_M_connected", "p90_M_connected"]],
        ("02_M_against_score", "02_M_against_score_AD", "02_M_against_score_CA", "02_selected_relationships", "02_score_bands"))
    section("3. What the clusters contain", (
        f"The primary resolution was fixed at {settings['clusters']['primary_resolution']}; all configured resolutions are reported. "
        "Composition includes every pair sharing a cluster, including missing or discarded graph edges. Both pair-weighted and equal non-singleton-cluster-weighted summaries are saved. "
        "The report uses M to describe separation beyond the immediate target. The M curves below pool AD and CA while class-specific counts remain in within_cluster_M.csv. "
        "The curve viewport ends at M=15; its denominators include every connected pair, and complete distributions are saved in the tables. "
        "Single-case clusters contribute to fragmentation but do not receive an invented pair precision. "
        "The true-target graph contains only AD(0) and CA(0,0) edges and uses the same clustering procedure. It is a structural diagnostic, not a certified upper bound. "
        "For A → B → C, a single cluster necessarily includes the non-target pair A–C; splitting necessarily loses a target pair. "
        f"The null comparison uses {settings['clusters']['null_replicates']} randomizations preserving cluster sizes, with a second null also preserving each cluster's counts within {settings['clusters']['temporal_block_days']}-day sampling bins. "
        "The ranges describe random assignments, not confidence intervals across epidemics. BCubed remains a secondary measure against the existing overlapping neighbourhood reference. "
        "Leiden follows the established pipeline's CPM objective and selection of restarts by generalized modularity; its RNG is explicitly seeded."),
        primary[["model", "n_clusters", "n_singletons", "target_pair_precision", "direct_edge_retention", "median_M_connected", "p90_M_connected", "bcubed_f1"]],
        ("03_cluster_composition", "03_total_intermediates", "03_cluster_nulls"))
    section("4. Assumption checks and separate-observation benchmarks", (
        "Cached scorer draws are compared with actual direct and shared-infector pairs. The production scorer receives absolute, rounded time differences, "
        "whereas its cached temporal draws are signed. The absolute-time curve is a diagnostic transformation only; none of the EpiLink scores are changed. "
        "The epidemic sequence simulator and pairwise scenario simulator also construct genetic branch durations differently. "
        "In particular, the epidemic simulator mutates a child sequence from its parent's sampled sequence using the sum of absolute sampling-to-transmission durations; "
        "the AD scenario uses a clipped signed sampling-time difference as its branch duration. "
        "Joint total-variation summaries round time and genetic draws into unit-width cells; raw marginal Wasserstein distances are also saved. "
        "Finite Monte Carlo size, shared cases, discretization, and different simulation constructions all affect these comparisons, so they are diagnostics rather than independent goodness-of-fit tests. "
        "The logistic and smoothed feature-cell lookup benchmarks train on a different realization of dates and genomes on this same tree. "
        "Their predictions do not use evaluation labels, and the lookup falls back to training prevalence for unseen cells. They are practical comparators, not performance ceilings. "
        "The lookup's prior strength is fixed in the exploration configuration. No outcome has been calibrated into a transmission probability here."),
        benchmark[["data_process", "model", "ap", "training"]],
        ("04_generator_checks", "04_benchmarks"))
    section("What remains for confirmation", (
        "This baseline establishes descriptive behavior and checks the evaluation machinery. It does not establish generalization across epidemic topologies, "
        "separation of independent introductions, or uncertainty across epidemics. Before using these observations to choose final success criteria, "
        "freeze the claim and operating rule, then evaluate additional simulation seeds and independently generated trees. "
        "Use the existing condition and scenario axes for matched/mismatched parameter experiments, and keep the observed-genetics and inference-genetics axes separate. "
        "Do not interpret pair-bootstrap intervals, randomization ranges, or the best resolution in this sweep as independent validation. "
        "Sampling experiments must retain hidden intermediates in the full-tree AD/CA encoding. "
        "All machine-readable tables and the run manifest remain next to this report."))

    lines = ["# Synthetic EpiLink exploration", ""]
    html_parts = ["<h1>Synthetic EpiLink exploration</h1><p class='subtitle'>Baseline · relationship-aware evaluation</p>"]
    for title, text, table, figures in sections:
        lines.extend([f"## {title}", "", text, ""])
        html_parts.extend([f"<section><h2>{html.escape(title)}</h2><p>{html.escape(text)}</p>"])
        if table is not None:
            lines.extend([md_table(table), ""])
            html_parts.append("<div class='table'>" + table.to_html(index=False, float_format=lambda x: f"{x:.4g}", border=0) + "</div>")
        for name in figures:
            lines.extend([f"![{name.replace('_', ' ')}](figures/{name}.png)", ""])
            html_parts.append(f"<img src='figures/{name}.png' alt='{html.escape(name)}'>")
        html_parts.append("</section>")
    (directory / "report.md").write_text("\n".join(lines))
    css = """body{margin:0;background:#f3f5f7;color:#243346;font:16px/1.65 system-ui,sans-serif}
    main{max-width:1180px;margin:50px auto;padding:36px 48px;background:white;border-radius:14px}
    h1{font-size:36px;line-height:1.15;color:#006b73}h2{font-size:24px;line-height:1.3;margin-top:42px}
    p{max-width:1000px}.subtitle{color:#697888}img{width:100%;margin:22px 0}.table{overflow:auto}
    table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:9px;border-bottom:1px solid #dde4e9;text-align:left}
    th{background:#edf5f5}section{border-top:1px solid #e7ecef;margin-top:30px}
    @media(max-width:700px){main{margin:0;padding:20px}h1{font-size:28px}}"""
    (directory / "report.html").write_text("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Synthetic EpiLink exploration</title><style>{css}</style><main>" + "\n".join(html_parts) + "</main></html>")
