#!/usr/bin/env python3
"""
Diagnostics figure generator for manuscript.

Generates a 6-panel figure showing:
- A1/A2: Feature-cell ambiguity (GD vs GD+TD)
- B: Oracle graph precision-recall curve with contamination inset
- C1-C3: TreeCluster threshold performance by method

Usage:
    python evaluation/00_synthetic_diagnostics/plot.py [--output-dir OUTPUT_DIR]

Outputs:
    - diagnostics_figure.pdf
    - diagnostics_figure.svg
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DIAGNOSTICS_ROOT = Path(__file__).parent / "outputs" / "diagnostics"
ENDPOINT_LABELS = {"M0": "M=0", "Mle1": "M≤1", "Mle2": "M≤2"}
PROCESS_LABELS = {"deterministic": "Deterministic", "stochastic": "Stochastic"}

# Colorblind-safe palette
COLORS = {
    "deterministic": "#0072B2",
    "stochastic": "#D55E00",
    "resolution_low": "#56B4E9",
    "resolution_high": "#009E73",
}

# Tree method labels
METHOD_LABELS = {
    "avg_clade": "avg_clade",
    "max_clade": "max_clade",
    "single_linkage": "single_linkage",
}


def load_current_run() -> Path:
    """Load the current run directory from current.json."""
    current_file = DIAGNOSTICS_ROOT / "current.json"
    if not current_file.exists():
        raise FileNotFoundError(f"Current run not found: {current_file}")

    with open(current_file) as f:
        current = json.load(f)

    run_dir = Path(current["run_directory"])
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    return run_dir


def load_data(run_dir: Path) -> dict:
    """Load all required data from the run directory."""
    data = {}

    # Feature ambiguity data
    summary_agg = run_dir / "observations" / "summary_aggregate.csv"
    data["summary_aggregate"] = pd.read_csv(summary_agg)

    # Oracle graph data
    graphs_summary = run_dir / "graphs" / "summary.csv"
    data["graphs"] = pd.read_csv(graphs_summary)

    # Tree data
    trees_summary = run_dir / "trees" / "summary.csv"
    data["trees"] = pd.read_csv(trees_summary)

    return data


def plot_feature_ambiguity(ax: plt.Axes, data: pd.DataFrame, feature_set: str, y_max: float = None):
    """
    Plot feature-cell ambiguity as bar chart.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    data : pd.DataFrame
        summary_aggregate data
    feature_set : str
        'GD' or 'GD_TD'
    y_max : float, optional
        Fixed y-axis maximum for consistent scaling across panels
    """
    # Convert to string for reliable comparison with Arrow arrays
    subset = data[data["feature_set"].astype(str) == feature_set].copy()

    # Calculate error magnitude
    subset = subset.copy()
    subset["error_mag"] = subset["mixed_cell_fraction_max"] - subset["mixed_cell_fraction_mean"]

    endpoints = ["M0", "Mle1", "Mle2"]
    x_pos = range(len(endpoints))
    bar_width = 0.35

    processes = ["deterministic", "stochastic"]
    for i, (process, color) in enumerate(zip(processes, [COLORS["deterministic"], COLORS["stochastic"]])):
        process_data = subset[subset["process"].astype(str) == process]
        y_vals = []
        y_err = []
        for ep in endpoints:
            ep_data = process_data[process_data["endpoint"].astype(str) == ep]
            if len(ep_data) > 0:
                y_vals.append(ep_data["mixed_cell_fraction_mean"].iloc[0])
                y_err.append(ep_data["error_mag"].iloc[0])
            else:
                y_vals.append(0)
                y_err.append(0)

        offset = [-bar_width / 2, bar_width / 2][i]
        ax.bar(
            [p + offset for p in x_pos],
            y_vals,
            bar_width,
            yerr=y_err,
            label=PROCESS_LABELS[process],
            color=color,
            capsize=3,
            error_kw={"elinewidth": 1},
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([ENDPOINT_LABELS[ep] for ep in endpoints], fontsize=9)
    ax.set_ylabel("Mixed cell fraction", fontsize=9)
    ax.set_title(f"{feature_set.replace('_', '+')} only", fontsize=10, fontweight="bold")
    
    # Use provided y_max or calculate from data
    if y_max is not None:
        ax.set_ylim(0, y_max)
    else:
        ax.set_ylim(0, max(subset["mixed_cell_fraction_max"]) * 1.2)
    
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def plot_oracle_pr(ax: plt.Axes, data: pd.DataFrame):
    """
    Plot oracle graph M0 F1, Precision, Recall, and M≥3 contamination vs resolution.

    Parameters
    ----------
    ax : plt.Axes
        Main axes for F1/P/R/contamination vs resolution
    data : pd.DataFrame
        graphs/summary.csv data
    """
    # Filter to Leiden algorithm and M0 endpoint (use astype(str) for Arrow arrays)
    leiden_m0 = data[
        (data["endpoint"].astype(str) == "M0") & (data["algorithm"].astype(str) == "leiden")
    ].copy()

    # Sort by resolution for connected line
    leiden_m0 = leiden_m0.copy()
    leiden_m0["resolution_float"] = leiden_m0["resolution"].astype(float)
    leiden_m0 = leiden_m0.sort_values("resolution_float")

    # Plot F1, Precision, Recall, and Contamination
    ax.plot(
        leiden_m0["resolution_float"],
        leiden_m0["M0_f1_mean"],
        "o-",
        color="#009E73",  # green
        linewidth=2.5,
        markersize=6,
        label="F1",
    )
    ax.plot(
        leiden_m0["resolution_float"],
        leiden_m0["M0_precision_mean"],
        "s-",
        color="#0072B2",  # blue
        linewidth=2,
        markersize=5,
        label="Precision",
        alpha=0.8,
    )
    ax.plot(
        leiden_m0["resolution_float"],
        leiden_m0["M0_recall_mean"],
        "^-",
        color="#D55E00",  # orange
        linewidth=2,
        markersize=5,
        label="Recall",
        alpha=0.8,
    )
    ax.plot(
        leiden_m0["resolution_float"],
        leiden_m0["Mge3_contamination_mean"],
        "d--",
        color="#CC79A7",  # pink/magenta
        linewidth=2,
        markersize=5,
        label="M≥3 contam.",
        alpha=0.7,
    )

    # Error bars for F1
    ax.errorbar(
        leiden_m0["resolution_float"],
        leiden_m0["M0_f1_mean"],
        yerr=leiden_m0["M0_f1_mean"] - leiden_m0["M0_f1_min"],
        fmt="none",
        capsize=3,
        elinewidth=1,
        color="gray",
        alpha=0.7,
    )

    ax.set_xlabel("Leiden Resolution (CPM)", fontsize=9)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title("Oracle Graph Partition", fontsize=10, fontweight="bold")
    ax.set_xlim(0.05, 1.05)  # Start from 0.1 (with small padding)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3, linestyle="--")
    
    # Add legend in center with semi-transparent background
    ax.legend(
        loc="center",
        fontsize=7,
        frameon=True,
        ncol=2,
        facecolor="white",
        edgecolor="gray",
        framealpha=0.9,
    )


def plot_tree_performance(ax: plt.Axes, data: pd.DataFrame, method: str):
    """
    Plot tree threshold performance for a single method.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    data : pd.DataFrame
        trees/summary.csv data
    method : str
        TreeCluster method name
    """
    # Use astype(str) for Arrow array comparison
    method_data = data[data["method"].astype(str) == method].copy()

    # Sort by threshold
    method_data = method_data.sort_values("threshold_hops")

    # Main F1 line
    ax.plot(
        method_data["threshold_hops"],
        method_data["M0_f1_mean"],
        "o-",
        linewidth=2,
        markersize=5,
        color=COLORS["resolution_high"],
    )

    # Error bars (min/max across seeds)
    ax.errorbar(
        method_data["threshold_hops"],
        method_data["M0_f1_mean"],
        yerr=method_data["M0_f1_mean"] - method_data["M0_f1_min"],
        fmt="none",
        capsize=3,
        elinewidth=1,
        color="gray",
        alpha=0.7,
    )

    # Find best F1
    best_idx = method_data["M0_f1_mean"].idxmax()
    best_row = method_data.loc[best_idx]
    best_f1 = best_row["M0_f1_mean"]
    best_contam = best_row["Mge3_contamination_mean"]
    best_thresh = best_row["threshold_hops"]

    ax.set_xlabel("Threshold (hops)", fontsize=9)
    ax.set_ylabel("M0 F1", fontsize=9)
    ax.set_title(METHOD_LABELS.get(method, method), fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")

    # Handle single_linkage special case
    if method == "single_linkage":
        # After hop=1, single_linkage selects all pairs (F1 undefined or very low)
        ax.set_ylim(-0.05, 1.05)
        ax.annotate(
            "Collapses to\nsingle cluster",
            xy=(8, 0.02),
            fontsize=7,
            ha="center",
            alpha=0.7,
        )
    else:
        ax.set_ylim(-0.05, max(method_data["M0_f1_max"]) * 1.15)

    # Set x-ticks to match actual thresholds
    ax.set_xticks(method_data["threshold_hops"].unique())

    # Dynamic annotation placement to avoid overlap with curve
    # Try different positions and pick the one with least curve overlap
    annotation_text = f"Best F1={best_f1:.2f}\n@ hop={best_thresh}\nContam={best_contam:.1%}"
    
    # Candidate positions in axes coordinates
    candidate_positions = [
        (0.98, 0.98, "right", "top"),      # top-right
        (0.98, 0.02, "right", "bottom"),   # bottom-right
        (0.02, 0.98, "left", "top"),       # top-left
        (0.02, 0.02, "left", "bottom"),    # bottom-left
        (0.98, 0.50, "right", "center"),   # middle-right
        (0.02, 0.50, "left", "center"),    # middle-left
    ]
    
    # Get curve data points for overlap checking
    curve_x = method_data["threshold_hops"].values
    curve_y = method_data["M0_f1_mean"].values
    
    best_pos = candidate_positions[0]
    min_overlap = float('inf')
    
    for x_frac, y_frac, ha, va in candidate_positions:
        # Convert to data coordinates
        x_data = ax.get_xlim()[0] + x_frac * (ax.get_xlim()[1] - ax.get_xlim()[0])
        y_data = ax.get_ylim()[0] + y_frac * (ax.get_ylim()[1] - ax.get_ylim()[0])
        
        # Estimate distance from curve (simple: min distance to curve points)
        min_dist = float('inf')
        for cx, cy in zip(curve_x, curve_y):
            dist = abs(x_data - cx) + abs(y_data - cy)  # Manhattan distance
            min_dist = min(min_dist, dist)
        
        # Prefer positions far from curve
        if min_dist > min_overlap:
            min_overlap = min_dist
            best_pos = (x_frac, y_frac, ha, va)
    
    # Place annotation at best position
    x_frac, y_frac, ha, va = best_pos
    ax.annotate(
        annotation_text,
        xy=(best_thresh, best_f1),
        xytext=(x_frac, y_frac),
        textcoords="axes fraction",
        fontsize=7,
        ha=ha,
        va=va,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )


def create_figure(data: dict, output_path: Path):
    """
    Create the 6-panel diagnostic figure.

    Parameters
    ----------
    data : dict
        Loaded data from load_data()
    output_path : Path
        Output path (extension determines format)
    """
    # Create figure with custom layout
    fig = plt.figure(figsize=(10, 12))

    # Define grid: 3 rows, 2 columns with varying heights
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1, 1.2, 0.9],
        width_ratios=[1, 1],
        hspace=0.35,
        wspace=0.25,
        left=0.07,
        right=0.95,
        top=0.95,
        bottom=0.05,
    )

    # Calculate common y_max for A1 and A2 to ensure same scale
    summary_agg = data["summary_aggregate"]
    gd_max = summary_agg[summary_agg["feature_set"].astype(str) == "GD"]["mixed_cell_fraction_max"].max()
    gd_td_max = summary_agg[summary_agg["feature_set"].astype(str) == "GD_TD"]["mixed_cell_fraction_max"].max()
    y_max_ambiguity = max(gd_max, gd_td_max) * 1.2

    # Panel A1: GD ambiguity (row 0, col 0)
    ax_a1 = fig.add_subplot(gs[0, 0])
    plot_feature_ambiguity(ax_a1, data["summary_aggregate"], "GD", y_max_ambiguity)

    # Panel A2: GD_TD ambiguity (row 0, col 1)
    ax_a2 = fig.add_subplot(gs[0, 1])
    plot_feature_ambiguity(ax_a2, data["summary_aggregate"], "GD_TD", y_max_ambiguity)

    # Panel B: Oracle P-R curve (row 1, spans both columns)
    ax_b = fig.add_subplot(gs[1, :])
    plot_oracle_pr(ax_b, data["graphs"])

    # Panel C1-C3: Tree methods (row 2)
    tree_methods = ["avg_clade", "max_clade", "single_linkage"]

    # For tree plots, we need narrower individual axes
    # Create a nested grid for row 2
    gs_row2 = gs[2, :].subgridspec(1, 3, wspace=0.3)

    for i, method in enumerate(tree_methods):
        ax_c = fig.add_subplot(gs_row2[0, i])
        plot_tree_performance(ax_c, data["trees"], method)

    # Add panel labels
    panel_labels = [("A1", 0.02, 0.98), ("A2", 0.52, 0.98), ("B", 0.02, 0.70)]
    for label, x, y in panel_labels:
        fig.text(x, y, f"{label}", fontsize=12, fontweight="bold", va="top")

    # Tree panel labels
    for i, label in enumerate(["C1", "C2", "C3"]):
        x_pos = [0.02, 0.35, 0.68][i]
        fig.text(x_pos, 0.28, f"{label}", fontsize=12, fontweight="bold", va="top")

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".pdf":
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
    elif output_path.suffix.lower() == ".svg":
        fig.savefig(output_path, format="svg", bbox_inches="tight")
    else:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)

    plt.close(fig)
    print(f"Figure saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate diagnostics manuscript figure"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DIAGNOSTICS_ROOT / "figures",
        help="Output directory for figures (default: diagnostics/outputs/figures)",
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "svg", "both"],
        default="both",
        help="Output format (default: both)",
    )

    args = parser.parse_args()

    # Load data
    print("Loading current run...")
    run_dir = load_current_run()
    print(f"Using run: {run_dir.name}")

    print("Loading data...")
    data = load_data(run_dir)

    # Generate figures
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format in ["pdf", "both"]:
        create_figure(data, output_dir / "diagnostics_figure.pdf")

    if args.format in ["svg", "both"]:
        create_figure(data, output_dir / "diagnostics_figure.svg")

    print("Done!")


if __name__ == "__main__":
    main()
