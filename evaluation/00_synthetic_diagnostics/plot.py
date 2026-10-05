"""Generate the synthetic diagnostics manuscript figure."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

DIAGNOSTICS_ROOT = Path(__file__).parent / "outputs" / "diagnostics"

ENDPOINTS = ("M0", "Mle1", "Mle2")
ENDPOINT_LABELS = {"M0": "M=0", "Mle1": "M≤1", "Mle2": "M≤2"}

FEATURE_SETS = ("GD", "GD_TD")
FEATURE_LABELS = {"GD": "GD", "GD_TD": "GD+TD"}

PROCESSES = ("deterministic", "stochastic")
PROCESS_LABELS = {"deterministic": "Deterministic", "stochastic": "Stochastic"}

TREE_METHODS = ("avg_clade", "max_clade", "single_linkage")
METHOD_LABELS = {
    "avg_clade": "Avg Clade",
    "max_clade": "Max Clade",
    "single_linkage": "Single Linkage",
}

COLORS = {
    "deterministic": "#0072B2",
    "stochastic": "#D55E00",
    "f1": "#009E73",
    "precision": "#0072B2",
    "recall": "#D55E00",
    "contamination": "#CC79A7",
}

@dataclass(frozen=True)
class MetricSpec:
    label: str
    suffix: str
    marker: str
    color: str
    linewidth: float
    markersize: float
    linestyle: str = "-"
    alpha: float | None = None

    @property
    def fmt(self) -> str:
        return f"{self.marker}{self.linestyle}"


SCORE_METRICS = (
    MetricSpec("F1", "f1", "o", COLORS["f1"], 1.8, 4),
    MetricSpec("Precision", "precision", "s", COLORS["precision"], 1.5, 3.5, alpha=0.8),
    MetricSpec("Recall", "recall", "^", COLORS["recall"], 1.5, 3.5, alpha=0.8),
    MetricSpec(
        "M≥3 contam.",
        "Mge3_contamination",
        "d",
        COLORS["contamination"],
        1.5,
        3.5,
        linestyle="--",
        alpha=0.7,
    ),
)


def metric_column(endpoint: str, suffix: str) -> str:
    if suffix == "Mge3_contamination":
        return "Mge3_contamination_mean"
    return f"{endpoint}_{suffix}_mean"


def load_current_run() -> Path:
    """Load the current run directory from current.json."""
    current_file = DIAGNOSTICS_ROOT / "current.json"
    if not current_file.exists():
        raise FileNotFoundError(f"Current run not found: {current_file}")

    with current_file.open() as f:
        current = json.load(f)

    run_dir = Path(current["run_directory"])
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    return run_dir


def load_data(run_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all CSV inputs required by the figure."""
    return {
        "summary_aggregate": pd.read_csv(
            run_dir / "observations" / "summary_aggregate.csv"
        ),
        "graphs": pd.read_csv(run_dir / "graphs" / "summary.csv"),
        "trees": pd.read_csv(run_dir / "trees" / "summary.csv"),
    }


def plot_feature_ambiguity(
    ax: Axes,
    data: pd.DataFrame,
    endpoint: str,
    y_max: float | None = 1.0,
) -> None:
    """Plot feature-cell ambiguity for one endpoint."""
    subset = data[data["endpoint"].astype(str) == endpoint].copy()
    subset["error_mag"] = (
        subset["mixed_cell_fraction_max"] - subset["mixed_cell_fraction_mean"]
    )

    x_pos = range(len(FEATURE_SETS))
    bar_width = 0.35

    for i, process in enumerate(PROCESSES):
        process_data = subset[subset["process"].astype(str) == process]
        y_vals = []
        y_err = []

        for feature_set in FEATURE_SETS:
            feature_data = process_data[
                process_data["feature_set"].astype(str) == feature_set
            ]
            if feature_data.empty:
                y_vals.append(0)
                y_err.append(0)
                continue

            y_vals.append(feature_data["mixed_cell_fraction_mean"].iloc[0])
            y_err.append(feature_data["error_mag"].iloc[0])

        offset = (-bar_width / 2, bar_width / 2)[i]
        ax.bar(
            [p + offset for p in x_pos],
            y_vals,
            bar_width,
            yerr=y_err,
            label=PROCESS_LABELS[process],
            color=COLORS[process],
            capsize=3,
            error_kw={"elinewidth": 1},
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([FEATURE_LABELS[feature_set] for feature_set in FEATURE_SETS])
    ax.set_ylim(0, y_max or subset["mixed_cell_fraction_max"].max() * 1.2)


def plot_score_metrics(
    ax: Axes,
    data: pd.DataFrame,
    endpoint: str,
    x_col: str,
    metrics: tuple[MetricSpec, ...],
) -> None:
    for metric in metrics:
        ax.plot(
            data[x_col],
            data[metric_column(endpoint, metric.suffix)],
            metric.fmt,
            color=metric.color,
            linewidth=metric.linewidth,
            markersize=metric.markersize,
            label=metric.label,
            alpha=metric.alpha,
        )


def plot_f1_error_bars(
    ax: Axes,
    data: pd.DataFrame,
    endpoint: str,
    x_col: str,
) -> None:
    f1_mean = f"{endpoint}_f1_mean"
    f1_min = f"{endpoint}_f1_min"
    ax.errorbar(
        data[x_col],
        data[f1_mean],
        yerr=(data[f1_mean] - data[f1_min]).abs(),
        fmt="none",
        capsize=3,
        elinewidth=1,
        color="gray",
        alpha=0.7,
    )


def plot_oracle_pr(ax: Axes, data: pd.DataFrame, endpoint: str) -> None:
    """Plot Leiden oracle F1, precision, recall, and contamination."""
    leiden_data = data[
        (data["endpoint"].astype(str) == endpoint)
        & (data["algorithm"].astype(str) == "leiden")
    ].copy()
    leiden_data["resolution_float"] = leiden_data["resolution"].astype(float)
    leiden_data = leiden_data.sort_values("resolution_float")

    plot_score_metrics(ax, leiden_data, endpoint, "resolution_float", SCORE_METRICS)
    plot_f1_error_bars(ax, leiden_data, endpoint, "resolution_float")

    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)


def plot_tree_performance(
    ax: Axes,
    data: pd.DataFrame,
    method: str,
    endpoint: str,
) -> None:
    """Plot tree threshold performance for one method and endpoint."""
    method_data = data[data["method"].astype(str) == method].copy()
    method_data = method_data.sort_values("threshold_hops")

    plot_score_metrics(ax, method_data, endpoint, "threshold_hops", SCORE_METRICS)
    plot_f1_error_bars(ax, method_data, endpoint, "threshold_hops")

    ax.set_ylim(-0.05, 1.05)


def metric_legend_handles(metrics: tuple[MetricSpec, ...]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=metric.marker,
            color=metric.color,
            linewidth=metric.linewidth,
            markersize=metric.markersize,
            linestyle=metric.linestyle,
            label=metric.label,
            alpha=metric.alpha,
        )
        for metric in metrics
    ]


def add_legends(fig: Figure, top_axes: Sequence[Axes]) -> None:
    """Keep the two distinct color keys above the grid."""
    fig.legend(
        handles=[
            Patch(facecolor=COLORS[process], label=PROCESS_LABELS[process])
            for process in PROCESSES
        ],
        loc="upper left",
        bbox_to_anchor=(top_axes[0].get_position().x0, 0.99),
        ncol=len(PROCESSES),
        fontsize=9,
        frameon=False,
        title="Process",
        title_fontsize=9,
        borderaxespad=0,
    )
    fig.legend(
        handles=metric_legend_handles(SCORE_METRICS),
        loc="upper right",
        bbox_to_anchor=(top_axes[-1].get_position().x1, 0.99),
        ncol=len(SCORE_METRICS),
        fontsize=9,
        frameon=False,
        title="Metrics",
        title_fontsize=9,
        borderaxespad=0,
    )


def add_row_labels(fig: Figure, row_axes: Sequence[Axes]) -> None:
    """Align each row label with its axes rather than fixed figure heights."""
    labels = [
        "Feature\nambiguity",
        "Leiden\nCPM objective",
        *[f"TreeCluster\n{METHOD_LABELS[method]}" for method in TREE_METHODS],
    ]
    for ax, label in zip(row_axes, labels):
        bounds = ax.get_position()
        y_pos = (bounds.y0 + bounds.y1) / 2
        fig.text(bounds.x0 - 0.16, y_pos, label, fontsize=9, va="center")


def add_panel_labels(panel_axes: Sequence[Axes]) -> None:
    """Label individual cells alphabetically in row-major order."""
    for index, ax in enumerate(panel_axes):
        ax.set_title(
            chr(ord("A") + index),
            loc="left",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )


def create_figure(data: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Create and save the diagnostics figure."""
    fig, axes = plt.subplots(
        2 + len(TREE_METHODS),
        len(ENDPOINTS),
        figsize=(12, 11),
        gridspec_kw={
            "hspace": 0.33,
            "wspace": 0.12,
            "left": 0.22,
            "right": 0.98,
            "top": 0.92,
            "bottom": 0.065,
        },
    )

    # Share directly with each group's reference so limits AND tick locators
    # stay identical, including across all three tree-method rows.
    for row, row_axes in enumerate(axes):
        x_reference = axes[min(2, row), 0]
        y_reference = axes[0 if row == 0 else 1, 0]
        for ax in row_axes:
            if ax is not x_reference:
                ax.sharex(x_reference)
            if ax is not y_reference:
                ax.sharey(y_reference)

    summary_agg = data["summary_aggregate"]
    y_max_ambiguity = summary_agg["mixed_cell_fraction_max"].max() * 1.2

    for col, endpoint in enumerate(ENDPOINTS):
        plot_feature_ambiguity(
            axes[0, col],
            summary_agg,
            endpoint,
            y_max_ambiguity,
        )
        plot_oracle_pr(axes[1, col], data["graphs"], endpoint)
        axes[0, col].set_title(
            ENDPOINT_LABELS[endpoint], fontsize=11, fontweight="bold", pad=10
        )

    for row, method in enumerate(TREE_METHODS):
        for col, endpoint in enumerate(ENDPOINTS):
            plot_tree_performance(
                axes[2 + row, col],
                data["trees"],
                method,
                endpoint,
            )

    axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[1, 0].set_yticks([0, 0.5, 1])
    axes[1, 0].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[2, 0].xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

    x_labels = ("Feature set", "Resolution") + (
        "Threshold (hops)",
    ) * len(TREE_METHODS)
    for row, row_axes in enumerate(axes):
        row_axes[0].set_ylabel(
            "Mixed cell fraction" if row == 0 else "Score", fontsize=9
        )
        for col, ax in enumerate(row_axes):
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="0.9", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="both", labelsize=8, length=3)
            ax.tick_params(axis="y", left=col == 0, labelleft=col == 0)
            ax.tick_params(axis="x", bottom=True, labelbottom=True)
            ax.set_xlabel(x_labels[row], fontsize=9)

    fig.align_ylabels(axes[:, 0])
    add_legends(fig, axes[0])
    add_row_labels(fig, axes[:, 0])
    add_panel_labels(axes.ravel())

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Figure saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate diagnostics manuscript figure")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DIAGNOSTICS_ROOT / "figures",
        help="Output directory for figures (default: diagnostics/outputs/figures)",
    )
    parser.add_argument(
        "--format",
        choices=("pdf", "svg", "both"),
        default="both",
        help="Output format (default: both)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading current run...")
    run_dir = load_current_run()
    print(f"Using run: {run_dir.name}")

    print("Loading data...")
    data = load_data(run_dir)

    if args.format in {"pdf", "both"}:
        create_figure(data, args.output_dir / "diagnostics_figure.pdf")

    if args.format in {"svg", "both"}:
        create_figure(data, args.output_dir / "diagnostics_figure.svg")

    print("Done!")


if __name__ == "__main__":
    main()
