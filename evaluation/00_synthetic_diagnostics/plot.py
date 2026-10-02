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
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DIAGNOSTICS_ROOT = Path(__file__).parent / "outputs" / "diagnostics"

ENDPOINTS = ("M0", "Mle1", "Mle2")
ENDPOINT_LABELS = {"M0": "M=0", "Mle1": "M≤1", "Mle2": "M≤2"}

FEATURE_SETS = ("GD", "GD_TD")
FEATURE_LABELS = {"GD": "GD", "GD_TD": "GD+TD"}

PROCESSES = ("deterministic", "stochastic")
PROCESS_LABELS = {"deterministic": "Deterministic", "stochastic": "Stochastic"}

TREE_METHODS = ("avg_clade", "max_clade", "single_linkage")
METHOD_LABELS = {
    "avg_clade": "avg_clade",
    "max_clade": "max_clade",
    "single_linkage": "single_linkage",
}

COLORS = {
    "deterministic": "#0072B2",
    "stochastic": "#D55E00",
    "f1": "#009E73",
    "precision": "#0072B2",
    "recall": "#D55E00",
    "contamination": "#CC79A7",
}

PANEL_LABEL_X = (0.02, 0.27, 0.52)


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
    MetricSpec("F1", "f1", "o", COLORS["f1"], 2.5, 6),
    MetricSpec("Precision", "precision", "s", COLORS["precision"], 2, 5, alpha=0.8),
    MetricSpec("Recall", "recall", "^", COLORS["recall"], 2, 5, alpha=0.8),
    MetricSpec(
        "M≥3 contam.",
        "Mge3_contamination",
        "d",
        COLORS["contamination"],
        2,
        5,
        linestyle="--",
        alpha=0.7,
    ),
)
TREE_METRICS = (SCORE_METRICS[0], SCORE_METRICS[3])


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
    ax.set_ylabel("Mixed cell fraction", fontsize=9)
    ax.set_title(ENDPOINT_LABELS[endpoint], fontsize=10, fontweight="bold")
    ax.set_ylim(0, y_max or subset["mixed_cell_fraction_max"].max() * 1.2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


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

    ax.set_xlabel("Leiden Resolution (CPM)", fontsize=9)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title(f"Oracle: {ENDPOINT_LABELS[endpoint]}", fontsize=10, fontweight="bold")
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3, linestyle="--")


def plot_connected_components(ax: Axes, data: pd.DataFrame, endpoint: str) -> None:
    """Plot connected components F1, precision, recall, and contamination."""
    cc_data = data[
        (data["endpoint"].astype(str) == endpoint)
        & (data["algorithm"].astype(str) == "components")
    ]
    row = cc_data.iloc[0]

    for metric in SCORE_METRICS:
        ax.axhline(
            row[metric_column(endpoint, metric.suffix)],
            color=metric.color,
            linewidth=metric.linewidth,
            linestyle=metric.linestyle,
            label=metric.label,
            alpha=metric.alpha,
        )

    ax.set_ylabel("Score", fontsize=9)
    ax.set_title(
        f"Connected Components: {ENDPOINT_LABELS[endpoint]}",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([])
    ax.grid(alpha=0.3, linestyle="--")


def plot_tree_performance(
    ax: Axes,
    data: pd.DataFrame,
    method: str,
    endpoint: str,
) -> None:
    """Plot tree threshold performance for one method and endpoint."""
    method_data = data[data["method"].astype(str) == method].copy()
    method_data = method_data.sort_values("threshold_hops")

    plot_score_metrics(ax, method_data, endpoint, "threshold_hops", TREE_METRICS)
    plot_f1_error_bars(ax, method_data, endpoint, "threshold_hops")

    ax.set_xlabel("Threshold (hops)", fontsize=9)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title(
        f"{METHOD_LABELS.get(method, method)} - {ENDPOINT_LABELS[endpoint]}",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xticks(method_data["threshold_hops"].unique())
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3, linestyle="--")


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
        )
        for metric in metrics
    ]


def add_legend(ax: Axes, handles: Sequence[Patch | Line2D], title: str) -> None:
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        fontsize=9,
        frameon=False,
        title=title,
        title_fontsize=10,
    )


def add_panel_labels(fig: Figure, labels: tuple[str, ...], y_pos: float) -> None:
    for x_pos, label in zip(PANEL_LABEL_X, labels):
        fig.text(x_pos, y_pos, label, fontsize=12, fontweight="bold", va="top")


def make_grid(fig: Figure) -> GridSpec:
    return GridSpec(
        6,
        4,
        figure=fig,
        height_ratios=[1, 1.2, 0.8, 1, 1, 1],
        width_ratios=[1, 1, 1, 0.3],
        hspace=0.35,
        wspace=0.25,
        left=0.06,
        right=0.95,
        top=0.95,
        bottom=0.05,
    )


def create_figure(data: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Create and save the diagnostics figure."""
    fig = plt.figure(figsize=(14, 18))
    gs = make_grid(fig)

    summary_agg = data["summary_aggregate"]
    y_max_ambiguity = summary_agg["mixed_cell_fraction_max"].max() * 1.2

    for col, endpoint in enumerate(ENDPOINTS):
        plot_feature_ambiguity(
            fig.add_subplot(gs[0, col]),
            summary_agg,
            endpoint,
            y_max_ambiguity,
        )
        plot_oracle_pr(fig.add_subplot(gs[1, col]), data["graphs"], endpoint)
        plot_connected_components(
            fig.add_subplot(gs[2, col]),
            data["graphs"],
            endpoint,
        )

    for row, method in enumerate(TREE_METHODS):
        for col, endpoint in enumerate(ENDPOINTS):
            plot_tree_performance(
                fig.add_subplot(gs[3 + row, col]),
                data["trees"],
                method,
                endpoint,
            )

    add_legend(
        fig.add_subplot(gs[0, 3]),
        [
            Patch(facecolor=COLORS[process], label=PROCESS_LABELS[process])
            for process in PROCESSES
        ],
        "Process",
    )
    add_legend(
        fig.add_subplot(gs[1, 3]),
        metric_legend_handles(SCORE_METRICS),
        "Oracle Metrics",
    )
    add_legend(
        fig.add_subplot(gs[2, 3]),
        metric_legend_handles(SCORE_METRICS),
        "CC Metrics",
    )
    add_legend(
        fig.add_subplot(gs[3:6, 3]),
        metric_legend_handles(TREE_METRICS),
        "Tree Metrics",
    )

    add_panel_labels(fig, ("A1", "A2", "A3"), 0.98)
    add_panel_labels(fig, ("B1", "B2", "B3"), 0.85)
    add_panel_labels(fig, ("C1", "C2", "C3"), 0.72)

    for i, method in enumerate(TREE_METHODS):
        y_pos = (0.56, 0.38, 0.20)[i]
        fig.text(0.01, y_pos, f"D{i + 1}", fontsize=12, fontweight="bold", va="top")
        fig.text(
            0.04,
            y_pos,
            METHOD_LABELS[method],
            fontsize=9,
            va="center",
            style="italic",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"bbox_inches": "tight"}
    if output_path.suffix.lower() not in {".pdf", ".svg"}:
        save_kwargs["dpi"] = 300

    fig.savefig(output_path, **save_kwargs)
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
