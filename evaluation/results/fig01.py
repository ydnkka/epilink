"""Generate the synthetic diagnostics manuscript figure (fig01)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from epilink_evaluation.utils import style

from ._paths import output_directory

DIAGNOSTICS_ROOT = (
    Path(__file__).resolve().parents[1]
    / "00_synthetic_diagnostics"
    / "outputs"
    / "diagnostics"
)

ENDPOINTS = ("M0", "Mle1", "Mle2")
ENDPOINT_LABELS = {"M0": "M=0", "Mle1": "M≤1", "Mle2": "M≤2"}

FEATURE_SETS = ("GD", "GD_TD")
FEATURE_LABELS = {"GD": "Genetics", "GD_TD": "Genetics\n+ sampling time"}

PROCESSES = ("deterministic", "stochastic")
PROCESS_LABELS = {"deterministic": "Deterministic", "stochastic": "Stochastic"}

TREE_METHODS = ("avg_clade", "max_clade", "single_linkage")
METHOD_LABELS = {
    "avg_clade": "Average clade",
    "max_clade": "Maximum clade",
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
    linestyle: str = "-"
    alpha: float | None = None

    @property
    def fmt(self) -> str:
        return f"{self.marker}{self.linestyle}"


SCORE_METRICS = (
    MetricSpec("$F_1$", "f1", "o", COLORS["f1"]),
    MetricSpec("Precision", "precision", "s", COLORS["precision"], alpha=0.8),
    MetricSpec("Recall", "recall", "^", COLORS["recall"], alpha=0.8),
    MetricSpec(
        "Distant pairs",
        "Mge3_contamination",
        "d",
        COLORS["contamination"],
        linestyle="--",
        alpha=0.7,
    ),
)


def metric_column(endpoint: str, suffix: str) -> str:
    if suffix == "Mge3_contamination":
        return "Mge3_contamination_mean"
    return f"{endpoint}_{suffix}_mean"


def load_current_run(path: Path | None = None) -> Path:
    """Load a pinned run or the current run from current.json."""
    if path is not None:
        run_dir = path.expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir
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
    subset["error_lower"] = (
        subset["mixed_cell_fraction_mean"] - subset["mixed_cell_fraction_min"]
    ).clip(lower=0)
    subset["error_upper"] = (
        subset["mixed_cell_fraction_max"] - subset["mixed_cell_fraction_mean"]
    ).clip(lower=0)

    x_pos = range(len(FEATURE_SETS))
    bar_width = 0.35

    for i, process in enumerate(PROCESSES):
        process_data = subset[subset["process"].astype(str) == process]
        y_vals = []
        y_err = [[], []]

        for feature_set in FEATURE_SETS:
            feature_data = process_data[
                process_data["feature_set"].astype(str) == feature_set
            ]
            if feature_data.empty:
                y_vals.append(0)
                y_err[0].append(0)
                y_err[1].append(0)
                continue

            y_vals.append(feature_data["mixed_cell_fraction_mean"].iloc[0])
            y_err[0].append(feature_data["error_lower"].iloc[0])
            y_err[1].append(feature_data["error_upper"].iloc[0])

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
            label=metric.label,
            alpha=metric.alpha,
        )


def plot_f1_error_bars(
    ax: Axes,
    data: pd.DataFrame,
    endpoint: str,
    x_col: str,
) -> None:
    """Show the observed minimum–maximum range around mean F1."""
    f1_mean = f"{endpoint}_f1_mean"
    f1_min = f"{endpoint}_f1_min"
    f1_max = f"{endpoint}_f1_max"
    ax.errorbar(
        data[x_col],
        data[f1_mean],
        yerr=[
            (data[f1_mean] - data[f1_min]).clip(lower=0),
            (data[f1_max] - data[f1_mean]).clip(lower=0),
        ],
        fmt="none",
        capsize=3,
        elinewidth=1,
        color="gray",
        alpha=0.7,
    )


def plot_oracle_pr(ax: Axes, data: pd.DataFrame, endpoint: str) -> None:
    """Compare Leiden with connected components on exact target links."""
    leiden_data = data[
        (data["endpoint"].astype(str) == endpoint)
        & (data["algorithm"].astype(str) == "leiden")
    ].copy()
    leiden_data["resolution_float"] = leiden_data["resolution"].astype(float)
    leiden_data = leiden_data.sort_values("resolution_float")

    plot_score_metrics(ax, leiden_data, endpoint, "resolution_float", SCORE_METRICS)
    plot_f1_error_bars(ax, leiden_data, endpoint, "resolution_float")

    components_data = data[
        (data["endpoint"].astype(str) == endpoint)
        & (data["algorithm"].astype(str) == "components")
    ].copy()
    # CC is a separate reference, not a point on the Leiden resolution curve.
    components_data["reference_position"] = -0.15
    for metric in SCORE_METRICS:
        ax.plot(
            components_data["reference_position"],
            components_data[metric_column(endpoint, metric.suffix)],
            marker=metric.marker,
            linestyle="none",
            color=metric.color,
            alpha=metric.alpha,
        )
    plot_f1_error_bars(ax, components_data, endpoint, "reference_position")
    ax.axvline(-0.025, color="0.75", linestyle=":", linewidth=0.8)

    ax.set_xlim(-0.25, 1.05)
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
        bbox_to_anchor=(top_axes[0].get_position().x0, 1.06),
        ncol=len(PROCESSES),
        title="Genetic observations",
        borderaxespad=0,
    )
    fig.legend(
        handles=metric_legend_handles(SCORE_METRICS),
        loc="upper right",
        bbox_to_anchor=(top_axes[-1].get_position().x1, 1.07),
        ncol=2,
        title="Metrics",
        borderaxespad=0,
    )


def add_row_labels(fig: Figure, row_axes: Sequence[Axes]) -> None:
    """Align each row label with its axes rather than fixed figure heights."""
    labels = [
        "Observation\nambiguity",
        "Exact target links\nCC / Leiden",
        *[f"TreeCluster\n{METHOD_LABELS[method]}" for method in TREE_METHODS],
    ]
    for ax, label in zip(row_axes, labels):
        ax.annotate(
            label,
            xy=(-0.1, 0.5),
            xytext=(-60, 0),
            xycoords="axes fraction",
            textcoords="offset points",
            fontweight="bold",
            ha="center",
            va="center",
        )
        # bounds = ax.get_position()
        # y_pos = (bounds.y0 + bounds.y1) / 2
        # fig.text(bounds.x0 - 0.15, y_pos, label, ha="right", va="top")


def create_figure(
    data: dict[str, pd.DataFrame],
    output_path: Path,
    *,
    save_pdf: bool = True,
    save_png: bool = True,
) -> None:
    """Create and save the diagnostics figure."""
    fig, axes = style.new_figure(
        width="double",
        height_in=8,
        nrows=2 + len(TREE_METHODS),
        ncols=len(ENDPOINTS),
        layout="constrained",
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
        axes[0, col].set_title(ENDPOINT_LABELS[endpoint], fontweight="bold", pad=10)

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
    axes[1, 0].set_xticks(
        [-0.15, 0.2, 0.4, 0.6, 0.8, 1.0],
        ["CC", "0.2", "0.4", "0.6", "0.8", "1.0"],
    )
    axes[2, 0].xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

    x_labels = ("Observed information", "Leiden resolution") + ("Cutoff (transmission hops)",) * len(TREE_METHODS)
    for row, row_axes in enumerate(axes):
        row_axes[0].set_ylabel(
            "Fraction of mixed\nobserved combinations" if row == 0 else "Metric value"
        )
        for col, ax in enumerate(row_axes):
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="0.9")
            ax.tick_params(axis="y", left=col == 0, labelleft=col == 0)
            ax.tick_params(axis="x", bottom=True, labelbottom=True)
            ax.set_xlabel(x_labels[row])

    fig.align_ylabels(axes[:, 0])
    add_legends(fig, axes[0])
    add_row_labels(fig, axes[:, 0])
    style.add_panel_labels(axes.ravel())

    saved_paths = style.save_figure(
        fig, output_path, width="double", save_pdf=save_pdf, save_png=save_png
    )
    for path in saved_paths.values():
        print(f"Figure saved to: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate diagnostics manuscript figure"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Pinned diagnostics run; default: diagnostics/current.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the run-specific results directory",
    )
    parser.add_argument(
        "--format",
        choices=("pdf", "png", "both"),
        default="both",
        help="Output format (default: both)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading diagnostics run...")
    run_dir = load_current_run(args.run_dir)
    print(f"Using run: {run_dir.name}")

    print("Loading data...")
    data = load_data(run_dir)

    create_figure(
        data,
        output_directory(run_dir, "00_synthetic_diagnostics", args.output_dir)
        / "fig01_diagnostics_figure",
        save_pdf=args.format in {"pdf", "both"},
        save_png=args.format in {"png", "both"},
    )

    print("Done!")


if __name__ == "__main__":
    main()
