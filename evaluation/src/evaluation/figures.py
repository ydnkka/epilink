"""Assemble manuscript-ready figures from completed EpiLink evaluation outputs.

Usage
-----
    python figures.py            # save PDF + TIF exports to results/figures/
    python figures.py --no-save  # preview only (no files written)

Required inputs (resolved via config.yaml)
------------------------------------------
- results/sparsification/score_surfaces.parquet
- results/synthetic/results.parquet
- results/synthetic/baseline_scores.parquet
- results/synthetic/baseline_summary.parquet
- results/stability/temporal_stability_{model}.parquet  (one per model)
- results/boston/cluster_composition.parquet
- results/boston/cluster_sizes.parquet

Generated outputs
-----------------
    results/figures/Fig2.{pdf,tif}  – baseline PR / PRG / calibration
    results/figures/Fig3.{pdf,tif}  – compatibility surfaces
    results/figures/Fig4.{pdf,tif}  – synthetic metric summaries
    results/figures/Fig5.{pdf,tif}  – sensitivity F1-loss lollipop
    results/figures/S1_Fig.{pdf,tif} – sensitivity AP-loss lollipop
    results/figures/Fig6.{pdf,tif}  – temporal stability
    results/figures/Fig7.{pdf,tif}  – Boston cluster descriptives
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config import load_config, outputs_root, project_root
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from plotting import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_ORDER,
    MODEL_LINESTYLES,
    MODEL_PALETTE,
    MODELS,
    PLOS_WIDTHS_CM,
    SCENARIO_LABELS,
    SCENARIO_ORDER,
    STABILITY_COLORS,
    STABILITY_LABELS,
    STABILITY_MARKERS,
    add_panel_labels,
    cm_to_inch,
    save_plos_figure,
    set_plos_theme,
)
from sklearn.metrics import precision_recall_curve

LOGGER = logging.getLogger(__name__)

# ─── Module-level config ──────────────────────────────────────────────────────

PROJECT_ROOT = project_root()
CONFIG = load_config()
RESULTS_ROOT = outputs_root(CONFIG)
FIGURE_OUTPUT_DIR = RESULTS_ROOT / "figures"

# Set to False to preview without writing any files.
SAVE_FIGURES = True

# Metric panels for the synthetic figure (column name → y-axis label).
METRIC_PANELS: list[tuple[str, str]] = [
    ("ap", "Average precision"),
    ("best_f1", "Best F1 score"),
    ("mean_stability", "Partition stability (mean)"),
    ("std_stability", "Partition stability (SD)"),
]

# Score columns for the compatibility-surface panels.
SURFACE_PANELS: tuple[str, ...] = (
    "compatibility_deterministic",
    "compatibility_stochastic",
)

# Exposure-count columns present in the Boston composition table.
EXPOSURE_LABELS: dict[str, str] = {
    "count::BHCHP": "BHCHP",
    "count::Other": "Other",
    "count::City": "City",
    "count::Conference": "Conference",
    "count::SNF": "SNF",
}

# Sensitivity lollipop: clamp values and x-tick positions per metric.
_SENSITIVITY_CLAMP: dict[str, float] = {"ap_loss": 1.1, "f1_loss": 0.7}
_SENSITIVITY_TICKS: dict[str, list[float]] = {
    "ap_loss": [-1.0, -0.5, 0.0, 0.5, 1.0],
    "f1_loss": [-0.6, -0.3, 0.0, 0.3, 0.6],
}

# Scenario display order and labels for sensitivity figures (non-baseline only).
_SENSITIVITY_KEYS: list[str] = SCENARIO_ORDER
_SENSITIVITY_LABELS: list[str] = [SCENARIO_LABELS[k] for k in SCENARIO_ORDER]

# Font sizes
TINY = "x-small"
SMALL = "small"
MEDIUM = "medium"

# ─── I/O helpers ──────────────────────────────────────────────────────────────


def read_result_table(*parts: str) -> pd.DataFrame:
    """Load a parquet table from the configured results root."""
    return pd.read_parquet(RESULTS_ROOT.joinpath(*parts))


def export_figure(fig: plt.Figure, stem: str, **kwargs) -> dict[str, Path]:
    """Save *fig* if SAVE_FIGURES is True; otherwise return an empty dict."""
    if not SAVE_FIGURES:
        return {}
    return save_plos_figure(fig, stem, out_dir=FIGURE_OUTPUT_DIR, **kwargs)


# ─── Compatibility surfaces ──────────────────────────────────────────


def _surface_matrix(surface_frame: pd.DataFrame, score: str) -> pd.DataFrame:
    """Pivot a long surface table into a days × SNP matrix."""
    return (
        surface_frame.pivot(index="days", columns="snp", values=score)
        .sort_index()
        .sort_index(axis=1)
    )


def _add_surface_panel(ax: plt.Axes, surface_frame: pd.DataFrame, score: str):
    pivot = _surface_matrix(surface_frame, score)
    filled = ax.contourf(
        pivot.columns.to_numpy(dtype=float),
        pivot.index.to_numpy(dtype=float),
        pivot.to_numpy(dtype=float),
        levels=np.linspace(0.0, 1.0, 11),
        cmap="mako",
        antialiased=True,
    )
    ax.xaxis.set_major_locator(MultipleLocator(3))
    ax.yaxis.set_major_locator(MultipleLocator(3))
    return filled


def make_fig_compatibility() -> plt.Figure:
    """Compatibility surfaces (deterministic vs stochastic mutation process)."""
    surfaces = read_result_table("sparsification", "score_surfaces.parquet")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            cm_to_inch(PLOS_WIDTHS_CM["text_column"]),
            cm_to_inch(PLOS_WIDTHS_CM["text_column"]) * 0.5,
        ),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"hspace": 0.1, "wspace": 0.1},
    )
    axes = np.atleast_1d(axes).flatten()

    filled = None
    for ax, score in zip(axes, SURFACE_PANELS):
        filled = _add_surface_panel(ax, surfaces, score)

    if filled is None:
        raise RuntimeError("No surface panels were drawn.")

    colorbar = fig.colorbar(filled, ax=axes, location="right", shrink=0.95, pad=0.03)
    colorbar.set_label("Compatibility score")
    fig.supxlabel("Genetic distance (SNPs)")
    fig.supylabel("Temporal gap (days)")
    add_panel_labels(list(axes))
    return fig


# ─── Baseline PR / PRG / calibration ─────────────────────────────────


def make_fig_baseline() -> plt.Figure:
    """Four-panel baseline performance figure.

    Panels
    ------
    A  PR curves with AP annotations and bootstrap CI
    B  Precision-Recall-Gain (PRG) curves
    C  Isotonic-calibration reliability diagrams
    D  Best-F1 horizontal bar chart

    Loads
    -----
    - ``synthetic/baseline_scores.parquet``   raw scores + IsRelated labels
    - ``synthetic/baseline_summary.parquet``  pre-computed summary from analyse_baseline()
    """
    scores_df = read_result_table("synthetic", "baseline_scores.parquet")
    summary_df = read_result_table("synthetic", "baseline_summary.parquet")

    y = scores_df["IsRelated"].values
    prevalence = float(y.mean())

    pr_data: dict[str, tuple] = {}
    for model in MODELS:
        scores = scores_df[model].values
        pr_data[model] = precision_recall_curve(y, scores)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            cm_to_inch(PLOS_WIDTHS_CM["text_column"]),
            cm_to_inch(PLOS_WIDTHS_CM["text_column"]) * 0.6,
        ),
        constrained_layout=True,
        gridspec_kw={"hspace": 0.1, "wspace": 0.1},
        width_ratios=[1.75, 1],
    )
    axes = np.atleast_1d(axes).flatten()

    ax_pr = axes[0]
    ax_bar = axes[1]

    # ── PR curves ────────────────────────────────────────────────────────
    for i, model in enumerate(MODELS):
        row = summary_df.loc[summary_df["model"] == model].iloc[0]
        prec, rec, _ = pr_data[model]
        ax_pr.plot(
            rec,
            prec,
            color=MODEL_PALETTE[i],
            ls=MODEL_LINESTYLES[i],
            lw=1.6,
            label=f"{model}: AP={row['ap']:.3f} [{row['ci_lo']:.3f},{row['ci_hi']:.3f}]",
        )

    ax_pr.axhline(
        prevalence,
        color="#555870",
        lw=1.0,
        ls="--",
        label=f"No-skill baseline ($\\pi$={prevalence:.3f})",
    )

    ax_pr.set_xlabel("Recall", fontsize=MEDIUM)
    ax_pr.set_ylabel("Precision", fontsize=MEDIUM)
    ax_pr.tick_params(labelsize=SMALL)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)
    ax_pr.grid(True, alpha=0.12, color="#555870")
    ax_pr.legend(
        loc="upper right",
        bbox_to_anchor=(1.00, 1.00),
        # framealpha=0.55,
        frameon=True,
        fancybox=True,
        facecolor="white",
        edgecolor="white",
        fontsize=TINY,
    )

    # ── Best-F1 bar chart ────────────────────────────────────────────────
    ordered = summary_df.sort_values("best_f1", ascending=True)
    y_pos = np.arange(len(MODELS))
    bars = ax_bar.barh(
        y_pos,
        ordered["best_f1"],
        color=[MODEL_PALETTE[MODELS.index(m)] for m in ordered["model"]],
    )
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(ordered["model"], fontsize=SMALL)
    ax_bar.set_xlabel("Best F1", fontsize=MEDIUM)
    ax_bar.tick_params(labelsize=SMALL)
    ax_bar.grid(True, alpha=0.12, color="#555870", axis="x")
    ax_bar.set_xlim(0, 1)
    for bar, (_, row) in zip(bars, ordered.iterrows()):
        ax_bar.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{row['best_f1']:.3f}",
            va="center",
            ha="left",
            fontsize=TINY,
        )

    add_panel_labels(list(axes))
    return fig


# ─── Sensitivity lollipop plots ─────────────────────────────────


def _draw_sensitivity_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    model: str,
    show_ylabels: bool,
) -> None:
    clamp = _SENSITIVITY_CLAMP[metric]
    n = len(_SENSITIVITY_KEYS)
    offset = 0.18
    model_df = df[df["model"] == model]

    for cond, y_offset in [("Matched", -offset), ("Mismatched", +offset)]:
        col = CONDITION_COLORS[cond]
        cond_df = model_df[model_df["condition"] == cond].set_index("scenario")
        for i, sk in enumerate(_SENSITIVITY_KEYS):
            if sk not in cond_df.index:
                continue
            v = cond_df.loc[sk, metric]
            if pd.isna(v):
                continue
            clipped = abs(v) > clamp
            xv = float(np.clip(v, -clamp, clamp))
            cy = float(i) + y_offset
            ax.plot(
                [0, xv],
                [cy, cy],
                color=col,
                linewidth=1.2,
                alpha=0.7,
                linestyle="--" if clipped else "-",
                solid_capstyle="round",
            )
            ax.plot(xv, cy, "o", color=col, markersize=3.0, zorder=3, markeredgewidth=0)
            if clipped:
                sign = 1 if v > 0 else -1
                ax.annotate(
                    "",
                    xy=(sign * clamp, cy),
                    xytext=(sign * clamp * 0.88, cy),
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.8),
                )

    ax.axvline(0, color="black", linewidth=0.6, alpha=0.25, zorder=0)
    ax.set_xlim(-clamp * 1.08, clamp * 1.08)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(_SENSITIVITY_LABELS, fontsize=MEDIUM)
    if not show_ylabels:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(_SENSITIVITY_TICKS[metric])
    ax.set_xticklabels(
        [("0" if t == 0.0 else f"{t:+.1f}") for t in _SENSITIVITY_TICKS[metric]],
        fontsize=SMALL,
    )
    ax.set_title(model, fontsize=SMALL, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, alpha=0.12, color="#555870", axis="x")


def make_fig_sensitivity(df: pd.DataFrame, metric: str) -> plt.Figure:
    """Main (metric='f1_loss') or Supplementary (metric='ap_loss') – sensitivity lollipops.

    Parameters
    ----------
    df:
        The synthetic results table, already condition-label-mapped and including
        non-baseline scenarios only (or with baseline rows present – they are
        ignored inside the panel drawing).
    metric:
        Either ``'ap_loss'`` or ``'f1_loss'``.
    """
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            cm_to_inch(PLOS_WIDTHS_CM["text_column"]),
            cm_to_inch(PLOS_WIDTHS_CM["text_column"]) * 1.2,
        ),
        constrained_layout=True,
        gridspec_kw={"hspace": 0.1},
    )
    for idx, (ax, model) in enumerate(zip(axes.flatten(), MODELS)):
        _draw_sensitivity_panel(ax, df, metric, model, show_ylabels=(idx % ncols == 0))

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=CONDITION_COLORS[cond],
            markersize=5,
            label=cond,
        )
        for cond in CONDITION_ORDER
    ]
    fig.legend(
        title="Condition",
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.6, 1.075),
        ncol=2,
        frameon=False,
    )
    metric_label = "Average precision (AP) loss" if metric == "ap_loss" else "F1 score loss"
    fig.supxlabel(f"{metric_label} relative to baseline", x=0.6, ha="center")
    return fig


# ─── Synthetic metric summaries ──────────────────────────────────────


def _plot_metric_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    title: str,
) -> None:
    sns.pointplot(
        data=df,
        x="model",
        y=metric,
        hue="condition",
        palette=CONDITION_COLORS,
        order=MODELS,
        hue_order=CONDITION_ORDER,
        dodge=True,
        errorbar=("ci", 95),
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(title)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.12, color="#555870")


def make_fig_synthetic(results: pd.DataFrame) -> plt.Figure:
    """Per-model metric summaries across non-baseline scenarios."""
    df = results.loc[results["scenario"] != "baseline"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            cm_to_inch(PLOS_WIDTHS_CM["text_column"]),
            cm_to_inch(PLOS_WIDTHS_CM["text_column"]),
        ),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"hspace": 0.1, "wspace": 0.1},
    )
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, METRIC_PANELS):
        _plot_metric_panel(ax, df, metric=metric, title=title)

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        if ax.legend_ is not None:
            ax.legend_.remove()

    fig.legend(
        handles,
        labels,
        title="Condition",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=2,
    )
    fig.supxlabel("Model")
    add_panel_labels(list(axes), size=MEDIUM)
    return fig


# ─── Temporal stability ───────────────────────────────────────────────


def _plot_stability_panel(df: pd.DataFrame, ax: plt.Axes, model: str) -> None:
    for metric in STABILITY_LABELS:
        sns.lineplot(
            data=df,
            x="t1",
            y=metric,
            color=STABILITY_COLORS[metric],
            marker=STABILITY_MARKERS[metric],
            ax=ax,
            label=STABILITY_LABELS[metric],
        )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim(0, 1.05)
    ax.set_title(model, fontsize=SMALL, fontweight="bold")
    ax.grid(True, alpha=0.12, color="#555870")


def make_fig_stability() -> plt.Figure:
    """Temporal stability across epidemic weeks, one panel per model."""
    stability_frames = {
        m: read_result_table("stability", f"temporal_stability_{m}.parquet") for m in MODELS
    }

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(
            cm_to_inch(PLOS_WIDTHS_CM["text_column"]),
            cm_to_inch(PLOS_WIDTHS_CM["text_column"]) * 1.15,
        ),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"hspace": 0.1, "wspace": 0.1},
    )
    axes = axes.flatten()

    for ax, (m, frame) in zip(axes, stability_frames.items()):
        _plot_stability_panel(frame, ax, m)

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        if ax.legend_ is not None:
            ax.legend_.remove()

    fig.legend(
        handles,
        labels,
        title="Stability metric",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.075),
        ncol=3,
    )
    fig.supylabel("Temporal stability")
    fig.supxlabel("Epidemic week")
    return fig


# ─── Boston cluster composition ──────────────────────────────────────


def make_fig_boston() -> plt.Figure:
    """Cluster-size histogram paired with exposure-composition heatmap."""
    boston_comp = read_result_table("boston", "cluster_composition.parquet")
    boston_sizes = read_result_table("boston", "cluster_sizes.parquet")

    available = [c for c in EXPOSURE_LABELS if c in boston_comp.columns]
    if not available:
        raise ValueError("No configured exposure columns found in Boston results.")

    exposure_counts = boston_comp[available].copy().rename(columns=EXPOSURE_LABELS)
    exposure_counts.index = boston_comp["cluster_id"].astype(int)
    size_counts = boston_sizes["size"].value_counts().sort_index()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            cm_to_inch(PLOS_WIDTHS_CM["text_column"]),
            cm_to_inch(PLOS_WIDTHS_CM["text_column"]) * 0.5,
        ),
        constrained_layout=True,
        gridspec_kw={"hspace": 0.1, "wspace": 0.1},
    )
    axes = np.atleast_1d(axes).flatten()

    ax_bar = axes[0]
    ax_map = axes[1]

    bars = ax_bar.bar(
        size_counts.index,
        size_counts.values,
        color="#DDEAF7",
        edgecolor="#0072B2",
        linewidth=0.8,
        width=2,
    )
    for patch in bars:
        patch.set_hatch("//")
    ax_bar.set(xlabel="Cluster size", ylabel="Clusters")
    ax_bar.text(
        0.98,
        0.98,
        (
            f"All clusters: {len(boston_sizes)}\n"
            f"Focus clusters: {int(boston_sizes['is_focus_cluster'].sum())}"
        ),
        transform=ax_bar.transAxes,
        ha="right",
        va="top",
    )
    ax_bar.grid(True, alpha=0.12, color="#555870")

    vmax = float(np.nanmax(exposure_counts.to_numpy(dtype=float)))
    sns.heatmap(
        exposure_counts.T,
        vmin=0,
        vmax=max(1.0, vmax),
        cmap="YlOrBr",
        annot=True,
        fmt=".0f",
        linewidths=0.01,
        linecolor="black",
        cbar=False,
        ax=ax_map,
    )
    ax_map.tick_params(axis="y", rotation=0)
    ax_map.set_xlabel("Focus cluster ID")
    ax_map.set_ylabel("")

    add_panel_labels(list(axes))
    return fig


# ─── Diagnostic helpers ───────────────────────────────────────────────────────


def print_baseline_metrics(results: pd.DataFrame) -> None:
    """Print per-model AP / F1 / stability at baseline (Matched condition)."""
    print("\n── Baseline metrics (Matched) ──────────────────────────────────────")
    baseline = results.loc[
        (results["scenario"] == "baseline") & (results["condition"] == "Matched")
    ]
    for row in baseline.to_dict(orient="records"):
        print(
            f"  {row['model']}: "
            f"AP={row['ap']:.3f}, "
            f"F1={row['best_f1']:.3f}, "
            f"Mean={row['mean_stability']:.3f}, "
            f"SD={row['std_stability']:.3f}"
        )


def print_stability_minima() -> None:
    """Print per-model minimum stability values across all epidemic weeks."""
    print("\n── Temporal stability minima ───────────────────────────────────────")
    metrics = list(STABILITY_LABELS)
    for model in MODELS:
        df = read_result_table("stability", f"temporal_stability_{model}.parquet")
        mins = df[metrics].min()
        parts = ", ".join(f"{m}={mins[m]:.3f}" for m in metrics)
        print(f"  {model}: {parts}")


def print_f1_loss_pivot(results: pd.DataFrame) -> None:
    """Print mismatched F1-loss pivot (scenarios × models)."""
    data = results.loc[(results["scenario"] != "baseline") & (results["condition"] == "Mismatched")]
    pivot = (
        data[["model", "scenario", "f1_loss"]]
        .pivot_table(index="scenario", columns="model", values="f1_loss")
        .reindex(index=SCENARIO_ORDER, columns=MODELS)
        .reset_index()
    )
    print("\n── F1 loss – Mismatched condition ──────────────────────────────────")
    with pd.option_context("display.float_format", "{:+.3f}".format, "display.width", 120):
        print(pivot.to_string(index=False))


# ─── Entry point ──────────────────────────────────────────────────────────────


def main(*, save: bool = True) -> None:
    global SAVE_FIGURES
    SAVE_FIGURES = save

    set_plos_theme()

    LOGGER.info("figures: project root  = %s", PROJECT_ROOT)
    LOGGER.info("figures: results root  = %s", RESULTS_ROOT)
    LOGGER.info("figures: figure output = %s", FIGURE_OUTPUT_DIR)
    LOGGER.info("figures: save figures  = %s", SAVE_FIGURES)

    # Fig 2 – compatibility surfaces
    fig2 = make_fig_compatibility()
    export_figure(fig2, "Fig2")
    if SAVE_FIGURES:
        plt.show()
    plt.close(fig2)

    # Fig 3 – baseline PR / PRG / calibration / F1 bar
    fig3 = make_fig_baseline()
    export_figure(fig3, "Fig3")
    if SAVE_FIGURES:
        plt.show()
    plt.close(fig3)

    # Fig 4 / S1 – load synthetic results once, share across figures
    results = read_result_table("synthetic", "results.parquet").copy()
    results["condition"] = results["condition"].map(CONDITION_LABELS).fillna(results["condition"])
    print_baseline_metrics(results)
    print_f1_loss_pivot(results)

    fig4 = make_fig_synthetic(results)
    export_figure(fig4, "Fig4")
    if SAVE_FIGURES:
        plt.show()
    plt.close(fig4)

    # Fig 5 S1 Fig – AP-loss/F1-loss lollipop
    for stem, metric in (("S1_Fig", "ap_loss"), ("Fig5", "f1_loss")):
        fig = make_fig_sensitivity(results, metric)
        export_figure(fig, stem)
        if SAVE_FIGURES:
            plt.show()
        plt.close(fig)

    # Fig 6 – temporal stability
    print_stability_minima()
    fig6 = make_fig_stability()
    export_figure(fig6, "Fig6")
    if SAVE_FIGURES:
        plt.show()
    plt.close(fig6)

    # Fig 7 – Boston cluster composition
    fig7 = make_fig_boston()
    export_figure(fig7, "Fig7")
    if SAVE_FIGURES:
        plt.show()
    plt.close(fig7)

    LOGGER.info("figures: done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assemble EpiLink manuscript figures.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Preview figures without writing any files.",
    )
    args = parser.parse_args()
    main(save=not args.no_save)
