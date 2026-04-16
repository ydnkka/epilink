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
    results/figures/surf.tif  – compatibility surfaces
    results/figures/baseline.tif  – baseline PR curves
    results/figures/perturbation.tif  – synthetic metric summaries
    results/figures/f1_loss.tif  – sensitivity F1-loss lollipop
    results/figures/ap_loss.tif – sensitivity AP-loss lollipop
    results/figures/temporal.tif  – temporal stability
    results/figures/boston.tif  – Boston cluster descriptives
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
from specs import MODEL_LABELS
from sklearn.metrics import precision_recall_curve

LOGGER = logging.getLogger(__name__)

# ─── Module-level config ──────────────────────────────────────────────────────

PROJECT_ROOT = project_root()
CONFIG = load_config()
RESULTS_ROOT = outputs_root(CONFIG)
FIGURE_OUTPUT_DIR = RESULTS_ROOT / "figures"

SAVE_FIGURES = True  # Set to False to avoid saving figures to disk.
SHOW_PLOTS = True  # Set to False to avoid showing plots.

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

# ─── I/O helpers ──────────────────────────────────────────────────────────────


def read_result_table(*parts: str) -> pd.DataFrame:
    """Load a parquet table from the configured results root."""
    return pd.read_parquet(RESULTS_ROOT.joinpath(*parts))


def export_figure(fig: plt.Figure, stem: str, **kwargs) -> dict[str, Path]:
    """Save *fig* if SAVE_FIGURES is True; otherwise return an empty dict."""
    if not SAVE_FIGURES:
        return {}
    return save_plos_figure(
        fig, stem, out_dir=FIGURE_OUTPUT_DIR, save_pdf=False, **kwargs
    )


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


# ─── Baseline PR Curves ────────────────────────────────


def make_fig_baseline() -> plt.Figure:
    """PR curves for all six models, including no-skill baseline."""
    scores_df = read_result_table("synthetic", "baseline_scores.parquet")

    y = scores_df["IsRelated"].values
    prevalence = float(y.mean())

    pr_data: dict[str, tuple] = {}
    for model in MODELS:
        scores = scores_df[model].values
        pr_data[model] = precision_recall_curve(y, scores)

    fig, ax = plt.subplots(
        figsize=(
            cm_to_inch(PLOS_WIDTHS_CM["text_column"]),
            cm_to_inch(PLOS_WIDTHS_CM["text_column"]) * 0.6,
        ),
        constrained_layout=True,
    )

    # ── PR curves ────────────────────────────────────────────────────────
    for i, model in enumerate(MODELS):
        prec, rec, _ = pr_data[model]
        ax.plot(
            rec,
            prec,
            color=MODEL_PALETTE[i],
            ls=MODEL_LINESTYLES[i],
            lw=1.6,
            label=f"{model}\n({MODEL_LABELS.get(model)})",
        )

    ax.axhline(
        prevalence,
        color="#555870",
        lw=1.0,
        ls="--",
        label=f"No-skill baseline ($\\pi$={prevalence:.3f})",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.12, color="#555870")
    ax.legend(
        title="Model",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        labelspacing=1.5,
        frameon=False,
    )
    return fig


# ─── Synthetic metric summaries ──────────────────────────────────────


def _plot_metric_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    title: str,
) -> None:
    sns.barplot(
        data=df,
        x="model",
        y=metric,
        hue="condition",
        palette=CONDITION_COLORS,
        hue_order=CONDITION_ORDER,
        errorbar=("ci", 95),
        capsize=0.2,
        err_kws={'linewidth': 1.2},
        width=0.75,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(title)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.12, color="#555870")


def make_fig_synthetic(results: pd.DataFrame) -> plt.Figure:
    """Per-model metric summaries across non-baseline scenarios."""

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
        _plot_metric_panel(ax, results, metric=metric, title=title)

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
    add_panel_labels(list(axes))
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
    ax.set_title(model, fontweight="bold")
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
    ax.set_yticklabels(_SENSITIVITY_LABELS)
    if not show_ylabels:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(_SENSITIVITY_TICKS[metric])
    ax.set_xticklabels(
        [("0" if t == 0.0 else f"{t:+.1f}") for t in _SENSITIVITY_TICKS[metric]],
    )
    ax.set_title(model, fontweight="bold")
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


def print_baseline_metrics() -> None:
    """Print per-model AP / F1 / stability at baseline."""

    df = read_result_table("synthetic", "baseline_summary.parquet").copy()

    print("\n── Baseline metrics ──────────────────────────────────────")

    for row in df.to_dict(orient="records"):
        print(
            f"  {row['model']}: "
            f"AP={row['ap']:.3f} (95% CI[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]), "
            f"Relative AP={row['relative_ap']:.3f} vs. prevalence AP={row['prevalence']:.5f}, "
            f"F1={row['best_f1']:.3f}, "
            f"Partition stability={row['mean_stability']:.3f} (SD={row['std_stability']:.3f}), "
            f"Brier score={row['brier']:.3f}"
        )


def print_stability_minima() -> None:
    """Print per-model minimum stability values across all epidemic weeks."""
    print("\n── Temporal stability mean (minima) ───────────────────────────────────────")
    metrics = list(STABILITY_LABELS)
    for model in MODELS:
        df = read_result_table("stability", f"temporal_stability_{model}.parquet")
        mins = df[metrics].min()
        means = df[metrics].mean()
        parts = ", ".join(f"{m}={means[m]:.3f} ({mins[m]:.3f})" for m in metrics)
        print(f"  {model}: {parts}")


def print_loss_pivot(results: pd.DataFrame) -> None:
    """Print mismatched F1-loss pivot (scenarios × models)."""
    matched = results.loc[results["condition"] == "Matched"]
    mismatched = results.loc[results["condition"] == "Mismatched"]

    print("\n── F1 loss – Matched condition ──────────────────────────────────")
    pivot = (
        matched[["model", "scenario", "f1_loss"]]
        .pivot_table(index="scenario", columns="model", values="f1_loss")
        .reindex(index=SCENARIO_ORDER, columns=MODELS)
        .reset_index()
    )
    with pd.option_context("display.float_format", "{:+.3f}".format, "display.width", 120):
        print(pivot.to_string(index=False))

    print("\n── F1 loss – Mismatched condition ──────────────────────────────────")
    pivot = (
        mismatched[["model", "scenario", "f1_loss"]]
        .pivot_table(index="scenario", columns="model", values="f1_loss")
        .reindex(index=SCENARIO_ORDER, columns=MODELS)
        .reset_index()
    )
    with pd.option_context("display.float_format", "{:+.3f}".format, "display.width", 120):
        print(pivot.to_string(index=False))

    print("\n── AP loss – Matched condition ──────────────────────────────────")
    pivot = (
        matched[["model", "scenario", "ap_loss"]]
        .pivot_table(index="scenario", columns="model", values="ap_loss")
        .reindex(index=SCENARIO_ORDER, columns=MODELS)
        .reset_index()
    )
    with pd.option_context("display.float_format", "{:+.3f}".format, "display.width", 120):
        print(pivot.to_string(index=False))

    print("\n── AP loss – Mismatched condition ──────────────────────────────────")
    pivot = (
        mismatched[["model", "scenario", "ap_loss"]]
        .pivot_table(index="scenario", columns="model", values="ap_loss")
        .reindex(index=SCENARIO_ORDER, columns=MODELS)
        .reset_index()
    )
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

    # Compatibility surfaces
    surf = make_fig_compatibility()
    if SAVE_FIGURES:
        export_figure(surf, "surfaces")

    if SHOW_PLOTS:
        plt.show()

    plt.close(surf)

    # Baseline PR curves
    baseline = make_fig_baseline()
    if SAVE_FIGURES:
        export_figure(baseline, "baseline")

    if SHOW_PLOTS:
        plt.show()

    plt.close(baseline)

    # Performance trend across scenarios
    results = read_result_table("synthetic", "results.parquet")
    results["condition"] = results["condition"].map(CONDITION_LABELS).fillna(results["condition"])
    results = results.loc[results["scenario"] != "baseline"].copy()
    results.sort_values("ap", ascending=False, inplace=True)
    trend = make_fig_synthetic(results)
    if SAVE_FIGURES:
        export_figure(trend, "perturbation")

    if SHOW_PLOTS:
        plt.show()

    plt.close(trend)

    # Temporal stability
    temp = make_fig_stability()
    if SAVE_FIGURES:
        export_figure(temp, "temporal")

    if SHOW_PLOTS:
        plt.show()

    plt.close(temp)

    # AP-loss/F1-loss lollipop
    for metric in ("ap_loss", "f1_loss"):
        fig = make_fig_sensitivity(results, metric)
        if SAVE_FIGURES:
            export_figure(fig, metric)

        if SHOW_PLOTS:
            plt.show()

        plt.close(fig)


    # Boston cluster composition
    boston = make_fig_boston()
    if SAVE_FIGURES:
        export_figure(boston, "boston")

    if SHOW_PLOTS:
        plt.show()

    plt.close(boston)

    # Diagnostic prints
    print_baseline_metrics()
    print_loss_pivot(results)
    print_stability_minima()

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
