"""Relative change sensitivity analysis with box plot summaries.

This script generates PLOS-compliant figures showing parameter perturbation effects
on EpiLink and Logistic model families under matched and mismatched conditions,
using relative change from baseline metrics.

Figure layout
-------------
- Two vertical panels: EpiLink (top), Logistic (bottom)
- X-axis: Perturbation scenarios (ordered by effect magnitude)
- Box plots: Paired boxes per scenario showing Matched vs Mismatched conditions
- Y-axis: Relative change from baseline (reference line at 0)

Usage
-----
    python sensitivity_analysis.py           # save PDF + TIFF exports
    python sensitivity_analysis.py --no-save # preview only (no files written)
    python sensitivity_analysis.py --metric ap  # AP metric only
    python sensitivity_analysis.py --metric f1  # F1 metric only
    python sensitivity_analysis.py --canonical-order  # use canonical scenario order

Required inputs (resolved via config.yaml)
------------------------------------------
- results/synthetic/results.parquet
- results/synthetic/baseline_summary.parquet

Generated outputs
-----------------
    results/figures/sensitivity_ap.tif  – AP relative change
    results/figures/sensitivity_f1.tif  – F1 relative change
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config import (
    configure_logging,
    get_pipeline_log_path,
    load_config,
    outputs_root,
    project_root,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from plotting import (
    MODEL_PALETTE,
    MODELS,
    PLOS_WIDTHS_CM,
    SCENARIO_LABELS,
    SCENARIO_ORDER,
    add_panel_labels,
    cm_to_inch,
    save_plos_figure,
    set_plos_theme,
)

# Model family groupings
EPILINK_MODELS = ["EDD", "EDS", "ESD", "ESS"]
LOGISTIC_MODELS = ["LD", "LS"]
MODEL_FAMILY = {m: "EpiLink" for m in EPILINK_MODELS}
MODEL_FAMILY.update({m: "Logistic" for m in LOGISTIC_MODELS})
MODEL_FAMILY_COLORS = {
    "EpiLink": "#3A86FF",
    "Logistic": "#FF6B35",
}
CONDITION_COLORS = {
    "Matched": "#5C4AE4",
    "Mismatched": "#F5A623",
}

LOGGER = logging.getLogger(__name__)

# ─── Module-level config ──────────────────────────────────────────────────────

PROJECT_ROOT = project_root()
CONFIG = load_config()
RESULTS_ROOT = outputs_root(CONFIG)
FIGURE_OUTPUT_DIR = RESULTS_ROOT / "figures"
FIGURE_LOG_PATH = RESULTS_ROOT / "logs" / "sensitivity_analysis.log"

SAVE_FIGURES = True
SHOW_PLOTS = False

# Metric configuration
METRIC_CONFIG = {
    "ap": {
        "column": "ap",
        "loss_column": "ap_loss",
        "label": "Relative change in AP from baseline",
        "file_stem": "sensitivity_ap",
    },
    "f1": {
        "column": "best_f1",
        "loss_column": "f1_loss",
        "label": "Relative change in best F1 score from baseline",
        "file_stem": "sensitivity_f1",
    },
}

# ─── I/O helpers ──────────────────────────────────────────────────────────────


def read_result_table(*parts: str) -> pd.DataFrame:
    """Load a parquet table from the configured results root."""
    path = RESULTS_ROOT.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing result table: {path}. Run the workflow step that "
            "produces this file before generating figures."
        )
    return pd.read_parquet(path)


def export_figure(fig: Figure, stem: str) -> dict[str, Path]:
    """Save fig if SAVE_FIGURES is True; otherwise return empty dict."""
    if not SAVE_FIGURES:
        return {}
    return save_plos_figure(
        fig,
        stem,
        out_dir=FIGURE_OUTPUT_DIR,
        dpi=600,
        save_pdf=True,
        save_png=False,
        save_tiff=True,
        save_eps=False,
    )


# ─── Scenario ordering ────────────────────────────────────────────────────────


def order_scenarios_by_magnitude(
    df: pd.DataFrame, metric: str, use_magnitude: bool = True
) -> list[str]:
    """Order scenarios by mean absolute relative change across models and conditions.

    Parameters
    ----------
    df : pd.DataFrame
        Results with relative change column.
    metric : str
        Either 'ap' or 'f1'.
    use_magnitude : bool
        If True, order by mean absolute relative change.
        If False, use canonical SCENARIO_ORDER.

    Returns
    -------
    list[str]
        Scenario keys in specified order.
    """
    if not use_magnitude:
        return SCENARIO_ORDER

    config = METRIC_CONFIG[metric]
    loss_col = config["loss_column"]

    subset = df[df["scenario"] != "baseline"]
    if subset.empty:
        return SCENARIO_ORDER

    mean_abs = (
        subset.groupby("scenario")[loss_col]
        .apply(lambda x: np.mean(np.abs(x)))
        .sort_values(ascending=False)
    )
    return mean_abs.index.tolist()


# ─── Visualization ────────────────────────────────────────────────────────────


def _draw_box_panel(
    ax: Axes,
    df: pd.DataFrame,
    metric: str,
    model_family: str,
    scenario_order: list[str],
    show_xlabel: bool,
) -> None:
    """Draw a single panel with box plots grouped by condition.

    Parameters
    ----------
    ax : Axes
        Target axes.
    df : pd.DataFrame
        Results with relative change column.
    metric : str
        Either 'ap' or 'f1'.
    model_family : str
        'EpiLink' or 'Logistic'.
    scenario_order : list[str]
        Ordered list of scenario keys.
    show_xlabel : bool
        Whether to render x-axis labels.
    """
    config = METRIC_CONFIG[metric]
    loss_col = config["loss_column"]

    models = EPILINK_MODELS if model_family == "EpiLink" else LOGISTIC_MODELS
    subset = df[
        (df["model"].isin(models))
        & (df["scenario"] != "baseline")
    ]

    # Prepare data for box plots: paired boxes per scenario (Matched, Mismatched)
    box_data = []
    positions = []
    box_colors = []

    n_scenarios = len(scenario_order)
    box_width = 0.30
    gap_between_pairs = 0.4
    gap_within_pair = 0.08

    for i, scenario in enumerate(scenario_order):
        scenario_data = subset[subset["scenario"] == scenario]
        if scenario_data.empty:
            continue

        base_pos = i * (2 + gap_between_pairs)

        # Matched box (left)
        matched_values = scenario_data[
            scenario_data["condition"] == "Matched"
        ][loss_col].dropna()
        if len(matched_values) > 0:
            box_data.append(matched_values.values)
            positions.append(base_pos + 0.5)
            box_colors.append(CONDITION_COLORS["Matched"])

        # Mismatched box (right)
        mismatched_values = scenario_data[
            scenario_data["condition"] == "Mismatched"
        ][loss_col].dropna()
        if len(mismatched_values) > 0:
            box_data.append(mismatched_values.values)
            positions.append(base_pos + 1 + gap_within_pair)
            box_colors.append(CONDITION_COLORS["Mismatched"])

    if not box_data:
        return

    # Create box plot
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(linewidth=1.0),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )

    # Color the boxes by condition
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Reference line at zero
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5, zorder=0)

    # X-axis formatting
    scenario_centers = [
        i * (2 + gap_between_pairs) + 1 + gap_within_pair / 2
        for i in range(n_scenarios)
    ]
    ax.set_xticks(scenario_centers)
    if show_xlabel:
        ax.set_xticklabels(
            [SCENARIO_LABELS.get(s, s) for s in scenario_order],
            fontsize=7,
            rotation=45,
            ha="right",
        )
    else:
        ax.set_xticklabels([])

    ax.set_xlim(-0.5, max(positions) + 0.5)
    ax.set_ylabel("Normalized change")
    ax.set_title(model_family, fontweight="bold", fontsize=10)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, alpha=0.12, color="#555870", axis="y")


def make_fig_sensitivity(
    df: pd.DataFrame, metric: str, order_by_magnitude: bool = True
) -> Figure:
    """Create box plot figure for relative change sensitivity.

    Parameters
    ----------
    df : pd.DataFrame
        Synthetic results with relative change column.
    metric : str
        Either 'ap' or 'f1'.
    order_by_magnitude : bool
        If True, order scenarios by mean absolute relative change.

    Returns
    -------
    Figure
        Matplotlib figure with two panels (EpiLink, Logistic).
    """
    config = METRIC_CONFIG[metric]
    
    # Use combined ordering (average of both conditions)
    all_subset = df[df["scenario"] != "baseline"]
    if order_by_magnitude and not all_subset.empty:
        scenario_order = order_scenarios_by_magnitude(df, metric, order_by_magnitude)
    else:
        scenario_order = SCENARIO_ORDER

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            cm_to_inch(PLOS_WIDTHS_CM["text_column"]),
            cm_to_inch(PLOS_WIDTHS_CM["text_column"]) * 1.3,
        ),
        constrained_layout=True,
        sharex=True,
    )
    axes = np.atleast_1d(axes).flatten()

    for idx, (ax, model_family) in enumerate(
        zip(axes, ["EpiLink", "Logistic"])
    ):
        _draw_box_panel(
            ax,
            df,
            metric=metric,
            model_family=model_family,
            scenario_order=scenario_order,
            show_xlabel=(idx == 1),
        )

    # Legend for conditions
    legend_elements = [
        Patch(color=CONDITION_COLORS["Matched"], alpha=0.6, label="Matched"),
        Patch(color=CONDITION_COLORS["Mismatched"], alpha=0.6, label="Mismatched"),
    ]
    fig.legend(
        handles=legend_elements,
        title="Condition",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )

    fig.supylabel(
        config["label"],
        fontsize=9,
    )

    add_panel_labels(list(axes))
    return fig


# ─── Diagnostic helpers ───────────────────────────────────────────────────────


def print_relative_change_summary(df: pd.DataFrame, metric: str) -> None:
    """Print summary statistics of relative change by model family and condition."""
    config = METRIC_CONFIG[metric]
    loss_col = config["loss_column"]

    LOGGER.info(f"\n── Relative {metric.upper()} Change Summary ──────────────────────────────────")

    for family, models in [("EpiLink", EPILINK_MODELS), ("Logistic", LOGISTIC_MODELS)]:
        LOGGER.info(f"\n  {family} family:")
        family_df = df[df["model"].isin(models) & (df["scenario"] != "baseline")]
        
        for condition in ["Matched", "Mismatched"]:
            subset = family_df[family_df["condition"] == condition]
            if subset.empty:
                continue
            
            LOGGER.info(f"    {condition} condition:")
            summary = (
                subset.groupby("scenario")[loss_col]
                .agg(["mean", "std", "min", "max"])
                .reindex(SCENARIO_ORDER)
            )
            with pd.option_context(
                "display.float_format", "{:+.4f}".format,
                "display.width", 140,
                "display.max_columns", None,
            ):
                for scenario in SCENARIO_ORDER:
                    if scenario in summary.index:
                        row = summary.loc[scenario]
                        label = SCENARIO_LABELS.get(scenario, scenario)
                        LOGGER.info(
                            f"      {label:25s}: mean={row['mean']:+.4f}, "
                            f"std={row['std']:.4f}, range=[{row['min']:+.4f}, {row['max']:+.4f}]"
                        )


# ─── Entry point ──────────────────────────────────────────────────────────────


def main(
    metric: str = "both",
    save: bool = True,
    show: bool = False,
    order_by_magnitude: bool = True,
) -> None:
    """Generate relative change sensitivity figures.

    Parameters
    ----------
    metric : str
        'ap', 'f1', or 'both'.
    save : bool
        If True, write PDF/TIFF files.
    show : bool
        If True, display figures interactively.
    order_by_magnitude : bool
        If True (default), order scenarios by mean absolute relative change.
        If False, use canonical SCENARIO_ORDER.
    """
    global SAVE_FIGURES, SHOW_PLOTS
    SAVE_FIGURES = save
    SHOW_PLOTS = show

    configure_logging(log_file=get_pipeline_log_path(CONFIG))
    configure_logging(log_file=FIGURE_LOG_PATH)

    set_plos_theme()

    ordering = "magnitude" if order_by_magnitude else "canonical"
    LOGGER.info("sensitivity_analysis: project root  = %s", PROJECT_ROOT)
    LOGGER.info("sensitivity_analysis: results root  = %s", RESULTS_ROOT)
    LOGGER.info("sensitivity_analysis: figure output = %s", FIGURE_OUTPUT_DIR)
    LOGGER.info("sensitivity_analysis: save figures  = %s", SAVE_FIGURES)
    LOGGER.info("sensitivity_analysis: scenario order = %s", ordering)

    results = read_result_table("synthetic", "results.parquet")

    results["condition"] = (
        results["condition"].map({"matched": "Matched", "mismatched": "Mismatched"})
        .fillna(results["condition"])
    )

    metrics_to_process = ["ap", "f1"] if metric == "both" else [metric]

    for met in metrics_to_process:
        LOGGER.info(f"Processing {met.upper()} metric...")
        print_relative_change_summary(results, met)

        fig = make_fig_sensitivity(results, met, order_by_magnitude=order_by_magnitude)

        if SAVE_FIGURES:
            config = METRIC_CONFIG[met]
            saved = export_figure(fig, config["file_stem"])
            for fmt, path in saved.items():
                LOGGER.info(f"Saved {fmt.upper()}: {path}")

        if SHOW_PLOTS:
            plt.show()

        plt.close(fig)

    LOGGER.info("sensitivity_analysis: done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate normalized change sensitivity analysis figures.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["ap", "f1", "both"],
        default="both",
        help="Metric to analyze: ap, f1, or both (default: both)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing figure files to disk.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after generation.",
    )
    parser.add_argument(
        "--canonical-order",
        action="store_true",
        help="Use canonical scenario order instead of ordering by effect magnitude.",
    )
    args = parser.parse_args()

    main(
        metric=args.metric,
        save=not args.no_save,
        show=args.show,
        order_by_magnitude=not args.canonical_order,
    )
