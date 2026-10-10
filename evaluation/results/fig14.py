"""EpiLink clustering paired mean/ranges for the four modes (fig14)."""

from __future__ import annotations

import argparse

import numpy as np
from matplotlib.lines import Line2D

from epilink_evaluation.utils import style

from ._perturbation.common import MODE_LABELS, PROCESS_LABELS, PROCESSES, Study, add_arguments, cluster_summary, scenario_axis
from ._perturbation.plotting import MODEL_COLORS, symmetric_bound


def create_figure(study: Study, output, *, fmt="both"):
    metrics = ("M0_f1", "Mge3_contamination")
    frame = cluster_summary(study, metrics)
    panels = study.panels
    width = max(style.FIG_WIDTHS_IN["double"], 3.3 * len(panels))
    # Validate joins and settings before displaying saved seed-level ranges.
    for metric in metrics:
        for _, pipeline in panels:
            for mode in study.config["modes"]:
                study.matrix(frame, identifiers=(pipeline,), metric=metric, mode=mode)
    fig, axes = style.new_figure(width="double", width_in=width, height_in=8.6, nrows=2, ncols=len(panels), squeeze=False, layout="constrained")
    for row, metric in enumerate(metrics):
        bound = symmetric_bound(frame[[f"delta_{metric}_min", f"delta_{metric}_max"]].to_numpy() * 100)
        for col, (process, pipeline) in enumerate(panels):
            ax = axes[row, col]
            for index, mode in enumerate(study.config["modes"]):
                group = frame.loc[frame["mode"].eq(mode) & frame.pipeline.eq(pipeline)].set_index("scenario").loc[study.scenario_names]
                mean, low, high = (group[f"delta_{metric}_{stat}"].to_numpy(float) * 100 for stat in ("mean", "min", "max"))
                valid = np.isfinite(mean)
                if np.any(valid & ((mean < low - 1e-10) | (mean > high + 1e-10))):
                    raise ValueError("Inconsistent paired mean/range")
                position = np.arange(len(group)) + (index - 1.5) * 0.15
                ax.errorbar(mean[valid], position[valid], xerr=np.vstack((np.maximum(0, mean[valid] - low[valid]), np.maximum(0, high[valid] - mean[valid]))), fmt="o", color=MODEL_COLORS[index], markersize=3.5, capsize=2)
                for y, count in zip(position, group[f"delta_{metric}_count"]):
                    if count < len(study.seeds):
                        ax.text(bound * 0.96, y, f"n={count}", ha="right", va="center", fontsize=6, color=MODEL_COLORS[index])
            scenario_axis(ax, study, show_labels=col == 0)
            ax.axvline(0, color="0.4", linestyle="--", lw=0.8)
            ax.set_xlim(-bound, bound)
            ax.grid(axis="x", color="0.9")
            ax.set_xlabel(("Δ M=0 $F_1$" if row == 0 else "Δ M≥3 contamination") + " (percentage points)")
            if row == 0:
                ax.set_title(f"{pipeline.split('/')[1]}\n{PROCESS_LABELS[process]} genetic observations")
    fig.legend(handles=[Line2D([0], [0], color=MODEL_COLORS[i], marker="o", linestyle="none", label=MODE_LABELS[mode].replace("\n", " / ")) for i, mode in enumerate(study.config["modes"])], loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=2)
    style.add_panel_labels(axes)
    paths = style.save_figure(fig, output / "fig14_cluster_f1_contamination_ranges", width="double", width_in=width, save_pdf=fmt in ("pdf", "both"), save_png=fmt in ("png", "both"))
    for path in paths.values():
        print(f"Figure saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, figure=True)
    args = parser.parse_args()
    study = Study.load(args.run_dir)
    create_figure(study, study.output(args.output_dir), fmt=args.format)


if __name__ == "__main__":
    main()
