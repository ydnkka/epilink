"""Manuscript figure: held-out graph-clustering metrics at frozen M=0 settings."""

from __future__ import annotations

import argparse

from manuscript_bars import (
    GRAPH_APPROACHES,
    graph_variants,
    metric_legend,
    plot_grouped_bars,
)
from manuscript_common import (
    PROCESS_LABELS,
    PROCESSES,
    add_arguments,
    load_run,
    output_directory,
    selected_summary,
)
from manuscript_plots import APPROACHES

from epilink_evaluation.utils import style


def create_figure(run, config, output, *, fmt="both") -> None:
    summary, points = selected_summary(run, config)
    fig, axes = style.new_figure(
        width="double",
        height_in=6.6,
        nrows=len(GRAPH_APPROACHES),
        ncols=2,
        layout="constrained",
        sharey=True,
    )
    for row, approach in enumerate(GRAPH_APPROACHES):
        for col, process in enumerate(PROCESSES):
            ax = axes[row, col]
            plot_grouped_bars(ax, graph_variants(approach, process), summary, points)
            if row == 0:
                ax.set_title(f"{PROCESS_LABELS[process]} observed genetics")
            if col == 0:
                ax.set_ylabel(APPROACHES[approach] + "\n(%)")
    fig.suptitle("Graph clustering at M=0 settings", fontweight="bold")
    fig.legend(
        handles=metric_legend(), loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=4
    )
    style.add_panel_labels(axes)
    paths = style.save_figure(
        fig,
        output / "graph_cluster_operating_bars",
        width="double",
        save_pdf=fmt in ("pdf", "both"),
        save_svg=fmt in ("svg", "both"),
    )
    for path in paths.values():
        print(f"Figure saved to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, figure=True)
    args = parser.parse_args()
    run, config = load_run(args.run_dir, evaluation=True)
    print(f"Using run: {run}")
    create_figure(run, config, output_directory(run, args.output_dir), fmt=args.format)


if __name__ == "__main__":
    main()
