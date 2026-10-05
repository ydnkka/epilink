"""Manuscript figure: held-out raw and dated TreeCluster at frozen M=0 settings."""

from __future__ import annotations

import argparse

from manuscript_bars import (
    TREE_KINDS,
    metric_legend,
    plot_grouped_bars,
    treecluster_variant,
)
from manuscript_common import (
    PROCESSES,
    add_arguments,
    load_run,
    output_directory,
    selected_summary,
)

from epilink_evaluation.utils import style


def create_figure(run, config, output, *, fmt="both") -> None:
    summary, points = selected_summary(run, config)
    fig, axes = style.new_figure(
        width="double",
        height_in=3.6,
        ncols=2,
        layout="constrained",
        sharey=True,
    )
    for ax, kind in zip(axes, TREE_KINDS):
        variants = [
            treecluster_variant(kind, process, points, config) for process in PROCESSES
        ]
        plot_grouped_bars(ax, variants, summary, points)
        ax.set_title(f"{kind.title()} tree")
    axes[0].set_ylabel("Held-out metric (%)")
    fig.suptitle("TreeCluster at frozen M=0 settings", fontweight="bold")
    fig.legend(
        handles=metric_legend(), loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=4
    )
    style.add_panel_labels(axes)
    paths = style.save_figure(
        fig,
        output / "treecluster_operating_bars",
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
