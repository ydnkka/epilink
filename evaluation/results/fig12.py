"""Paired full-graph EpiLink clustering sensitivity across all four modes (fig12)."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from epilink_evaluation.utils import style

from ._perturbation.common import MODE_LABELS, PROCESS_LABELS, PROCESSES, Study, add_arguments, cluster_summary
from ._perturbation.plotting import delta_heatmap, symmetric_bound


def create_figure(study: Study, output, *, fmt="both", stem="fig12_cluster_sensitivity"):
    metrics = ("M0_f1", "Mge3_contamination")
    frame = cluster_summary(study, metrics)
    processes = [p for p in PROCESSES if study.pipelines(p)]
    content, records = {}, []
    for process in processes:
        pipelines = study.pipelines(process)
        labels = [f"{pipeline.split('/')[1]}\n{MODE_LABELS[mode]}" for mode in study.config["modes"] for pipeline in pipelines]
        for metric in metrics:
            matrices = [study.matrix(frame, identifiers=pipelines, metric=metric, mode=mode) for mode in study.config["modes"]]
            values = np.concatenate([m[0] for m in matrices], axis=1)
            counts = np.concatenate([m[1] for m in matrices], axis=1)
            content[metric, process] = values, counts, labels
            for i, scenario in enumerate(study.scenario_names):
                for j, label in enumerate(labels):
                    records.append({"scenario": scenario, "process": process, "metric": metric, "arm": label.replace("\n", " / "), "mean_change_pp": values[i, j], "defined_realizations": counts[i, j]})
    bounds = {metric: symmetric_bound(*(content[metric, process][0] for process in processes)) for metric in metrics}
    fig, axes = style.new_figure(width="double", height_in=8.2, nrows=2, ncols=len(processes), squeeze=False, layout="constrained")
    for row, metric in enumerate(metrics):
        for col, process in enumerate(processes):
            values, counts, labels = content[metric, process]
            delta_heatmap(axes[row, col], study, values, counts, labels, bound=bounds[metric], improvement_positive=metric == "M0_f1", labels_on_left=col == 0)
            axes[row, col].tick_params(axis="x", labelsize=6)
            if row == 0:
                axes[row, col].set_title(f"{PROCESS_LABELS[process]} genetic observations")
            if col == 0:
                axes[row, col].set_ylabel("Target-pair $F_1$" if row == 0 else "Distant-pair contamination")
        fig.colorbar(axes[row, -1].images[0], ax=axes[row, :], orientation="horizontal", shrink=0.72, label=("Δ M=0 $F_1$" if row == 0 else "Δ M≥3 contamination") + " (percentage points)")
    style.add_panel_labels(axes)
    paths = style.save_figure(fig, output / stem, width="double", save_pdf=fmt in ("pdf", "both"), save_png=fmt in ("png", "both"))
    pd.DataFrame(records).to_csv(output / f"{stem}.csv", index=False)
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
