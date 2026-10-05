"""Main-text LaTeX table: prespecified M=0 held-out comparisons (tab01)."""

from __future__ import annotations

import argparse

from ._baseline.common import (
    PROCESS_LABELS,
    PROCESSES,
    add_arguments,
    format_metric,
    load_run,
    method_label,
    output_directory,
    selected_summary,
    setting_label,
)

from epilink_evaluation.utils.latex_tables import write_latex_grouped_column_table


def main_pipelines(process: str) -> list[str]:
    suffix = "D" if process == "deterministic" else "S"
    matched = "EDD" if process == "deterministic" else "ESS"
    scorers = (matched, f"GD_{suffix}", f"LOGIT_{suffix}")
    return [
        *(f"pairwise/{scorer}" for scorer in scorers),
        *(f"leiden/{scorer}/binary" for scorer in scorers),
        f"treecluster/{process}/raw",
        f"treecluster/{process}/dated",
    ]


def build_rows(summary, points, config):
    rows = []
    for process in PROCESSES:
        for pipeline in main_pipelines(process):
            point = points.get(pipeline)
            if point is None or point["status"] != "selected":
                raise ValueError(
                    f"Main-text method lacks a frozen M=0 setting: {pipeline}"
                )
            definition = point["definition"]
            if definition["data_process"] != process:
                raise ValueError(f"Incorrect observed genetics for {pipeline}")
            data = summary.loc[pipeline]
            rows.append(
                [
                    PROCESS_LABELS[process],
                    method_label(definition),
                    setting_label(definition, config),
                    format_metric(data, "M0_precision"),
                    format_metric(data, "M0_recall"),
                    format_metric(data, "M0_f1"),
                    format_metric(data, "Mge3_contamination"),
                    f"{format_metric(data, 'direct_edge_retention')} / "
                    f"{format_metric(data, 'shared_infector_retention')}",
                    format_metric(data, "selected_pairs", percent=False),
                    format_metric(data, "largest_cluster_fraction"),
                ]
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    run, config = load_run(args.run_dir, evaluation=True)
    summary, points = selected_summary(run, config)
    rows = build_rows(summary, points, config)
    path = write_latex_grouped_column_table(
        output_directory(run, args.output_dir) / "tab01_operating_points.tex",
        caption=(
            "Frozen balanced M=0 operating points on held-out observations. "
            "Prespecified matched EpiLink, genetic distance, and logistic pairwise "
            "and binary Leiden comparisons, plus raw and dated TreeCluster. "
            "Values are equal-realization means (sample SD); ratios are percentages. "
            "AD0 / CA00 denotes direct-transmission / shared-infector retention. "
            "Pairs are all selected unordered pairs; largest is the percentage of "
            "cases in the largest cluster. Dashes denote undefined metrics. "
            "Observed processes are distinct; three evaluation realizations share one backbone."
        ),
        short_caption="Held-out M=0 operating points",
        label="tab:baseline-operating-points",
        row_columns=["Observed", "Method", "Frozen cutoff"],
        column_groups=[
            ("M=0 recovery", ["Precision", "Recall", "F1"]),
            ("Contamination and workload", ["M>=3", "AD0 / CA00", "Pairs", "Largest"]),
        ],
        rows=rows,
        column_spec="lllrrrrrrr",
        addlinespace_after={len(main_pipelines(PROCESSES[0])) - 1},
        landscape=True,
    )
    print(f"Table saved to: {path} (run: {run.name})")


if __name__ == "__main__":
    main()
