"""Main-text LaTeX table: prespecified M=0 held-out comparisons (tab01)."""

from __future__ import annotations

import argparse

from ._baseline.common import (
    PROCESS_LABELS,
    PROCESSES,
    SCORES_BY_PROCESS,
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
    epilink_es = "ESD" if process == "deterministic" else "ESS"
    scorers = SCORES_BY_PROCESS[process]
    return [
        *(f"pairwise/{scorer}" for scorer in scorers),
        *(f"leiden/{scorer}/binary" for scorer in scorers),
        f"leiden/{epilink_es}/native",
        f"leiden/LOGIT_{suffix}/native",
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
            "Identification of direct transmission and infection from a shared source "
            "in held-out synthetic observations. Cutoffs and clustering settings were "
            "selected by mean development F1 and applied unchanged. Both EpiLink "
            "inference formulations, genetic distance, and logistic regression are "
            "shown as pairwise rules and binary-edge Leiden inputs; score-weighted "
            "stochastic EpiLink and logistic Leiden and undated/dated TreeCluster "
            "provide additional clustering comparisons. Values are equally weighted "
            "means (sample SD) across three observation realizations on one fixed "
            "transmission tree, expressed as percentages. Cluster metrics include "
            "every within-cluster pair. Distant pairs have M>=3; this measure does "
            "not include every false positive for M=0. Tree cutoffs are shown in SNP "
            "counts for undated trees and days for dated trees. Dashes indicate "
            "undefined values. The complete method comparison is in the appendix."
        ),
        short_caption="Held-out identification of recent transmission relationships",
        label="tab:baseline-operating-points",
        row_columns=["Genetic observations", "Method", "Selected setting"],
        column_groups=[
            ("Target-pair performance (%)", ["Precision", "Recall", "F1"]),
            ("Distant pairs (%)", ["M>=3"]),
        ],
        rows=rows,
        column_spec="lllrrrr",
        addlinespace_after={len(main_pipelines(PROCESSES[0])) - 1},
        landscape=True,
    )
    print(f"Table saved to: {path} (run: {run.name})")


if __name__ == "__main__":
    main()
