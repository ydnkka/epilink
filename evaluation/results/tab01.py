"""Main LaTeX table: selected methods evaluated on new synthetic observations (tab01)."""

from __future__ import annotations

import argparse

from epilink_evaluation.utils.latex_tables import write_latex_grouped_column_table

from ._baseline.common import (
    PROCESS_LABELS,
    PROCESSES,
    SCORES_BY_PROCESS,
    add_arguments,
    format_metric,
    graph_scorers,
    load_run,
    method_label,
    output_directory,
    selected_summary,
    setting_label,
)


def main_pipelines(process: str) -> list[str]:
    scorers = SCORES_BY_PROCESS[process]
    return [
        *(f"pairwise/{scorer}" for scorer in scorers),
        *(f"leiden/{scorer}" for scorer in graph_scorers("leiden", process)),
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
            "Performance of methods for identifying direct transmission and infection "
            "from a shared source ($M=0$) in synthetic evaluation data. Settings were "
            "chosen to maximise the mean $F_1$ score in development data and then "
            "applied without changes. EpiLink, genetic distance, and logistic "
            "regression identify pairs of cases; Leiden and TreeCluster group cases "
            "into clusters. Leiden uses every observed pair, with EpiLink or logistic "
            "regression scores giving the strength of each connection. Only its "
            "resolution is chosen during development. Values are means (sample "
            "standard deviations) across three sets of simulated observations from "
            "the same transmission tree, with each set given equal weight. All values "
            "are percentages. For clustering methods, every pair of cases in the "
            r"same cluster is evaluated. Distant pairs have $M\ge3$ and account for "
            "only some of the pairs incorrectly identified as $M=0$. Cutoffs for "
            "undated trees are given in SNP counts and cutoffs for dated trees in "
            "days. The complete comparison is shown in "
            r"Appendix~Table~\ref{tab:baseline_operating_points_full}."
        ),
        caption_is_latex=True,
        headers_are_latex=True,
        short_caption="Identification of recent transmission relationships",
        label="tab:baseline_operating_points",
        row_columns=["Genetic observations", "Method", "Selected setting"],
        column_groups=[
            ("Target performance (%)", ["Precision", "Recall", "$F_1$"]),
            ("Distant pairs (%)", [r"$M\ge3$"]),
        ],
        rows=rows,
        column_spec="lllrrrr",
        addlinespace_after={len(main_pipelines(PROCESSES[0])) - 1},
        landscape=True,
    )
    print(f"Table saved to: {path} (run: {run.name})")


if __name__ == "__main__":
    main()
