"""Supplementary LaTeX longtable: every selected frozen M=0 pipeline (tab02)."""

from __future__ import annotations

import argparse

from epilink_evaluation.utils.latex_tables import write_latex_longtable

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


def build_rows(summary, points, config):
    rows = []
    order = {name: index for index, name in enumerate(PROCESSES)}
    kind_order = {"pairwise": 0, "components": 1, "leiden": 2, "treecluster": 3}
    selected = sorted(
        (point for point in points.values() if point["status"] == "selected"),
        key=lambda point: (
            order[point["definition"]["data_process"]],
            kind_order[point["definition"]["kind"]],
            point["pipeline"],
        ),
    )
    for point in selected:
        definition = point["definition"]
        data = summary.loc[point["pipeline"]]
        rows.append(
            [
                PROCESS_LABELS[definition["data_process"]],
                method_label(definition),
                setting_label(definition, config),
                format_metric(data, "M0_precision"),
                format_metric(data, "M0_recall"),
                format_metric(data, "M0_f1"),
                format_metric(data, "Mge3_contamination"),
                format_metric(data, "direct_edge_retention"),
                format_metric(data, "shared_infector_retention"),
                format_metric(data, "selected_pairs", percent=False),
                format_metric(data, "singleton_fraction"),
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
    process_boundary = sum(row[0] == PROCESS_LABELS[PROCESSES[0]] for row in rows) - 1
    path = write_latex_longtable(
        output_directory(run, args.output_dir) / "tab02_operating_points_full.tex",
        caption=(
            "Complete comparison of methods for identifying direct transmission and "
            "infection from a shared source ($M=0$) in synthetic evaluation data. "
            "Settings were chosen to maximise the mean $F_1$ score in development "
            "data and applied without changes. Values are means (sample standard "
            "deviations) across three sets of observations from the same transmission "
            "tree, with each set given equal weight. Values are percentages except "
            "for the number of selected pairs. For clustering methods, every pair "
            r"of cases in the same cluster is evaluated. Distant pairs ($M\ge3$) "
            "account for only some of the pairs incorrectly identified as $M=0$. "
            "Direct and shared recovery give the percentages of true direct "
            "transmission pairs and shared source pairs identified by each method. "
            "The singleton column shows the percentage of all cases that form a "
            "cluster of one. The largest cluster column shows the percentage of "
            "all cases in the largest cluster. Cutoffs "
            "for undated trees are given in SNP counts and cutoffs for dated trees "
            "in days. A dash indicates that a value cannot be calculated."
        ),
        caption_is_latex=True,
        headers_are_latex=True,
        short_caption="Complete method comparison in synthetic evaluation data",
        label="tab:baseline_operating_points_full",
        columns=[
            "Genetics",
            "Method",
            "Setting",
            "Precision",
            "Recall",
            "$F_1$",
            "Distant pairs",
            "Direct recovery",
            "Shared recovery",
            "Pairs",
            "Singletons",
            "Largest",
        ],
        rows=rows,
        column_spec="lllrrrrrrrrr",
        addlinespace_after={process_boundary},
        landscape=True,
        tiny=True,
    )
    print(f"Table saved to: {path} (run: {run.name}; {len(rows)} pipelines)")


if __name__ == "__main__":
    main()
