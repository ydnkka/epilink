"""Supplementary LaTeX longtable: every selected frozen M=0 pipeline (tab02)."""

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

from epilink_evaluation.utils.latex_tables import write_latex_longtable


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
            "All selected balanced M=0 pipelines evaluated at their frozen settings. "
            "Equal-realization means (sample SD) across held-out observations on one "
            "transmission backbone; ratios are percentages and pairs are counts. "
            "Precision includes every within-cluster pair, not just input edges. "
            "M>=3 is distant contamination, not the entire M=0 false-positive fraction. "
            "Raw-tree thresholds use SNP counts, dated-tree thresholds use days. "
            "Dashes denote undefined metrics; n indicates a reduced defined-value count."
        ),
        short_caption="All held-out M=0 operating points",
        label="tab:baseline-operating-points-full",
        columns=[
            "Observed",
            "Method",
            "Frozen cutoff",
            "Precision",
            "Recall",
            "F1",
            "M>=3",
            "AD0",
            "CA00",
            "Pairs",
            "Singleton",
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
