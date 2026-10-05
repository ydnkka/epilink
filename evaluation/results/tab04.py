"""Manuscript LaTeX table: Boston exposure recovery and concentration at frozen settings (tab04)."""

from __future__ import annotations

import argparse

from ._boston.common import EXPOSURE_LABELS, FOCUS, BostonStudy, add_arguments, method_label

from epilink_evaluation.utils.latex_tables import write_latex_grouped_column_table


def build_rows(study: BostonStudy) -> list[list[str]]:
    rows = []
    for _, pipeline, _ in FOCUS:
        point = study.point(pipeline)
        partition = study.partition(point)
        for exposure in study.config["assessment"]["focus_exposures"]:
            named = study.representative(point, exposure)
            rows.append(
                [
                    method_label(point),
                    EXPOSURE_LABELS.get(exposure, exposure),
                    f"{int(named.n_exposure)} / {study.exposure_totals[exposure]}"
                    if named is not None
                    else "--",
                    f"{int(named.n_cases)}" if named is not None else "--",
                    f"{int(partition.n_clusters)}",
                    f"{int(partition.n_singleton_cases)}",
                    f"{int(partition.largest_cluster)}",
                    f"{100 * named.exposure_fraction:.1f}"
                    if named is not None
                    else "--",
                    f"{100 * named.exposure_recovery:.1f}"
                    if named is not None
                    else "--",
                ]
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    study = BostonStudy.load(args.run_dir)
    focus = ", ".join(
        f"{exposure} {study.exposure_totals[exposure]}"
        for exposure in study.config["assessment"]["focus_exposures"]
    )
    rows = build_rows(study)
    path = write_latex_grouped_column_table(
        study.output(args.output_dir) / "tab04_boston_frozen_exposures.tex",
        caption=(
            f"Exposure concentration and recovery among {study.inputs['n_cases']} "
            f"Boston cases ({focus}; SNF denotes skilled nursing facility). Methods "
            "use settings selected for M=0 in synthetic development observations, "
            "without adjustment using Boston exposure labels. For each exposure, "
            "the representative cluster contains the most labelled cases among "
            "clusters with at least two cases. Labelled n / total gives its labelled "
            "case count over all cases with that exposure. Concentration is the "
            "percentage of that cluster carrying the label; recovery is the "
            "percentage of the exposure group captured by that one cluster. "
            "Cluster counts include singletons, and largest n is the largest "
            "cluster's case count. Graph methods use the distance-censored TN93 "
            "observations; trees use the sequence alignment. D/S scorer labels "
            "identify synthetic development conditions; all methods receive the "
            "same empirical observations. These summaries describe exposure "
            "correspondence rather than validated transmission accuracy."
        ),
        short_caption="Boston exposure concentration and recovery",
        label="tab:boston-frozen-exposures",
        row_columns=[
            "Method",
            "Exposure",
            "Labelled n / total",
            "Cluster n",
            "Clusters",
            "Singleton cases",
            "Largest n",
        ],
        column_groups=[("Representative cluster (%)", ["Concentration", "Recovery"])],
        rows=rows,
        column_spec="llrrrrrrr",
        addlinespace_after={2 * index - 1 for index in range(1, len(FOCUS))},
        landscape=True,
    )
    print(f"Table saved to: {path} (run: {study.run.name})")


if __name__ == "__main__":
    main()
