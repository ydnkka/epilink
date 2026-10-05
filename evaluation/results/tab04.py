"""Manuscript LaTeX table: Boston exposure recovery and concentration at frozen settings (tab04)."""

from __future__ import annotations

import argparse

from ._boston.common import FOCUS, BostonStudy, add_arguments, method_label

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
                    exposure,
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
            "Boston exposure composition under baseline-frozen balanced M=0 settings "
            f"for {study.inputs['n_cases']} cases ({focus}). Each exposure's representative "
            "eligible cluster has the largest exposed-case count within that setting; "
            "the setting itself was selected on synthetic development data, not Boston. "
            "Selected reports exposed cases in that cluster over all cases carrying "
            "the exposure label. Concentration is the percentage of cases in the "
            "representative cluster with that exposure; recovery is the percentage "
            "of all exposed cases in that one cluster. Cluster count includes "
            "singletons; largest is the largest cluster size in cases. "
            "The graph score table is distance-censored; observed candidate-pair "
            f"coverage is {study.inputs['n_observed_pairs']:,} / "
            f"{study.inputs['n_all_pairs']:,}, while raw and dated trees use "
            "uncensored alignment-based distances. These exposure summaries are "
            "descriptive, not transmission precision or recall."
        ),
        short_caption="Boston exposure composition at frozen settings",
        label="tab:boston-frozen-exposures",
        row_columns=[
            "Frozen pipeline",
            "Exposure",
            "Exposed n / total",
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
