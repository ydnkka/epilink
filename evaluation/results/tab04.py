"""Manuscript LaTeX table: Boston exposure recovery and concentration at frozen settings (tab04)."""

from __future__ import annotations

import argparse

from epilink_evaluation.utils.latex_tables import write_latex_grouped_column_table

from ._boston.common import (
    EXPOSURE_LABELS,
    FOCUS,
    BostonStudy,
    add_arguments,
    method_label,
    setting_label,
)


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
                    setting_label(point),
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
    exposures = study.config["assessment"]["focus_exposures"]
    min_cluster_size = study.config["assessment"].get("min_cluster_size", 2)
    path = write_latex_grouped_column_table(
        study.output(args.output_dir) / "tab04_boston_frozen_exposures.tex",
        caption=(
            f"Exposure concentration and recovery among {study.inputs['n_cases']} "
            f"Boston cases ({focus}; SNF means skilled nursing facility). Settings "
            "were chosen for $M=0$ using synthetic development data and applied "
            "to Boston without adjustment based on exposure labels. For each "
            "exposure group, the table describes the cluster containing the most "
            f"cases from that group among clusters with at least {min_cluster_size} "
            "cases. Labelled $n$ / total shows the number of cases with that label "
            "in this cluster and in the whole data set. Concentration is the "
            "percentage of cases in this cluster with the label. Recovery is the "
            "percentage of all cases with that label found in this cluster. Cluster "
            "counts include cases that form a cluster of one. Largest $n$ is the "
            "number of cases in the largest cluster. Leiden includes every observed "
            "pair, with EpiLink or logistic regression scores giving the strength of "
            "each connection; its resolution comes from the synthetic analysis. "
            "Genetic distance is compared using connected components at the selected "
            "SNP cutoff. TN93 measurements include only pairs below the reporting "
            "cutoff; missing pairs are treated as unobserved, not as having zero "
            "distance. Trees use the complete sequence alignment. Cutoffs for "
            "undated trees keep the same SNP counts as in the baseline analysis, "
            "adjusted for the Boston alignment length. Cutoffs for dated trees are "
            "given in days. D and S refer to deterministic and stochastic conditions "
            "in the synthetic analysis, not to separate Boston measurements. These "
            "results describe how clusters correspond to reported exposures, rather "
            "than transmission accuracy."
        ),
        caption_is_latex=True,
        headers_are_latex=True,
        short_caption="Boston exposure concentration and recovery",
        label="tab:boston_frozen_exposures",
        row_columns=[
            "Method",
            "Frozen setting",
            "Exposure",
            "Labelled $n$ / total",
            "Cluster $n$",
            "Clusters",
            "Singleton cases",
            "Largest $n$",
        ],
        column_groups=[("Representative cluster (%)", ["Concentration", "Recovery"])],
        rows=rows,
        column_spec="lllrrrrrrr",
        addlinespace_after={
            len(exposures) * index - 1 for index in range(1, len(FOCUS))
        },
        landscape=True,
    )
    print(f"Table saved to: {path} (run: {study.run.name})")


if __name__ == "__main__":
    main()
