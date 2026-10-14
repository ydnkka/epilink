"""Details of every Boston cluster containing sequences with known exposures."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from epilink_evaluation.utils.latex_tables import write_latex_longtable

from ._boston.common import (
    CRITERIA,
    EXPOSURE_LABELS,
    BostonStudy,
    add_arguments,
    method_label,
    setting_label,
)

COMPOSITION_COLUMNS = (
    "setting_id",
    "pipeline",
    "baseline_setting_id",
    "cluster_id",
    "n_cases",
    "evidence_type",
    "evidence_label",
    "n_evidence",
    "fraction_in_cluster",
)


def cluster_exposure_rows(
    study: BostonStudy, criterion: str = "balanced_M0"
) -> pd.DataFrame:
    """Validate complete exposure counts before extracting all labelled clusters."""
    points = study.selected_points(criterion)
    ids = {point["setting_id"] for point in points}
    composition = pd.read_csv(
        study.run / "assessment/cluster_composition.csv",
        usecols=list(COMPOSITION_COLUMNS),
        dtype={"setting_id": str, "baseline_setting_id": str},
    )
    composition = composition.loc[
        composition.evidence_type.eq("Exposure") & composition.setting_id.isin(ids)
    ].copy()
    if set(composition.setting_id) != ids:
        raise ValueError("Incomplete Boston exposure composition for selected settings")
    if composition.duplicated(["setting_id", "cluster_id", "evidence_label"]).any():
        raise ValueError("Duplicate Boston cluster exposure counts")
    numbers = composition[["cluster_id", "n_cases", "n_evidence"]].to_numpy(float)
    if (
        not np.isfinite(numbers).all()
        or not np.equal(numbers, np.floor(numbers)).all()
        or not composition.n_evidence.gt(0).all()
        or not composition.n_cases.between(1, study.inputs["n_cases"]).all()
        or not composition.n_evidence.le(composition.n_cases).all()
        or not composition.evidence_label.isin(study.exposure_totals).all()
        or not np.allclose(
            composition.fraction_in_cluster,
            composition.n_evidence / composition.n_cases,
        )
    ):
        raise ValueError("Invalid Boston cluster exposure counts or fractions")

    metadata = []
    for order, point in enumerate(points):
        subset = composition.loc[composition.setting_id == point["setting_id"]]
        partition = study.partition(point)
        grouped = subset.groupby("cluster_id")
        sizes = grouped.n_cases.first()
        if (
            not subset.pipeline.eq(point["pipeline"]).all()
            or not subset.baseline_setting_id.eq(point["baseline_setting_id"]).all()
            or not grouped.n_cases.nunique().eq(1).all()
            or not grouped.n_evidence.sum().eq(sizes).all()
            or len(sizes) != partition.n_clusters
            or sizes.sum() != study.inputs["n_cases"]
            or sizes.eq(1).sum() != partition.n_singleton_cases
            or sizes.max() != partition.largest_cluster
            or subset.groupby("evidence_label").n_evidence.sum().to_dict()
            != study.exposure_totals
        ):
            raise ValueError(
                f"Incomplete or inconsistent Boston cluster composition: {point['pipeline']}"
            )
        metadata.append(
            {
                "setting_id": point["setting_id"],
                "criterion": criterion,
                "method": method_label(point),
                "selected_setting": setting_label(point),
                "baseline_data_process": point["definition"]["baseline_data_process"],
                "method_order": order,
            }
        )

    rows = composition.loc[composition.evidence_label.ne("Unlabeled")].rename(
        columns={"evidence_label": "exposure", "n_evidence": "n_exposure"}
    )
    rows = rows.merge(pd.DataFrame(metadata), on="setting_id", validate="many_to_one")
    rows["exposure_total"] = rows.exposure.map(study.exposure_totals)
    rows["concentration_percent"] = 100 * rows.n_exposure / rows.n_cases
    rows["recovery_percent"] = 100 * rows.n_exposure / rows.exposure_total
    rows = rows.sort_values(["method_order", "cluster_id", "exposure"])
    return rows[
        [
            "criterion",
            "pipeline",
            "setting_id",
            "baseline_setting_id",
            "baseline_data_process",
            "method",
            "selected_setting",
            "cluster_id",
            "n_cases",
            "exposure",
            "n_exposure",
            "exposure_total",
            "concentration_percent",
            "recovery_percent",
        ]
    ].reset_index(drop=True)


def build_rows(frame: pd.DataFrame) -> list[list[str]]:
    return [
        [
            row.method,
            row.selected_setting,
            str(int(row.cluster_id)),
            EXPOSURE_LABELS.get(row.exposure, row.exposure),
            str(int(row.n_cases)),
            str(int(row.n_exposure)),
            str(int(row.exposure_total)),
            f"{row.concentration_percent:.1f}",
            f"{row.recovery_percent:.1f}",
        ]
        for row in frame.itertuples(index=False)
    ]


def create_table(study: BostonStudy, output, criterion: str = "balanced_M0"):
    frame = cluster_exposure_rows(study, criterion)
    endpoint = {
        "balanced_M0": "$M=0$",
        "balanced_Mle1": "$M\\le1$",
        "balanced_Mle2": "$M\\le2$",
    }[criterion]
    boundaries = {
        index - 1
        for index in range(1, len(frame))
        if frame.setting_id.iloc[index] != frame.setting_id.iloc[index - 1]
    }
    prefix = f"tab05_boston_exposure_clusters_{criterion}"
    path = write_latex_longtable(
        output / f"{prefix}.tex",
        caption=(
            "Details of every Boston cluster containing at least one sequence with "
            f"a known exposure label, for all methods selected for {endpoint} in "
            "synthetic development data. Each row describes one cluster and one "
            "exposure group present in that cluster. Clusters of one sequence are "
            "included. Cluster IDs are specific to each method and selected setting. "
            "Cluster n counts all sequences in the cluster, including those without "
            "a known exposure. Labelled n counts sequences from the exposure group "
            "in that cluster, and group total counts all sequences from that group "
            "in Boston. Concentration is labelled n divided by cluster n; recovery "
            "is labelled n divided by group total. Both are percentages. SNF means "
            "skilled nursing facility, and BHCHP means Boston Health Care for the "
            "Homeless Program. D and S refer to conditions in the synthetic analysis. "
            "Exposure labels describe the resulting clusters and were not used to "
            "choose the clustering settings."
        ),
        caption_is_latex=True,
        short_caption="Boston clusters containing sequences with known exposures",
        label=f"tab:boston_exposure_clusters_{criterion}",
        columns=[
            "Method",
            "Selected setting",
            "Cluster ID",
            "Exposure",
            "Cluster n",
            "Labelled n",
            "Group total",
            "Concentration (%)",
            "Recovery (%)",
        ],
        rows=build_rows(frame),
        column_spec="llrlrrrrr",
        addlinespace_after=boundaries,
        landscape=True,
        tiny=True,
    )
    csv_path = output / f"{prefix}.csv"
    frame.to_csv(csv_path, index=False)
    clusters = len(frame.drop_duplicates(["setting_id", "cluster_id"]))
    print(
        f"Table saved to: {path} ({clusters} clusters across selected methods; {len(frame)} rows)"
    )
    print(f"Cluster details saved to: {csv_path}")
    return path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument(
        "--criterion", choices=("all", *CRITERIA), default="balanced_M0"
    )
    args = parser.parse_args()
    study = BostonStudy.load(args.run_dir)
    criteria = CRITERIA if args.criterion == "all" else (args.criterion,)
    for criterion in criteria:
        create_table(study, study.output(args.output_dir), criterion)


if __name__ == "__main__":
    main()
