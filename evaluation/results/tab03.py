"""LaTeX table: absolute performance on fresh unperturbed control seeds (tab03)."""

from __future__ import annotations

import argparse

import pandas as pd

from epilink_evaluation.utils.latex_tables import write_latex_grouped_column_table

from ._baseline.common import TREE_KIND_LABELS, TREE_METHOD_LABELS, WEIGHT_LABELS
from ._perturbation.common import (
    FOCUS_CLUSTER,
    FOCUS_PAIR,
    PROCESS_LABELS,
    PROCESSES,
    SCORE_LABELS,
    Study,
    add_arguments,
    cluster_summary,
    pair_summary,
)


def describe_setting(point: dict) -> str:
    definition = point["definition"]
    cutoff = definition.get("threshold")
    if cutoff is None:
        result = "empty"
    elif definition["kind"] == "treecluster":
        suffix = "substitutions/site" if definition["tree_kind"] == "raw" else "days"
        result = f"{cutoff:g} {suffix}"
    elif definition["score_name"].startswith("GD_"):
        result = f"GD <= {cutoff:g} SNP"
    else:
        result = f"score >= {cutoff:g}"
    if definition["kind"] == "leiden":
        result += f"; resolution {definition['resolution']:g}"
    return result


def metric_cell(row: pd.Series, metric: str, seeds: int) -> str:
    count = int(row[f"{metric}_count"])
    if not count:
        return "--"
    mean = row[f"{metric}_mean"]
    if pd.isna(mean):
        raise ValueError(f"Missing control mean for {metric} with count {count}")
    result = f"{100 * mean:.1f}"
    sd = row[f"{metric}_std"]
    if pd.notna(sd):
        result += f" ({100 * sd:.1f})"
    if count != seeds:
        result += f" [n={count}]"
    return result


def control_row(
    frame: pd.DataFrame,
    key: str,
    identifier: str,
    study: Study,
    *,
    criterion: str | None = None,
) -> pd.Series:
    subset = frame.loc[
        (frame.scenario == "baseline")
        & (frame["mode"] == "matched")
        & (frame[key] == identifier)
    ]
    if criterion is not None:
        subset = subset.loc[subset.criterion == criterion]
    if len(subset) != 1 or subset.n_realizations.iloc[0] != len(study.seeds):
        raise ValueError(f"Incomplete fresh unperturbed control: {identifier}")
    if (
        criterion is not None
        and subset.setting_id.iloc[0] != study.point(identifier)["setting_id"]
    ):
        raise ValueError(f"Control differs from frozen setting: {identifier}")
    return subset.iloc[0]


def pipeline_label(point: dict) -> str:
    definition = point["definition"]
    pipeline = point["pipeline"]
    if definition["kind"] == "treecluster":
        method = TREE_METHOD_LABELS[definition["method"]]
        return f"TreeCluster {TREE_KIND_LABELS[definition['tree_kind']]} ({method})"
    score = SCORE_LABELS[definition["score_name"]]
    if pipeline.startswith("leiden/"):
        return f"Leiden {score} ({WEIGHT_LABELS[definition['weight_policy']]})"
    return f"Pairwise {score}"


def build_rows(study: Study) -> list[list[str]]:
    rank = pair_summary(study, delta=False)
    results = cluster_summary(study, ("M0_f1", "Mge3_contamination"), delta=False)
    rows = []
    for process in PROCESSES:
        for score in FOCUS_PAIR[process]:
            point = study.point(f"pairwise/{score}")
            control = control_row(rank, "score_name", score, study)
            rows.append(
                [
                    PROCESS_LABELS[process],
                    "Pairwise",
                    SCORE_LABELS[score],
                    describe_setting(point),
                    metric_cell(control, "M0_AP", len(study.seeds)),
                    "--",
                    "--",
                ]
            )
        for pipeline in FOCUS_CLUSTER[process]:
            point = study.point(pipeline)
            control = control_row(
                results, "pipeline", pipeline, study, criterion="balanced_M0"
            )
            rows.append(
                [
                    PROCESS_LABELS[process],
                    "Cluster",
                    pipeline_label(point),
                    describe_setting(point),
                    "--",
                    metric_cell(control, "M0_f1", len(study.seeds)),
                    metric_cell(control, "Mge3_contamination", len(study.seeds)),
                ]
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    study = Study.load(args.run_dir)
    rows = build_rows(study)
    path = write_latex_grouped_column_table(
        study.output(args.output_dir) / "tab03_fresh_control_performance.tex",
        caption=(
            "Performance on the fresh unperturbed controls used for sensitivity "
            "analysis. The target is direct transmission or infection from a shared "
            "source (M=0). AP is average precision across score cutoffs. Cluster F1 "
            "and distant-pair contamination use settings selected in the baseline "
            "development study and include every within-cluster pair. Distant pairs "
            "have M>=3. Values are equally weighted means (sample SD) in percent "
            "across three new observation realisations on the same transmission tree. "
            "These controls are paired with each perturbed scenario; they are "
            "separate from the earlier held-out observations. Dashes indicate "
            "metrics that do not apply to that analysis level."
        ),
        short_caption="Fresh unperturbed controls for sensitivity comparisons",
        label="tab:perturbation-control",
        row_columns=["Genetic observations", "Level", "Method", "Selected setting"],
        column_groups=[("Control performance (%)", ["AP", "F1", "Distant pairs"])],
        rows=rows,
        column_spec="llllrrr",
        addlinespace_after={2, 7, 10},
        landscape=True,
    )
    print(f"Table saved to: {path} (run: {study.run.name})")


if __name__ == "__main__":
    main()
