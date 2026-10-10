"""LaTeX table: full-graph EpiLink clustering on fresh controls in all four arms."""

import argparse

import pandas as pd

from epilink_evaluation.utils.latex_tables import write_latex_grouped_column_table

from ._perturbation.common import MODE_LABELS, PROCESS_LABELS, PROCESSES, Study, add_arguments, cluster_summary


def metric_cell(row, metric, seeds):
    count = int(row[f"{metric}_count"])
    if not count:
        return "--"
    mean, sd = row[f"{metric}_mean"], row[f"{metric}_std"]
    if pd.isna(mean):
        raise ValueError("Missing control mean with defined values")
    result = f"{100 * mean:.1f}" + (f" ({100 * sd:.1f})" if pd.notna(sd) else "")
    return result + (f" [n={count}]" if count != seeds else "")


def build_rows(study):
    frame = cluster_summary(study, ("M0_f1", "Mge3_contamination"), delta=False)
    rows = []
    for process in PROCESSES:
        for mode in study.config["modes"]:
            for pipeline in study.pipelines(process):
                study.matrix(frame, identifiers=(pipeline,), metric="M0_f1", mode=mode, delta=False)
                point = study.point("baseline", mode, pipeline)
                control = frame.loc[frame.scenario.eq("baseline") & frame["mode"].eq(mode) & frame.pipeline.eq(pipeline)].iloc[0]
                rows.append([
                    PROCESS_LABELS[process], pipeline.split("/")[1], MODE_LABELS[mode].replace("\n", " / "),
                    f"{point['definition']['resolution']:g}", metric_cell(control, "M0_f1", len(study.seeds)),
                    metric_cell(control, "Mge3_contamination", len(study.seeds)),
                ])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    study = Study.load(args.run_dir)
    path = write_latex_grouped_column_table(
        study.output(args.output_dir) / "tab03_fresh_control_performance.tex",
        caption="Full-graph, score-weighted EpiLink Leiden clustering on fresh unperturbed controls. Four arms cross baseline/matched inference with baseline/updated resolution. Updated resolutions use separate development observations. Values are equal-seed means (sample SD) in percent on the paired evaluation observations. The target is $M=0$; distant-pair contamination is $M\\ge3$. All within-cluster pairs are evaluated.",
        caption_is_latex=True, headers_are_latex=True, short_caption="Fresh controls for EpiLink clustering sensitivity",
        label="tab:perturbation_control", row_columns=["Genetic observations", "EpiLink", "Arm", "Resolution"],
        column_groups=[("Control performance (%)", ["$F_1$", "Distant pairs"])],
        rows=build_rows(study), column_spec="llllrr", landscape=True,
    )
    print(f"Table saved to: {path} (run: {study.run.name})")


if __name__ == "__main__":
    main()
