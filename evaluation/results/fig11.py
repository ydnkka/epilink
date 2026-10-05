"""Manuscript figure: paired M=0 AP changes for focused pairwise scorers (fig11)."""

from __future__ import annotations

import argparse

from epilink_evaluation.utils import style

from ._perturbation.common import (
    FOCUS_PAIR,
    PROCESS_LABELS,
    PROCESSES,
    SCORE_LABELS,
    Study,
    add_arguments,
    pair_summary,
)
from ._perturbation.plotting import delta_heatmap, symmetric_bound


def create_figure(
    study: Study,
    output,
    *,
    fmt: str = "both",
    all_scorers: bool = False,
    endpoint: str = "M0",
) -> None:
    if endpoint not in ("M0", "Mle1", "Mle2"):
        raise ValueError(f"Unknown relationship endpoint: {endpoint}")
    frame = pair_summary(study, f"{endpoint}_AP")
    scores_by_process = (
        {
            "deterministic": ("EDD", "ESD", "GD_D", "LOGIT_D"),
            "stochastic": ("EDS", "ESS", "GD_S", "LOGIT_S"),
        }
        if all_scorers
        else FOCUS_PAIR
    )
    matrices = [
        study.matrix(
            frame,
            key="score_name",
            metric=f"{endpoint}_AP",
            identifiers=scores_by_process[process],
        )
        for process in PROCESSES
    ]
    bound = symmetric_bound(*(values for values, _ in matrices))
    fig, axes = style.new_figure(
        width="double",
        height_in=6.3,
        ncols=2,
        layout="constrained",
        sharey=True,
    )
    for ax, process, (values, counts) in zip(axes, PROCESSES, matrices):
        scores = scores_by_process[process]
        image = delta_heatmap(
            ax,
            study,
            values,
            counts,
            [SCORE_LABELS[score] for score in scores],
            bound=bound,
            improvement_positive=True,
            labels_on_left=process == PROCESSES[0],
        )
        ax.set_title(f"{PROCESS_LABELS[process]} genetic observations")
        ax.set_xlabel("Pairwise model")
    # endpoint_label = {"M0": "M=0", "Mle1": "M≤1", "Mle2": "M≤2"}[endpoint]
    # fig.suptitle(
    #     f"Paired change in {endpoint_label} average precision", fontweight="bold"
    # )
    fig.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        shrink=0.7,
        label="Perturbed − control (percentage points)",
    )
    style.add_panel_labels(axes)
    name = (
        f"fig16_pairwise_ap_all_{endpoint}"
        if all_scorers
        else "fig11_pairwise_ap_sensitivity"
    )
    paths = style.save_figure(
        fig,
        output / name,
        width="double",
        save_pdf=fmt in ("pdf", "both"),
        save_png=fmt in ("png", "both"),
    )
    for path in paths.values():
        print(f"Figure saved to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, figure=True)
    args = parser.parse_args()
    study = Study.load(args.run_dir)
    print(f"Using run: {study.run}")
    create_figure(study, study.output(args.output_dir), fmt=args.format)


if __name__ == "__main__":
    main()
