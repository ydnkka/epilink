"""Supplementary Boston figures: all frozen graph/tree partition agreements (fig21)."""

from __future__ import annotations

import argparse

import matplotlib as mpl
import numpy as np

from epilink_evaluation.utils import style

from ._boston.common import CRITERIA, BostonStudy, add_arguments, method_label


def agreement_grid(study: BostonStudy, criterion: str):
    points = study.selected_points(criterion)
    graph = sorted(
        (point for point in points if point["definition"]["kind"] != "treecluster"),
        key=lambda point: point["pipeline"],
    )
    trees = sorted(
        (point for point in points if point["definition"]["kind"] == "treecluster"),
        key=lambda point: (
            {"raw": 0, "dated": 1}[point["definition"]["tree_kind"]],
            point["definition"]["baseline_data_process"],
        ),
    )
    if not graph or not trees:
        raise ValueError(f"Missing frozen graph or tree comparators for {criterion}")
    values = {
        "adjusted_rand": np.empty((len(graph), len(trees))),
        "adjusted_mutual_information": np.empty((len(graph), len(trees))),
    }
    for i, model in enumerate(graph):
        for j, tree in enumerate(trees):
            rows = study.agreement.loc[
                (study.agreement.setting_id == model["setting_id"])
                & (study.agreement.tree_setting_id == tree["setting_id"])
            ]
            if len(rows) != 1:
                raise ValueError(
                    f"Missing frozen graph/tree comparison: {model['pipeline']}"
                )
            for metric, matrix in values.items():
                matrix[i, j] = rows[metric].iloc[0]
    if not all(np.isfinite(matrix).all() for matrix in values.values()):
        raise ValueError("Undefined agreement among frozen Boston partitions")
    labels = [method_label(point) for point in graph]
    tree_labels = [
        f"{'Undated' if point['definition']['tree_kind'] == 'raw' else 'Dated'}\n"
        f"[{point['definition']['baseline_data_process'][0].upper()} source]"
        for point in trees
    ]
    return values, labels, tree_labels


def create_figure(
    study: BostonStudy, criterion: str, output, *, fmt: str = "both"
) -> None:
    values, labels, tree_labels = agreement_grid(study, criterion)
    fig, axes = style.new_figure(
        width="double",
        width_in=10,
        height_in=8.2,
        ncols=2,
        layout="constrained",
        sharey=True,
    )
    for ax, metric, title, show_labels in zip(
        axes,
        ("adjusted_rand", "adjusted_mutual_information"),
        ("Adjusted Rand index", "Adjusted mutual information"),
        (True, False),
    ):
        matrix = values[metric]
        bound = max(0.1, float(np.abs(matrix).max()))
        image = ax.imshow(
            matrix,
            cmap=mpl.colormaps["RdBu"],
            vmin=-bound,
            vmax=bound,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xticks(range(len(tree_labels)), tree_labels)
        ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
        ax.tick_params(axis="y", labelleft=show_labels)
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if abs(value) > bound * 0.65 else "0.12",
            )
        fig.colorbar(image, ax=ax, shrink=0.6, label=title)
    # endpoint = {"balanced_M0": "M=0", "balanced_Mle1": "M≤1", "balanced_Mle2": "M≤2"}[
    #     criterion
    # ]
    # fig.suptitle(
    #     f"Boston partition agreement for {endpoint}-selected settings",
    #     fontweight="bold",
    # )
    style.add_panel_labels(axes)
    paths = style.save_figure(
        fig,
        output / f"fig21_boston_all_agreement_{criterion}",
        width="double",
        width_in=10,
        save_pdf=fmt in ("pdf", "both"),
        save_png=fmt in ("png", "both"),
    )
    for path in paths.values():
        print(f"Figure saved to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, figure=True)
    parser.add_argument("--criterion", choices=("all", *CRITERIA), default="all")
    args = parser.parse_args()
    study = BostonStudy.load(args.run_dir)
    print(f"Using run: {study.run}")
    criteria = CRITERIA if args.criterion == "all" else (args.criterion,)
    for criterion in criteria:
        create_figure(study, criterion, study.output(args.output_dir), fmt=args.format)


if __name__ == "__main__":
    main()
