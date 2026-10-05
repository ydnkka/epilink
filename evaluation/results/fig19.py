"""Manuscript figure: Boston partition burden and graph/tree agreement (fig19)."""

from __future__ import annotations

import argparse

import matplotlib as mpl
import numpy as np
from ._boston.common import FOCUS, BostonStudy, add_arguments

from epilink_evaluation.utils import style


def agreement_matrix(study: BostonStudy) -> tuple[np.ndarray, np.ndarray]:
    graph = FOCUS[:3]
    tree = FOCUS[3:]
    ari = np.empty((len(graph), len(tree)), dtype=float)
    ami = np.empty_like(ari)
    for row, (_, graph_pipeline, _) in enumerate(graph):
        graph_id = study.point(graph_pipeline)["setting_id"]
        for col, (_, tree_pipeline, _) in enumerate(tree):
            tree_id = study.point(tree_pipeline)["setting_id"]
            match = study.agreement.loc[
                (study.agreement.setting_id == graph_id)
                & (study.agreement.tree_setting_id == tree_id)
            ]
            if len(match) != 1:
                raise ValueError(
                    f"Missing frozen partition agreement: {graph_pipeline}/{tree_pipeline}"
                )
            ari[row, col] = match.adjusted_rand.iloc[0]
            ami[row, col] = match.adjusted_mutual_information.iloc[0]
    if not np.isfinite(ari).all() or not np.isfinite(ami).all():
        raise ValueError("Undefined ARI/AMI between frozen Boston partitions")
    return ari, ami


def create_figure(study: BostonStudy, output, *, fmt: str = "both") -> None:
    ari, ami = agreement_matrix(study)
    fig, axes = style.new_figure(
        width="double",
        height_in=3.9,
        ncols=2,
        layout="constrained",
        gridspec_kw={"width_ratios": [1.35, 1]},
    )
    ax, heat = axes
    labels = [label for label, _, _ in FOCUS]
    positions = np.arange(len(FOCUS))
    n = study.inputs["n_cases"]
    singletons = (
        np.array(
            [
                study.partition(study.point(pipeline)).n_singleton_cases
                for _, pipeline, _ in FOCUS
            ],
            dtype=float,
        )
        / n
        * 100
    )
    largest = (
        np.array(
            [
                study.partition(study.point(pipeline)).largest_cluster
                for _, pipeline, _ in FOCUS
            ],
            dtype=float,
        )
        / n
        * 100
    )
    ax.barh(
        positions - 0.17,
        singletons,
        height=0.32,
        color="#0072B2",
        label="Singleton cases",
    )
    ax.barh(
        positions + 0.17,
        largest,
        height=0.32,
        color="#D55E00",
        label="Cases in largest cluster",
    )
    ax.set_yticks(positions, labels)
    ax.set_ylim(len(labels) - 0.6, -0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Fraction of all Boston cases (%)")
    ax.set_title("Cluster burden")
    ax.grid(axis="x", color="0.9")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=7)

    bound = max(0.1, float(np.abs(ari).max()))
    image = heat.imshow(
        ari,
        cmap=mpl.colormaps["RdBu"],
        vmin=-bound,
        vmax=bound,
        aspect="auto",
        interpolation="nearest",
    )
    heat.set_xticks(range(2), ["Raw", "Dated"])
    heat.set_yticks(range(3), [label for label, _, _ in FOCUS[:3]])
    heat.set_title("Graph–tree agreement")
    heat.set_xlabel("Frozen TreeCluster rule")
    for (row, col), value in np.ndenumerate(ari):
        heat.text(
            col,
            row,
            f"ARI {value:.2f}\nAMI {ami[row, col]:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if abs(value) > bound * 0.65 else "0.15",
        )
    fig.colorbar(image, ax=heat, shrink=0.65, label="Adjusted Rand index")
    fig.suptitle(
        "Structure and agreement of frozen Boston partitions", fontweight="bold"
    )
    style.add_panel_labels(axes)
    paths = style.save_figure(
        fig,
        output / "fig19_boston_partition_context",
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
    study = BostonStudy.load(args.run_dir)
    print(f"Using run: {study.run}")
    create_figure(study, study.output(args.output_dir), fmt=args.format)


if __name__ == "__main__":
    main()
