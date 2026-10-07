"""Primary M=0 EpiLink compatibility across genetic and temporal distances."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from epilink_evaluation.scorers.registry import ScoringContext
from epilink_evaluation.utils import style

from ._baseline.common import add_arguments, load_run, output_directory

SNPS = np.arange(16)
DAYS = np.arange(21)
INFERENCE_MODELS = (
    ("deterministic", "EpiLink ED (deterministic)"),
    ("stochastic", "EpiLink ES (stochastic)"),
)


def score_surfaces(
    context: ScoringContext, snps: np.ndarray, days: np.ndarray
) -> dict[str, np.ndarray]:
    """Score the same SNP/day inputs under each EpiLink inference model."""
    snp_grid, day_grid = np.meshgrid(snps, days)
    surfaces = {}
    for process, _ in INFERENCE_MODELS:
        scores = np.asarray(
            context.epilink(process).score_target(
                sample_time_difference=day_grid.ravel(),
                genetic_distance=snp_grid.ravel(),
            ),
            dtype=float,
        )
        if (
            scores.size != snp_grid.size
            or not np.isfinite(scores).all()
            or (scores < 0).any()
        ):
            raise ValueError(f"Invalid {process} EpiLink compatibility surface")
        surfaces[process] = scores.reshape(snp_grid.shape)
    return surfaces


def create_figure(
    surfaces: dict[str, np.ndarray], output: Path, *, fmt: str = "both"
) -> None:
    """Use identical integer cells and a shared compatibility scale in both panels."""
    vmax = max(float(surface.max()) for surface in surfaces.values())
    if vmax <= 0:
        raise ValueError("Compatibility surface has no positive scores")
    fig, axes = style.new_figure(
        width="onehalf",
        height_in=2.5,
        ncols=2,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    image = None
    for ax, (process, title) in zip(axes, INFERENCE_MODELS):
        image = ax.imshow(
            surfaces[process],
            origin="lower",
            interpolation="nearest",
            aspect="equal",
            extent=(-0.5, SNPS[-1] + 0.5, -0.5, DAYS[-1] + 0.5),
            cmap="viridis",
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Genetic distance (SNPs)")
        ax.set_xticks([0, 3, 6, 9, 12, 15])
        ax.set_yticks([0, 5, 10, 15, 20])
        ax.grid(False)
    axes[0].set_ylabel("Sampling-time difference (days)")
    fig.colorbar(image, ax=axes, shrink=0.75, pad=0.02, label="M=0 compatibility score")
    style.add_panel_labels(axes)
    paths = style.save_figure(
        fig,
        output / "fig00_primary_compatibility_surfaces",
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
    run, config = load_run(args.run_dir)
    print(f"Using run: {run}")
    context = ScoringContext(config, logistic_models={})
    surfaces = score_surfaces(context, SNPS, DAYS)
    create_figure(surfaces, output_directory(run, args.output_dir), fmt=args.format)


if __name__ == "__main__":
    main()
