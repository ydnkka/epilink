"""Generate offspring, concentration and generation summaries of a pinned backbone."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from epilink_evaluation.provenance import write_json
from epilink_evaluation.reporting.backbone import (
    backbone_caption,
    draw_backbone,
    load_backbone_evidence,
)
from epilink_evaluation.utils import style

from ._paths import output_directory
from .fig01 import load_current_run

PREFIX = "fig24_backbone_characterisation"


def create_figure(run: Path, output: Path, *, fmt="both"):
    """Export the display and plotted evidence without recomputing diagnostics."""
    manifest, summary, tables = load_backbone_evidence(run)
    fig, axes = style.new_figure(
        width="double",
        height_in=3.2,
        ncols=3,
        layout="constrained",
    )
    draw_backbone(axes, summary, tables)
    smoke = manifest["config"]["inputs"].get("smoke_cases") is not None
    scope = "Smoke backbone" if smoke else "Selected transmission backbone"
    fig.suptitle(f"{scope}: {summary['n_cases']:,} cases", fontsize=11)
    style.add_panel_labels(axes)
    paths = style.save_figure(
        fig,
        output / PREFIX,
        width="double",
        save_pdf=fmt in ("pdf", "both"),
        save_png=fmt in ("png", "both"),
    )
    plt.close(fig)
    for name, table in tables.items():
        table.to_csv(output / f"{PREFIX}_{name}.csv", index=False)
    pd.DataFrame([{k: v for k, v in summary.items() if k != "bootstrap"}]).to_csv(
        output / f"{PREFIX}_summary.csv",
        index=False,
    )
    write_json(
        output / f"{PREFIX}.json",
        {
            "run_directory": str(Path(run).resolve()),
            "backbone": manifest["signature"]["backbone"],
            "summary": summary,
        },
    )
    (output / f"{PREFIX}.md").write_text(
        "# Transmission-backbone characterisation\n\n"
        + backbone_caption(manifest, summary)
        + f"\n\nSource diagnostics run: `{Path(run).resolve()}`.\n"
    )
    for path in paths.values():
        print(f"Figure saved to: {path}")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Pinned diagnostics run; default: diagnostics/current.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Override the run-specific results directory"
    )
    parser.add_argument("--format", choices=("pdf", "png", "both"), default="both")
    args = parser.parse_args()
    run = load_current_run(args.run_dir)
    create_figure(
        run,
        output_directory(run, "00_synthetic_diagnostics", args.output_dir),
        fmt=args.format,
    )


if __name__ == "__main__":
    main()
