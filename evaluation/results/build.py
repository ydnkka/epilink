"""Rebuild figures 01–20 in order and the four LaTeX tables from saved results."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for study in ("diagnostic", "baseline", "perturbation", "boston"):
        parser.add_argument(f"--{study}-run", type=Path, help=f"Pin the {study} run")
    parser.add_argument("--format", choices=("pdf", "png", "both"), default="both")
    args = parser.parse_args()
    groups = (
        (args.diagnostic_run, ("fig01", "fig02")),
        (
            args.baseline_run,
            (
                "fig03",
                "fig04",
                "fig05",
                "fig06",
                "fig07",
                "fig08",
                "fig09",
                "fig10",
                "fig11",
                "fig12",
                "tab01",
                "tab02",
            ),
        ),
        (
            args.perturbation_run,
            (
                "fig13",
                "fig14",
                "fig15",
                "fig16",
                "tab03",
            ),
        ),
        (args.boston_run, ("fig17", "fig18", "fig19", "fig20", "tab04")),
    )
    # Separate processes release each figure's raster memory before the next display.
    for run, modules in groups:
        for module in modules:
            command = [sys.executable, "-m", f"evaluation.results.{module}"]
            if run is not None:
                command.extend(("--run-dir", str(run)))
            if module.startswith("fig"):
                command.extend(("--format", args.format))
            print(f"Building {module}", flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
