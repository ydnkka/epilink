"""Supplementary figures: all eight pairwise scorers at each relationship horizon."""

from __future__ import annotations

import argparse

from manuscript_common import Study, add_arguments
from plot_pairwise_sensitivity import create_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, figure=True)
    args = parser.parse_args()
    study = Study.load(args.run_dir)
    print(f"Using run: {study.run}")
    for endpoint in ("M0", "Mle1", "Mle2"):
        create_figure(study, study.output(args.output_dir), fmt=args.format,
                      all_scorers=True, endpoint=endpoint)


if __name__ == "__main__":
    main()
