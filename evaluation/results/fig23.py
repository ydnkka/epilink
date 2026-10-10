"""Main-results overview of the four full-graph EpiLink clustering modes (fig23)."""

import argparse

from ._perturbation.common import Study, add_arguments
from .fig12 import create_figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, figure=True)
    args = parser.parse_args()
    study = Study.load(args.run_dir)
    create_figure(study, study.output(args.output_dir), fmt=args.format, stem="fig23_parameter_sensitivity_overview")


if __name__ == "__main__":
    main()
