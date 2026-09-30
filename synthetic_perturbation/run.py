"""Thin entry point for perturbation studies; install with pip install -e ."""
import sys

from epilink_evaluation.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["perturbation", *sys.argv[1:]]))
