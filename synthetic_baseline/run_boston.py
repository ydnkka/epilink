"""Thin entry point for Boston empirical clustering; install with pip install -e ."""
import sys

from epilink_evaluation.workflows.boston import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
