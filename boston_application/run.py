"""Thin entry point for Boston empirical clustering; install with pip install -e ."""
import sys

from epilink_evaluation.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["boston", *sys.argv[1:]]))
