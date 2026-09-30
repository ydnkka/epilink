"""Thin baseline entry point; install the shared package with pip install -e ."""
import sys

from epilink_evaluation.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["baseline", *sys.argv[1:]]))
