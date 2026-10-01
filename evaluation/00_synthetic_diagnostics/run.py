"""Thin entry point for synthetic diagnostics; install with pip install -e ."""
import sys

from epilink_evaluation.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["diagnostics", *sys.argv[1:]]))
