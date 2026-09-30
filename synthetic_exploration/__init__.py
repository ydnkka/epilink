"""Standalone, relationship-aware exploration of the synthetic benchmark."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# The established evaluation modules use both package and sibling imports.
for path in (ROOT / "src", ROOT / "src/evaluation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
