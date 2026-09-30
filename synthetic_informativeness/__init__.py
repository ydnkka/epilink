"""Matched-baseline informativeness analyses for synthetic EpiLink outputs."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "src/evaluation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
