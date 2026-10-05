"""Run-specific destinations for manuscript displays."""

from pathlib import Path


def output_directory(run: Path, study: str, override: Path | None = None) -> Path:
    """Keep results from different studies and pinned runs separate."""
    if override is not None:
        return override
    return Path(__file__).resolve().parent / "outputs" / study / run.name
