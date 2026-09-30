"""Shared definitions for the synthetic informativeness workflow."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_exploration.common import save_table, write_json, log


@dataclass(frozen=True)
class Endpoint:
    key: str
    label: str
    horizon: int


ENDPOINTS = (
    Endpoint("M0", "M == 0", 0),
    Endpoint("Mle1", "M <= 1", 1),
    Endpoint("Mle2", "M <= 2", 2),
)

M_CATEGORIES = ("M0", "M1", "M2", "Mge3", "undefined_M")


def endpoint_mask(pairs: pd.DataFrame, endpoint: Endpoint) -> np.ndarray:
    """Return the positive-label mask for a near-transmission endpoint."""
    m = pairs.M.to_numpy(dtype=float, na_value=np.nan)
    return np.isfinite(m) & (m <= endpoint.horizon)


def contamination_mask(pairs: pd.DataFrame) -> np.ndarray:
    """Pairs with three or more total intermediates are negative contamination."""
    m = pairs.M.to_numpy(dtype=float, na_value=np.nan)
    return np.isfinite(m) & (m >= 3)


def m_category_counts(pairs: pd.DataFrame, mask: np.ndarray | None = None) -> dict[str, int]:
    """Count selected pairs in the M categories used throughout the report."""
    if mask is None:
        mask = np.ones(len(pairs), dtype=bool)
    m = pairs.M.to_numpy(dtype=float, na_value=np.nan)
    selected = np.asarray(mask, dtype=bool)
    return {
        "M0": int(np.sum(selected & (m == 0))),
        "M1": int(np.sum(selected & (m == 1))),
        "M2": int(np.sum(selected & (m == 2))),
        "Mge3": int(np.sum(selected & np.isfinite(m) & (m >= 3))),
        "undefined_M": int(np.sum(selected & ~np.isfinite(m))),
    }


def m_category_rows(base: dict, pairs: pd.DataFrame, mask: np.ndarray | None = None) -> list[dict]:
    counts = m_category_counts(pairs, mask)
    total = sum(counts.values())
    return [
        {**base, "M_category": category, "n_pairs": count,
         "proportion": count / total if total else np.nan}
        for category, count in counts.items()
    ]


def selected_m_summary(pairs: pd.DataFrame, mask: np.ndarray) -> dict[str, float]:
    """Compact M summaries for selected pair sets or within-cluster pair sets."""
    counts = m_category_counts(pairs, mask)
    total = int(np.sum(mask))
    m = pairs.M.to_numpy(dtype=float, na_value=np.nan)
    finite = mask & np.isfinite(m)
    return {
        "selected_pairs": total,
        "M0_fraction": counts["M0"] / total if total else np.nan,
        "Mle1_fraction": (counts["M0"] + counts["M1"]) / total if total else np.nan,
        "Mle2_fraction": (counts["M0"] + counts["M1"] + counts["M2"]) / total if total else np.nan,
        "Mge3_contamination_fraction": counts["Mge3"] / total if total else np.nan,
        "undefined_M_fraction": counts["undefined_M"] / total if total else np.nan,
        "median_M_connected": float(np.median(m[finite])) if np.any(finite) else np.nan,
        "p90_M_connected": float(np.quantile(m[finite], 0.9)) if np.any(finite) else np.nan,
    }


def score_threshold_for_fraction(values: np.ndarray, fraction: float) -> float:
    """Tie-aware threshold selecting at least a requested score fraction."""
    frame = pd.DataFrame({"score": np.asarray(values, dtype=float)}).groupby(
        "score", sort=True
    ).size().rename("n_pairs").iloc[::-1].reset_index()
    frame["selected_fraction"] = frame.n_pairs.cumsum() / len(values)
    return float(frame.loc[frame.selected_fraction >= fraction, "score"].iloc[0])


def resolve_relative(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()
