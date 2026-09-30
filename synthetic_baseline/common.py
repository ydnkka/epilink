"""Shared definitions for the synthetic baseline assessment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_exploration.common import save_table, write_json, log


@dataclass(frozen=True)
class Endpoint:
    """Near-transmission endpoint for pairwise ranking."""
    key: str
    label: str
    horizon: int


ENDPOINTS = (
    Endpoint("M0", "M == 0", 0),
    Endpoint("Mle1", "M <= 1", 1),
    Endpoint("Mle2", "M <= 2", 2),
)


@dataclass(frozen=True)
class RelationshipCategory:
    """Relationship category for cluster composition."""
    key: str
    label: str
    ad: int | None
    ca: int | None
    m1: int | None
    m2: int | None
    m_sum: int | None


RELATIONSHIP_CATEGORIES = (
    RelationshipCategory("AD0", "AD(0): direct transmission", 0, 0, None, None, 0),
    RelationshipCategory("AD1", "AD(1): one intermediate", 1, 0, None, None, 1),
    RelationshipCategory("AD2", "AD(2): two intermediates", 2, 0, None, None, 2),
    RelationshipCategory("ADge3", "AD(>=3): three or more intermediates", None, 0, None, None, None),
    RelationshipCategory("CA00", "CA(0,0): shared infector", 0, 1, 0, 0, 0),
    RelationshipCategory("CA01", "CA(0,1)==CA(1,0): co-primary", 0, 1, None, None, 1),
    RelationshipCategory("CA11", "CA(1,1): one intermediate each", 0, 1, 1, 1, 2),
    RelationshipCategory("CAge3", "CA(m1+m2>=3): distant shared ancestry", 0, 1, None, None, None),
)


def endpoint_mask(pairs: pd.DataFrame, endpoint: Endpoint) -> np.ndarray:
    """Return the positive-label mask for a near-transmission endpoint."""
    m = pairs.M.to_numpy(dtype=float, na_value=np.nan)
    return np.isfinite(m) & (m <= endpoint.horizon)


def contamination_mask(pairs: pd.DataFrame) -> np.ndarray:
    """Pairs with three or more total intermediates are negative contamination."""
    m = pairs.M.to_numpy(dtype=float, na_value=np.nan)
    return np.isfinite(m) & (m >= 3)


def relationship_category(pairs: pd.DataFrame) -> pd.Series:
    """Assign each pair to one of the 8 relationship categories."""
    ad = pairs.AD.to_numpy(dtype=float, na_value=np.nan)
    ca = pairs.CA.to_numpy(dtype=float, na_value=np.nan)
    m = pairs.M.to_numpy(dtype=float, na_value=np.nan)
    m1 = pairs.m1.to_numpy(dtype=float, na_value=np.nan)
    m2 = pairs.m2.to_numpy(dtype=float, na_value=np.nan)
    
    conditions = [
        (ad == 1) & (m == 0),
        (ad == 1) & (m == 1),
        (ad == 1) & (m == 2),
        (ad == 1) & (m >= 3),
        (ca == 1) & (m1 == 0) & (m2 == 0),
        (ca == 1) & (m1 + m2 == 1),
        (ca == 1) & (m1 == 1) & (m2 == 1),
        (ca == 1) & (m1 + m2 >= 3),
    ]
    choices = ["AD0", "AD1", "AD2", "ADge3", "CA00", "CA01", "CA11", "CAge3"]
    
    result = pd.Series(np.select(conditions, choices, default="unknown"), index=pairs.index)
    return result


def m_category_counts(pairs: pd.DataFrame, mask: np.ndarray | None = None) -> dict[str, int]:
    """Count selected pairs in the M categories."""
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


def relationship_category_counts(pairs: pd.DataFrame, mask: np.ndarray | None = None) -> dict[str, int]:
    """Count selected pairs in the 8 relationship categories."""
    if mask is None:
        mask = np.ones(len(pairs), dtype=bool)
    categories = relationship_category(pairs)
    selected_categories = categories[mask]
    counts = selected_categories.value_counts().to_dict()
    result = {}
    for cat in RELATIONSHIP_CATEGORIES:
        result[cat.key] = counts.get(cat.key, 0)
    result["unknown"] = counts.get("unknown", 0)
    return result


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
