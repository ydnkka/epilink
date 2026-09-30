"""Shared score metadata and strict pair/partition alignment contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

ENDPOINTS = {"M0": 0, "Mle1": 1, "Mle2": 2}
DISTANCES = {"deterministic": "GD_deterministic", "stochastic": "GD_stochastic"}


@dataclass(frozen=True)
class ScoreSpec:
    name: str
    family: str
    data_process: str
    inference_process: str = "none"
    target: str = "M0"
    higher_is_better: bool = True

    def metadata(self):
        return asdict(self)


def validate_pairs(observations, cases):
    ids = cases.case_id.astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("Case identifiers must be unique")
    if len(observations) != len(ids) * (len(ids) - 1) // 2:
        raise ValueError(
            "Evaluation requires the complete sampled unordered pair universe"
        )
    if observations.pair_id.duplicated().any():
        raise ValueError("Duplicate pair IDs")
    a, b = np.triu_indices(len(ids), k=1)
    if not (np.array_equal(observations.a, a) and np.array_equal(observations.b, b)):
        raise ValueError("Pairs must use canonical sampled-case order")
    for column in ("TD", *DISTANCES.values()):
        values = observations[column].to_numpy(float)
        if not np.all(np.isfinite(values) & (values >= 0)):
            raise ValueError(f"Invalid observed distance: {column}")


def align_scores(observations, scores):
    if not scores.pair_id.equals(observations.pair_id):
        raise ValueError("Score rows do not match the observation pair IDs")
    return scores


def validate_partition(cases, memberships):
    if memberships.case_id.duplicated().any():
        raise ValueError("Duplicate membership IDs")
    if set(memberships.case_id.astype(str)) != set(cases.case_id.astype(str)):
        raise ValueError("Partition and observed sample universes differ")
    ordered = (
        memberships.assign(case_id=memberships.case_id.astype(str))
        .set_index("case_id")
        .loc[cases.case_id.astype(str), "cluster_id"]
    )
    if ordered.isna().any():
        raise ValueError("Missing cluster assignments")
    return pd.factorize(ordered, sort=False)[0]
