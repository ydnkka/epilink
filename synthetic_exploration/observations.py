"""Investigation 1: relationships, observable overlap, and exact input ties."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import log, save_table
from .truth import RELATIONSHIPS, ENCODING_COLUMNS, canonical_encoding

DISTANCES = {"deterministic": "DeterministicDistance", "stochastic": "StochasticDistance"}


def feature_cells(pairs, distance_column):
    frame = pairs.groupby(
        ["SamplingDateDistanceDays", distance_column], observed=True
    ).IsRelated.agg(n_pairs="size", n_target="sum").reset_index()
    frame = frame.rename(columns={"SamplingDateDistanceDays": "time_days", distance_column: "genetic_distance"})
    frame["n_other"] = frame.n_pairs - frame.n_target
    frame["target_fraction"] = frame.n_target / frame.n_pairs
    frame["mixed"] = (frame.n_target > 0) & (frame.n_other > 0)
    return frame


def ambiguity_summary(cells):
    n, positives, negatives = cells.n_pairs.sum(), cells.n_target.sum(), cells.n_other.sum()
    mixed = cells.loc[cells.mixed]
    return {
        "n_pairs": int(n), "n_target": int(positives), "n_cells": len(cells),
        "mixed_cells": len(mixed), "pair_fraction_in_mixed_cells": mixed.n_pairs.sum() / n,
        "target_fraction_in_mixed_cells": mixed.n_target.sum() / positives if positives else np.nan,
        "non_target_fraction_in_mixed_cells": mixed.n_other.sum() / negatives if negatives else np.nan,
        "target_non_target_distribution_overlap": float(np.minimum(
            cells.n_target / positives, cells.n_other / negatives).sum())
            if positives and negatives else np.nan,
        # Empirical lower bound on classification error for this observed table only.
        "minimum_feature_only_misclassifications": int(np.minimum(cells.n_target, cells.n_other).sum()),
    }


def investigate_observations(pairs, directory, settings):
    log("1/4: observation overlap and true relationship geometry")
    counts = pairs.relationship.value_counts(sort=False).reindex(RELATIONSHIPS, fill_value=0)
    save_table(directory, "relationship_prevalence", pd.DataFrame({
        "relationship": counts.index, "n_pairs": counts.values,
        "prevalence": counts.values / len(pairs),
    }))
    geometry = pairs.groupby(
        ["relationship", "lca_steps_a", "lca_steps_b", "tree_hops", "M"], observed=True, dropna=False
    ).size().rename("n_pairs").reset_index()
    save_table(directory, "relationship_geometry", geometry)
    save_table(directory, "epilink_encoding_counts", canonical_encoding(pairs).groupby(
        ENCODING_COLUMNS, observed=True, dropna=False
    ).size().rename("n_pairs").reset_index())
    m_counts = pairs.groupby(["AD", "CA", "M"], observed=True, dropna=False).size().rename("n_pairs").reset_index()
    m_counts["prevalence"] = m_counts.n_pairs / len(pairs)
    save_table(directory, "M_prevalence", m_counts)
    summaries = []
    for process, column in DISTANCES.items():
        cells = feature_cells(pairs, column)
        save_table(directory, f"{process}_feature_cells", cells)
        summaries.append({"data_process": process, **ambiguity_summary(cells)})
        by_relation = pairs.groupby(
            ["SamplingDateDistanceDays", column, "relationship"], observed=True
        ).size().rename("n_pairs").reset_index().rename(columns={
            "SamplingDateDistanceDays": "time_days", column: "genetic_distance"})
        save_table(directory, f"{process}_joint_relationships", by_relation)
    save_table(directory, "ambiguity_summary", summaries)
