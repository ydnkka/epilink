"""Stage 1: Observable ambiguity and relationship prevalence."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import log, save_table, relationship_category, RELATIONSHIP_CATEGORIES


def relationship_prevalence(pairs: pd.DataFrame) -> pd.DataFrame:
    """Count pairs by the 8 relationship categories."""
    cats = relationship_category(pairs)
    counts = cats.value_counts().reindex([c.key for c in RELATIONSHIP_CATEGORIES], fill_value=0)
    
    rows = []
    for cat in RELATIONSHIP_CATEGORIES:
        count = counts.get(cat.key, 0)
        rows.append({
            "category": cat.key,
            "label": cat.label,
            "n_pairs": int(count),
            "fraction": count / len(pairs),
        })
    return pd.DataFrame(rows)


def m_prevalence(pairs: pd.DataFrame) -> pd.DataFrame:
    """Count pairs by M value (total intermediates)."""
    m = pairs.M.dropna()
    counts = m.value_counts().sort_index()
    
    rows = []
    for m_val in range(0, 20):
        count = int(counts.get(m_val, 0))
        rows.append({"M": m_val, "n_pairs": count, "fraction": count / len(pairs)})
    
    # Group M >= 20
    m_ge_20 = m[m >= 20]
    if len(m_ge_20):
        rows.append({"M": 20, "n_pairs": int(len(m_ge_20)), "fraction": len(m_ge_20) / len(pairs)})
    
    return pd.DataFrame(rows)


def feature_cells(pairs: pd.DataFrame, genetic_column: str) -> pd.DataFrame:
    """Aggregate pairs into identical (time_days, genetic_distance) cells."""
    frame = pd.DataFrame({
        "time_days": np.rint(pairs.SamplingDateDistanceDays.to_numpy()).astype(int),
        "genetic_distance": np.rint(pairs[genetic_column].to_numpy()).astype(int),
        "category": relationship_category(pairs),
        "is_target": pairs.IsRelated.to_numpy(bool),
    })
    
    cells = frame.groupby(["time_days", "genetic_distance", "category"], sort=True).agg(
        n_pairs=("is_target", "size"),
        n_target=("is_target", "sum"),
    ).reset_index()
    
    cells["n_other"] = cells.n_pairs - cells.n_target
    cells["target_fraction"] = cells.n_target / cells.n_pairs
    return cells


def ambiguity_summary(pairs: pd.DataFrame, genetic_column: str) -> dict:
    """Quantify ambiguity: cells containing both target and non-target pairs."""
    cells = feature_cells(pairs, genetic_column)
    
    # Group by (time, genetic) to check for mixed cells
    cell_targets = cells.groupby(["time_days", "genetic_distance"]).agg(
        n_pairs=("n_pairs", "sum"),
        n_target=("n_target", "sum"),
        n_other=("n_other", "sum"),
    ).reset_index()
    
    mixed = cell_targets.loc[(cell_targets.n_target > 0) & (cell_targets.n_other > 0)]
    
    total_target_in_mixed = mixed.n_target.sum()
    total_pairs_in_mixed = mixed.n_pairs.sum()
    
    return {
        "n_cells": len(cell_targets),
        "mixed_cells": len(mixed),
        "mixed_cell_fraction": len(mixed) / len(cell_targets) if len(cell_targets) else np.nan,
        "target_fraction_in_mixed_cells": float(total_target_in_mixed / total_pairs_in_mixed) if total_pairs_in_mixed else np.nan,
        "target_pairs_in_mixed": int(total_target_in_mixed),
        "total_pairs_in_mixed": int(total_pairs_in_mixed),
    }


def investigate_observations(pairs: pd.DataFrame, directory, settings) -> None:
    """Stage 1: Analyze relationship prevalence and observable ambiguity."""
    log("1/4: relationship prevalence and observable ambiguity")
    directory.mkdir(parents=True, exist_ok=True)
    
    # Relationship prevalence (8 categories)
    rel_prev = relationship_prevalence(pairs)
    save_table(directory, "relationship_prevalence", rel_prev)
    
    # M prevalence
    m_prev = m_prevalence(pairs)
    save_table(directory, "M_prevalence", m_prev)
    
    # Feature cells for both data processes
    for process, genetic_column in [("deterministic", "DeterministicDistance"), 
                                     ("stochastic", "StochasticDistance")]:
        cells = feature_cells(pairs, genetic_column)
        save_table(directory, f"{process}_feature_cells", cells)
        
        ambig = ambiguity_summary(pairs, genetic_column)
        ambig["data_process"] = process
        save_table(directory, f"{process}_ambiguity_summary", [ambig])
    
    # Combined ambiguity summary
    ambig_rows = []
    for process, genetic_column in [("deterministic", "DeterministicDistance"),
                                     ("stochastic", "StochasticDistance")]:
        ambig = ambiguity_summary(pairs, genetic_column)
        ambig_rows.append({"data_process": process, **ambig})
    save_table(directory, "ambiguity_summary", ambig_rows)
