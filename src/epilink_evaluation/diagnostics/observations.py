"""Exact observed-feature cells, with the shared pairwise truth definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics.pairwise import PairTruth
from ..schemas import DISTANCES, ENDPOINTS


def _aligned_truth(observations, truth):
    if "pair_id" not in observations or "pair_id" not in truth:
        raise ValueError("Observations and truth must contain pair IDs")
    if not observations.pair_id.equals(truth.pair_id):
        raise ValueError("Observation and truth pair IDs are not exactly aligned")
    if observations.pair_id.isna().any() or observations.pair_id.duplicated().any():
        raise ValueError("Pair IDs must be nonmissing and unique")
    return PairTruth(truth)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def _distance_values(series):
    # Preserve the saved numeric dtype, including nullable integer columns.
    if series.isna().any():
        raise ValueError(f"Invalid observed distance: {series.name}")
    values = series.to_numpy(dtype=getattr(series.dtype, "numpy_dtype", None))
    if values.dtype.kind not in "iuf" or not np.all(
        np.isfinite(values) & (values >= 0)
    ):
        raise ValueError(f"Invalid observed distance: {series.name}")
    return values


def observation_diagnostics(
    observations: pd.DataFrame, truth: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(cells, summary, prevalence, relationships)`` for aligned pairs.

    Cells use exact saved GD or (GD, TD) values, without binning or rounding.
    Both ``DISTANCES`` processes and all ``ENDPOINTS`` are reported in long
    format. ``TD`` is NaN for GD-only cells. Each cell includes all exhaustive
    ``PairTruth`` relationship counts (``n_AD0``, ..., ``n_separate``).

    Summary denominators are deliberately distinct:

    * ``mixed_cell_fraction``: mixed cells / all occupied cells.
    * ``pair_fraction_in_mixed_cells``: pairs in mixed cells / all pairs.
    * ``target_fraction_in_mixed_cells``: targets in mixed cells / all targets.
    * ``target_prevalence_in_mixed_cells``: targets / pairs within mixed cells.
    * ``non_target_fraction_in_mixed_cells``: non-targets in mixed cells / all non-targets.
    * ``class_conditional_overlap``: sum over cells of min(n_target_cell / all_targets, n_other_cell / all_non_targets).

    ``minimum_feature_only_misclassifications`` sums min(n_target, n_other)
    over cells; its rate divides by all pairs. This is an empirical minimum
    for target/non-target decisions constant within these exact feature cells, not a
    universal performance ceiling or a bound on partition metrics.
    Undefined ratios, including class-conditional overlap with an absent
    class, are NaN.

    ``prevalence`` has endpoint, n_pairs, n_target, n_other, target_prevalence;
    ``relationships`` has relationship, n_pairs, pair_fraction. These two
    tables depend only on truth and have no process/feature-set replication.

    Pair IDs (including row order/index) must match exactly and be unique and
    nonmissing. Grouping occurs once per process/feature set; NumPy histograms
    reuse those groups for relationship and endpoint counts, without joins or
    endpoint-specific pair-level DataFrames.
    """
    pair_truth = _aligned_truth(observations, truth)
    td = _distance_values(observations["TD"])
    prevalence = pd.DataFrame(
        [
            {
                "endpoint": endpoint,
                "n_pairs": pair_truth.n,
                "n_target": total,
                "n_other": pair_truth.n - total,
                "target_prevalence": _ratio(total, pair_truth.n),
            }
            for endpoint, total in pair_truth.totals.items()
        ]
    )
    relationships = pd.DataFrame(
        [
            {
                "relationship": category,
                "n_pairs": total,
                "pair_fraction": _ratio(total, pair_truth.n),
            }
            for category, total in pair_truth.category_totals.items()
        ]
    )
    cells, summaries = [], []
    for process, column in DISTANCES.items():
        gd = _distance_values(observations[column])
        for feature_set in ("GD", "GD_TD"):
            if feature_set == "GD":
                cell_gd, groups = np.unique(gd, return_inverse=True)
                cell_td = np.full(len(cell_gd), np.nan)
            else:
                # Structured keys avoid promoting integer GD to float when TD
                # is floating point (which can merge distinct large integers).
                keys = np.rec.fromarrays(
                    [gd, td], dtype=np.dtype([("GD", gd.dtype), ("TD", td.dtype)])
                )
                unique, groups = np.unique(keys, return_inverse=True)
                cell_gd, cell_td = unique["GD"], unique["TD"]
                del keys, unique
            n_cells = len(cell_gd)
            sizes = np.bincount(groups, minlength=n_cells)
            category_counts = {
                f"n_{category}": np.bincount(groups[mask], minlength=n_cells)
                for category, mask in pair_truth.categories.items()
            }
            for endpoint in ENDPOINTS:
                target = np.bincount(
                    groups[pair_truth.masks[endpoint]], minlength=n_cells
                )
                other = sizes - target
                mixed = (target > 0) & (other > 0)
                cells.append(
                    pd.DataFrame(
                        {
                            "process": process,
                            "feature_set": feature_set,
                            "endpoint": endpoint,
                            "GD": cell_gd,
                            "TD": cell_td,
                            "n_pairs": sizes,
                            "n_target": target,
                            "n_other": other,
                            "target_fraction": np.divide(
                                target,
                                sizes,
                                out=np.full(n_cells, np.nan),
                                where=sizes > 0,
                            ),
                            "mixed": mixed,
                            **category_counts,
                        }
                    )
                )
                total_target = pair_truth.totals[endpoint]
                total_other = pair_truth.n - total_target
                mixed_cells = int(mixed.sum())
                mixed_pairs = int(sizes[mixed].sum())
                mixed_target = int(target[mixed].sum())
                mixed_other = mixed_pairs - mixed_target
                minimum_errors = int(np.minimum(target, other).sum())
                summaries.append(
                    {
                        "process": process,
                        "feature_set": feature_set,
                        "endpoint": endpoint,
                        "n_pairs": pair_truth.n,
                        "n_target": total_target,
                        "n_other": total_other,
                        "n_cells": n_cells,
                        "mixed_cells": mixed_cells,
                        "n_pairs_in_mixed_cells": mixed_pairs,
                        "n_target_in_mixed_cells": mixed_target,
                        "n_other_in_mixed_cells": mixed_other,
                        "mixed_cell_fraction": _ratio(mixed_cells, n_cells),
                        "pair_fraction_in_mixed_cells": _ratio(
                            mixed_pairs, pair_truth.n
                        ),
                        "target_fraction_in_mixed_cells": _ratio(
                            mixed_target, total_target
                        ),
                        "target_prevalence_in_mixed_cells": _ratio(
                            mixed_target, mixed_pairs
                        ),
                        "non_target_fraction_in_mixed_cells": _ratio(
                            mixed_other, total_other
                        ),
                        "minimum_feature_only_misclassifications": minimum_errors,
                        "minimum_feature_only_misclassification_rate": _ratio(
                            minimum_errors, pair_truth.n
                        ),
                        "class_conditional_overlap": float(
                            np.minimum(target / total_target, other / total_other).sum()
                        )
                        if total_target and total_other
                        else np.nan,
                    }
                )
    # Long-format concatenation can otherwise round large integer coordinates
    # when another process has floating GD, or GD-only cells supply NaN TD.
    # Use object only where the common numeric dtype cannot preserve integers.
    for column in ("GD", "TD"):
        common_dtype = np.result_type(*(frame[column].dtype for frame in cells))
        if common_dtype.kind == "f" and any(
            frame[column].dtype.kind in "iu"
            and not frame.empty
            and int(frame[column].max()) > 2 ** (np.finfo(common_dtype).nmant + 1)
            for frame in cells
        ):
            for frame in cells:
                frame[column] = frame[column].astype(object)
    return (
        pd.concat(cells, ignore_index=True),
        pd.DataFrame(summaries),
        prevalence,
        relationships,
    )
