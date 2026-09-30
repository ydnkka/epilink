from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from ..schemas import ENDPOINTS

CATEGORIES = (
    "AD0",
    "AD1",
    "AD2",
    "ADge3",
    "CA00",
    "CA01",
    "CA02",
    "CA11",
    "CAge3",
    "separate",
)


class PairTruth:
    def __init__(self, truth):
        self.m = truth.M.to_numpy(float, na_value=np.nan)
        ad, ca = truth.AD.to_numpy(bool), truth.CA.to_numpy(bool)
        m1 = truth.m1.to_numpy(float, na_value=np.nan)
        m2 = truth.m2.to_numpy(float, na_value=np.nan)
        if np.any(~np.isfinite(self.m) & (ad | ca)):
            raise ValueError("Connected pairs must have a defined M")
        self.masks = {
            name: np.isfinite(self.m) & (self.m <= horizon)
            for name, horizon in ENDPOINTS.items()
        }
        self.categories = {
            "AD0": ad & (self.m == 0),
            "AD1": ad & (self.m == 1),
            "AD2": ad & (self.m == 2),
            "ADge3": ad & (self.m >= 3),
            "CA00": ca & (self.m == 0),
            "CA01": ca & (self.m == 1),
            "CA02": ca & (self.m == 2) & ((m1 == 0) | (m2 == 0)),
            "CA11": ca & (m1 == 1) & (m2 == 1),
            "CAge3": ca & (self.m >= 3),
            "separate": ~(ad | ca),
        }
        if not np.all(
            sum(mask.astype(np.int8) for mask in self.categories.values()) == 1
        ):
            raise ValueError(
                "Relationship categories must exhaustively partition the pair universe"
            )
        self.totals = {name: int(mask.sum()) for name, mask in self.masks.items()}
        self.category_totals = {
            name: int(mask.sum()) for name, mask in self.categories.items()
        }
        self.n = len(truth)

    def statistics(self, selected):
        counts = {
            name: int(np.count_nonzero(selected & mask))
            for name, mask in self.categories.items()
        }
        return count_statistics(counts, self)


def count_statistics(counts, truth):
    n = sum(counts.values())
    positive = {
        "M0": counts["AD0"] + counts["CA00"],
        "Mle1": counts["AD0"] + counts["CA00"] + counts["AD1"] + counts["CA01"],
    }
    positive["Mle2"] = (
        positive["Mle1"] + counts["AD2"] + counts["CA02"] + counts["CA11"]
    )
    result = {
        "selected_pairs": n,
        "selected_fraction": n / truth.n if truth.n else np.nan,
        "Mge3_contamination": (counts["ADge3"] + counts["CAge3"]) / n if n else np.nan,
        "separate_fraction": counts["separate"] / n if n else np.nan,
    }
    for endpoint, tp in positive.items():
        total = truth.totals[endpoint]
        result.update(
            {
                f"{endpoint}_precision": tp / n if n else np.nan,
                f"{endpoint}_recall": tp / total if total else np.nan,
                f"{endpoint}_f1": 2 * tp / (n + total) if n + total else np.nan,
                f"{endpoint}_enrichment": (tp / n) / (total / truth.n)
                if n and total
                else np.nan,
            }
        )
    for name, count in counts.items():
        result[f"n_{name}"] = count
        result[f"fraction_{name}"] = count / n if n else np.nan
    result["direct_edge_retention"] = (
        counts["AD0"] / truth.category_totals["AD0"]
        if truth.category_totals["AD0"]
        else np.nan
    )
    result["shared_infector_retention"] = (
        counts["CA00"] / truth.category_totals["CA00"]
        if truth.category_totals["CA00"]
        else np.nan
    )
    return result


def precision_recall_curve(values, spec, truth):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or len(values) != truth.n:
        raise ValueError("Invalid or unaligned scores")
    oriented = values if spec.higher_is_better else -values
    unique, groups = np.unique(oriented, return_inverse=True)
    counts = {
        name: np.bincount(groups, weights=mask.astype(int), minlength=len(unique))[::-1]
        .cumsum()
        .astype(np.int64)
        for name, mask in truth.categories.items()
    }
    rows = [
        count_statistics({name: counts[name][i] for name in counts}, truth)
        for i in range(len(unique))
    ]
    frame = pd.DataFrame(rows)
    frame.insert(
        0, "threshold", unique[::-1] if spec.higher_is_better else -unique[::-1]
    )
    frame["ties_at_threshold"] = np.bincount(groups, minlength=len(unique))[::-1]
    summary = {
        "score_name": spec.name,
        "score_family": spec.family,
        "data_process": spec.data_process,
        "n_pairs": truth.n,
        "unique_scores": len(unique),
    }
    for endpoint in ENDPOINTS:
        recall = frame[f"{endpoint}_recall"].to_numpy()
        precision = frame[f"{endpoint}_precision"].to_numpy()
        summary[f"{endpoint}_AP"] = float(
            np.sum(np.diff(np.r_[0.0, recall]) * precision)
        )
        summary[f"{endpoint}_prevalence"] = truth.totals[endpoint] / truth.n
    return frame, summary


def calibration(values, truth, bins=10):
    values = np.asarray(values)
    y = truth.masks["M0"]
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Calibration requires probabilities in [0,1]")
    indices = np.minimum((values * bins).astype(int), bins - 1)
    rows = []
    for index in range(bins):
        selected = indices == index
        rows.append(
            {
                "bin_lower": index / bins,
                "bin_upper": (index + 1) / bins,
                "n_pairs": int(selected.sum()),
                "mean_probability": float(values[selected].mean())
                if selected.any()
                else np.nan,
                "observed_fraction": float(y[selected].mean())
                if selected.any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows), {
        "brier_score": brier_score_loss(y, values),
        "log_loss": log_loss(y, values, labels=[False, True]),
    }
