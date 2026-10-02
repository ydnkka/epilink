from __future__ import annotations

import numpy as np
import pandas as pd

from ..schemas import ENDPOINTS


def aggregate_settings(frame):
    numeric = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column != "seed"
    ]
    grouped = frame.groupby(["pipeline", "setting_id"], sort=True)
    result = grouped[numeric].agg(["mean", "std", "min", "max"])
    result.columns = [f"{metric}_{statistic}" for metric, statistic in result.columns]
    result["n_realizations"] = grouped.size()
    return result.reset_index()


def select_operating_points(frame, definitions, criteria, development_seeds):
    if frame.empty:
        return []
    if not frame.split.eq("development").all():
        raise ValueError("Operating-point selection accepts development evidence only")
    if set(frame.seed) != set(development_seeds):
        raise ValueError("Development seed coverage differs from the configured split")
    if frame.duplicated(["pipeline", "setting_id", "seed"]).any():
        raise ValueError("Duplicate development evidence")
    records = []
    for pipeline, group in frame.groupby("pipeline", sort=True):
        for criterion in criteria:
            objective = criterion["objective"]
            if objective not in group:
                raise ValueError(f"Unknown selection objective: {objective}")
            candidates = []
            for setting_id, evidence in group.groupby("setting_id", sort=True):
                if set(evidence.seed) != set(development_seeds):
                    continue
                feasible = np.isfinite(evidence[objective]).all()
                for metric, bounds in criterion.get("constraints", {}).items():
                    if metric not in evidence or not set(bounds) <= {"min", "max"}:
                        raise ValueError(f"Invalid constraint: {metric} {bounds}")
                    values = evidence[metric]
                    feasible &= np.isfinite(values).all()
                    if "min" in bounds:
                        feasible &= (values >= bounds["min"]).all()
                    if "max" in bounds:
                        feasible &= (values <= bounds["max"]).all()
                if feasible:
                    candidates.append(
                        (
                            float(evidence[objective].mean()),
                            float(evidence[objective].std(ddof=0)),
                            str(setting_id),
                        )
                    )
            base = {
                "pipeline": pipeline,
                "criterion": criterion["name"],
                "rule": criterion,
            }
            if candidates:
                mean, sd, setting_id = sorted(
                    candidates, key=lambda row: (-row[0], row[1], row[2])
                )[0]
                records.append(
                    {
                        **base,
                        "status": "selected",
                        "setting_id": setting_id,
                        "development_objective_mean": mean,
                        "development_objective_sd": sd,
                        "definition": definitions[setting_id],
                    }
                )
            else:
                records.append({**base, "status": "infeasible", "setting_id": None})
    return records


def pareto_frontier(frame, precision="M0_precision", recall="M0_recall"):
    """Keep non-dominated precision/recall settings (including exact ties)."""
    rows = []
    for pipeline, group in frame.groupby("pipeline", sort=True):
        group = group.dropna(subset=[precision, recall]).sort_values(
            [precision, recall], ascending=False
        )
        best_recall = -np.inf
        for _, equal_precision in group.groupby(precision, sort=False):
            local_max = equal_precision[recall].max()
            if local_max > best_recall:
                rows.append(equal_precision.loc[equal_precision[recall] == local_max])
            best_recall = max(best_recall, local_max)
    return pd.concat(rows, ignore_index=True) if rows else frame.iloc[:0].copy()


def endpoint_frontiers(summary):
    """Long-form endpoint-labelled frontiers; retain explicit mean column names."""
    return pd.concat([
        pareto_frontier(summary, f"{ep}_precision_mean", f"{ep}_recall_mean").copy().assign(endpoint=ep)
        for ep in ENDPOINTS
    ], ignore_index=True)
