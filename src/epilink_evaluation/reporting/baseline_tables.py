"""Endpoint-aware summaries of frozen operating points."""

import pandas as pd

from ..schemas import ENDPOINTS


def objective_endpoint(objective):
    prefix = objective.split("_", 1)[0]
    return prefix if prefix in ENDPOINTS else None


def operating_summary(evaluation, frozen):
    """Keep all endpoints while labelling the objective actually replayed."""
    metrics = [
        f"{ep}_{metric}"
        for ep in ENDPOINTS
        for metric in ("precision", "recall", "f1", "enrichment")
    ]
    metrics += [
        "Mge3_contamination",
        "separate_fraction",
        "direct_edge_retention",
        "shared_infector_retention",
        "selected_pairs",
        "selected_fraction",
        "singleton_fraction",
        "largest_cluster_fraction",
        "n_clusters",
    ]
    rules = {
        (point["criterion"], point["pipeline"]): point
        for point in frozen["operating_points"]
        if point["status"] == "selected"
    }
    rows = []
    for (criterion, pipeline), group in evaluation.groupby(
        ["criterion", "pipeline"], sort=True
    ):
        point = rules[criterion, pipeline]
        if (
            set(group.setting_id) != {point["setting_id"]}
            or group.seed.duplicated().any()
        ):
            raise ValueError(
                "Operating results differ from the frozen criterion/setting mapping"
            )
        objective = point["rule"]["objective"]
        row = {
            "criterion": criterion,
            "pipeline": pipeline,
            "setting_id": point["setting_id"],
            "objective": objective,
            "objective_endpoint": objective_endpoint(objective),
            "n_realizations": group.seed.nunique(),
        }
        for metric in dict.fromkeys([*metrics, objective]):
            if metric not in group:
                continue
            stats = group[metric].agg(["mean", "std", "min", "max", "count"])
            row.update({f"{metric}_{stat}": value for stat, value in stats.items()})
        for stat in ("mean", "std", "min", "max", "count"):
            row[f"objective_{stat}"] = row[f"{objective}_{stat}"]
        rows.append(row)
    return pd.DataFrame(rows)
