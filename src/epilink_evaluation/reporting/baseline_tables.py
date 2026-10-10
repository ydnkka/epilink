"""Endpoint-aware presentation and development-only grid diagnostics."""

import numpy as np
import pandas as pd

from ..config import reference_grid_config
from ..metrics.pairwise import metrics_at_thresholds
from ..schemas import ENDPOINTS
from ..scorers import SCORERS
from ..selection.operating import select_operating_points
from ..workflows.settings import settings_registry


def objective_endpoint(objective):
    prefix = objective.split("_", 1)[0]
    return prefix if prefix in ENDPOINTS else None


def operating_summary(evaluation, frozen):
    """Keep all endpoints while labelling the objective actually replayed."""
    metrics = [f"{ep}_{metric}" for ep in ENDPOINTS for metric in ("precision", "recall", "f1", "enrichment")]
    metrics += [
        "Mge3_contamination", "separate_fraction", "direct_edge_retention",
        "shared_infector_retention", "selected_pairs", "selected_fraction",
        "singleton_fraction", "largest_cluster_fraction", "n_clusters",
    ]
    rules = {
        (point["criterion"], point["pipeline"]): point
        for point in frozen["operating_points"] if point["status"] == "selected"
    }
    rows = []
    for (criterion, pipeline), group in evaluation.groupby(["criterion", "pipeline"], sort=True):
        point = rules[criterion, pipeline]
        if set(group.setting_id) != {point["setting_id"]} or group.seed.duplicated().any():
            raise ValueError("Operating results differ from the frozen criterion/setting mapping")
        objective = point["rule"]["objective"]
        row = {
            "criterion": criterion, "pipeline": pipeline, "setting_id": point["setting_id"],
            "objective": objective, "objective_endpoint": objective_endpoint(objective),
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


def _feasible(group, criterion, seeds):
    if set(group.seed) != set(seeds) or not np.isfinite(group[criterion["objective"]]).all():
        return False
    for metric, bounds in criterion.get("constraints", {}).items():
        values = group[metric]
        if not np.isfinite(values).all():
            return False
        if "min" in bounds and (values < bounds["min"]).any():
            return False
        if "max" in bounds and (values > bounds["max"]).any():
            return False
    return True


def _boundary(definition, axis, values):
    value = definition.get(axis)
    if value is None:
        return "empty" if definition.get("empty") else "not_applicable"
    lower = value == min(values)
    upper = value == max(values)
    if lower and upper:
        return "single_value"
    if lower:
        # A zero distance/probability cutoff is a real domain boundary; a
        # positive minimum CPM resolution is merely the edge of the search.
        return "natural_lower" if axis == "threshold" and value == 0 else "search_lower"
    if upper:
        if axis == "threshold" and definition.get("score_name", "").startswith("LOGIT") and value == 1:
            return "natural_upper"
        return "search_upper"
    return "interior"


def grid_adequacy(frame, definitions, config, curves_by_seed):
    """Compare the current and declared reference searches on identical dev data.

    A small refinement gain does not establish a global optimum. Neighbors hold
    other numerical settings and the TreeCluster method fixed. No evaluation
    rows or per-realization optimal thresholds are used.
    """
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    if not frame.split.eq("development").all():
        raise ValueError("Grid diagnostics accept development evidence only")
    seeds = config["splits"]["development"]
    if set(frame.seed) != set(seeds):
        return pd.DataFrame(), pd.DataFrame()
    criteria = config["selection"]["criteria"]
    points = select_operating_points(frame, definitions, criteria, seeds)
    audit = config.get("grid_audit", {})
    tolerance = audit.get("objective_tolerance", 0.005)
    ref_definitions = settings_registry(reference_grid_config(config))
    ref_parts = [frame.loc[frame.setting_id.isin(ref_definitions) & ~frame.pipeline.str.startswith("pairwise/")]]
    # Arbitrary reference cutoffs can fall between exact candidate cutoffs.
    # Read their metrics from the saved whole-tie cumulative curves.
    for seed, curves in curves_by_seed.items():
        for name, curve in curves.groupby("score_name"):
            choices = {key: d for key, d in ref_definitions.items() if d["kind"] == "pairwise" and d["score_name"] == name}
            empty_ids = [key for key, d in definitions.items() if d["kind"] == "pairwise" and d["score_name"] == name and d["empty"]]
            empty_rows = frame.loc[frame.seed.eq(seed) & frame.setting_id.isin(empty_ids)]
            if empty_rows.empty:
                continue
            columns = [c for c in curve if c not in {"threshold", "ties_at_threshold", "split", "seed", "score_name", "data_process"}]
            empty = empty_rows.iloc[0][columns].to_dict()
            values = metrics_at_thresholds(curve, [d["threshold"] for d in choices.values()], SCORERS[name].spec.higher_is_better, empty)
            ref_parts.append(values.assign(
                split="development", seed=seed, pipeline=f"pairwise/{name}", setting_id=list(choices),
            ))
    ref_frame = pd.concat(ref_parts, ignore_index=True)
    ref_points = select_operating_points(ref_frame, ref_definitions, criteria, seeds) if not ref_frame.empty and set(ref_frame.seed) == set(seeds) else []
    ref_points = {(p["pipeline"], p["criterion"]): p for p in ref_points}
    grouped = {key: group for key, group in frame.groupby("setting_id")}
    ref_coverage = set(zip(ref_frame.seed, ref_frame.setting_id))
    rows, neighbors = [], []
    for point in points:
        pipeline, criterion = point["pipeline"], point["rule"]
        row = {
            "pipeline": pipeline, "criterion": point["criterion"], "objective": criterion["objective"],
            "endpoint": objective_endpoint(criterion["objective"]), "status": point["status"],
            "setting_id": point["setting_id"], "objective_tolerance": tolerance,
            "n_candidates": frame.loc[frame.pipeline == pipeline, "setting_id"].nunique(),
        }
        if point["status"] != "selected":
            rows.append(row)
            continue
        definition = point["definition"]
        row.update({key: definition.get(key) for key in ("threshold", "resolution", "method", "weight_policy", "threshold_units")})
        row["objective_mean"] = point["development_objective_mean"]
        row["objective_sd"] = point["development_objective_sd"]
        expected_ref = {key for key, d in ref_definitions.items() if d["pipeline"] == pipeline}
        complete_ref = {(seed, key) for seed in seeds for key in expected_ref} <= ref_coverage
        previous = ref_points.get((pipeline, point["criterion"]), {})
        row["reference_n_candidates"] = len(expected_ref)
        row["reference_search_identical"] = expected_ref == {
            key for key, d in definitions.items() if d["pipeline"] == pipeline
        }
        row["reference_status"] = previous.get("status", "not_available") if complete_ref else "incomplete"
        if complete_ref and previous.get("status") == "selected":
            row["reference_setting_id"] = previous["setting_id"]
            row["reference_objective_mean"] = previous["development_objective_mean"]
            row["refinement_gain"] = row["objective_mean"] - row["reference_objective_mean"]
            row["refinement_assessment"] = "material_improvement" if row["refinement_gain"] > tolerance else (
                "reference_better" if row["refinement_gain"] < -tolerance else "within_tolerance"
            )
            if row["reference_search_identical"]:
                row["refinement_assessment"] = "not_refined"
        else:
            row["refinement_assessment"] = "unassessed"
        row["search_scope"] = "exhaustive_development_scores" if definition["kind"] == "pairwise" and config["pairwise"].get("threshold_mode") == "all_development_scores" else "finite_grid"
        for axis in ("threshold", "resolution"):
            if row["search_scope"] == "exhaustive_development_scores":
                row[f"{axis}_boundary"] = "not_applicable"
                continue
            local = {
                key: d for key, d in definitions.items()
                if d["pipeline"] == pipeline and not d.get("empty", False)
                and d.get(axis) is not None and key in grouped
                and all(d.get(other) == definition.get(other) for other in ("method", "threshold", "resolution") if other != axis)
            }
            values = sorted({d[axis] for d in local.values()})
            row[f"{axis}_boundary"] = _boundary(definition, axis, values) if values else "not_applicable"
            if definition.get(axis) is None or not values:
                continue
            index = values.index(definition[axis])
            for side, neighbor_index in (("lower", index - 1), ("upper", index + 1)):
                if not 0 <= neighbor_index < len(values):
                    continue
                key = next(key for key, d in local.items() if d[axis] == values[neighbor_index])
                evidence = grouped[key]
                metrics = [criterion["objective"], "Mge3_contamination", "singleton_fraction", "largest_cluster_fraction"]
                metrics += [f"{ep}_{m}" for ep in ENDPOINTS for m in ("precision", "recall")]
                neighbors.append({
                    "pipeline": pipeline, "criterion": point["criterion"], "selected_setting_id": point["setting_id"],
                    "axis": axis, "side": side, "value": values[neighbor_index], "setting_id": key,
                    "feasible": _feasible(evidence, criterion, seeds),
                    **{f"{m}_mean": evidence[m].mean() for m in dict.fromkeys(metrics) if m in evidence},
                })
        row["inspect_search_boundary"] = any(row.get(f"{axis}_boundary") in {"search_lower", "search_upper", "single_value"} for axis in ("threshold", "resolution"))
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(neighbors)
