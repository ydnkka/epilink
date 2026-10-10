"""Deterministic bounded coarse-to-fine searches on development evidence."""

from collections.abc import Mapping
from itertools import pairwise

import numpy as np


def _settings(value, *, name, integer=False, allow_zero=False, scale="log"):
    if not isinstance(value, Mapping):
        values = list(value)
        if not values or any(
            not np.isfinite(x)
            or x < 0
            or (x == 0 and not allow_zero)
            or (integer and (isinstance(x, bool) or int(x) != x))
            for x in values
        ):
            raise ValueError(f"Invalid {name} values")
        cast = int if integer else float
        return {"values": sorted(set(map(cast, values))), "budget": len(set(values))}
    result = {
        "initial_points": 8,
        "budget": 28,
        "scale": scale,
        "tolerance": 1e-3,
        **value,
    }
    if set(result) != {"min", "max", "initial_points", "budget", "scale", "tolerance"}:
        raise ValueError(f"{name} search requires min/max and known search settings")
    if not (
        np.isfinite(result["min"])
        and np.isfinite(result["max"])
        and 0 <= result["min"] < result["max"]
        and (allow_zero or result["min"] > 0)
    ):
        raise ValueError(f"Invalid {name} bounds")
    if integer and any(
        isinstance(result[k], bool) or int(result[k]) != result[k]
        for k in ("min", "max")
    ):
        raise ValueError(f"{name} bounds must be integers")
    if (
        any(type(result[k]) is not int for k in ("initial_points", "budget"))
        or not 2 <= result["initial_points"] <= result["budget"]
    ):
        raise ValueError(f"{name} budget must cover at least two initial points")
    if (
        result["scale"] not in {"log", "linear"}
        or (result["scale"] == "log" and result["min"] == 0)
        or not np.isfinite(result["tolerance"])
        or result["tolerance"] <= 0
    ):
        raise ValueError(f"Invalid {name} search scale or tolerance")
    return result


def resolution_settings(value):
    """Resolve positive resolution bounds/budget or a fixed comparison list."""
    return _settings(value, name="Resolution")


def cutoff_settings(value, *, integer=False):
    """Resolve nonnegative SNP/day cutoffs; SNPs have an integer domain."""
    return _settings(
        value,
        name="TreeCluster cutoff",
        integer=integer,
        allow_zero=True,
        scale="linear",
    )


def initial_resolutions(value):
    return _initial_values(resolution_settings(value))


def initial_cutoffs(value, *, integer=False):
    return _initial_values(cutoff_settings(value, integer=integer), integer=integer)


def _initial_values(settings, *, integer=False):
    if "values" in settings:
        return settings["values"]
    space = np.geomspace if settings["scale"] == "log" else np.linspace
    count = settings["initial_points"]
    if integer:
        count = min(count, int(settings["max"] - settings["min"]) + 1)
    values = space(settings["min"], settings["max"], count).tolist()
    values[0], values[-1] = float(settings["min"]), float(settings["max"])
    return sorted(set(map(int, np.rint(values)))) if integer else values


def next_resolutions(value, frame, definitions, criteria, seeds):
    """Refine promising intervals for each pipeline/criterion, reserving exploration.

    All pipelines use the same trials; every trial must cover every development
    seed. Restart selection remains the clusterer's own algorithm objective.
    """
    definitions = {k: d for k, d in definitions.items() if d["kind"] == "leiden"}
    return _next_values(
        resolution_settings(value), frame, definitions, criteria, seeds, "resolution"
    )


def next_cutoffs(value, frame, definitions, criteria, seeds, *, integer=False):
    """Refine a common TreeCluster cutoff pool across processes and methods."""
    definitions = {
        k: d
        for k, d in definitions.items()
        if d["kind"] == "treecluster"
        and d["tree_kind"] == ("raw" if integer else "dated")
    }
    return _next_values(
        cutoff_settings(value, integer=integer),
        frame,
        definitions,
        criteria,
        seeds,
        "threshold_input",
        integer=integer,
        groups=("pipeline", "method"),
    )


def _next_values(
    settings,
    frame,
    definitions,
    criteria,
    seeds,
    parameter,
    *,
    integer=False,
    groups=("pipeline",),
):
    if not frame.empty and not frame.split.eq("development").all():
        raise ValueError("Adaptive search accepts development evidence only")
    if "values" in settings:
        return []
    frame = frame.loc[frame.setting_id.isin(definitions)]
    tested = sorted({d[parameter] for d in definitions.values()})
    budget = settings["budget"]
    if integer:
        budget = min(budget, int(settings["max"] - settings["min"]) + 1)
    remaining = budget - len(tested)
    if remaining <= 0:
        return []
    transform = np.log if settings["scale"] == "log" else float
    inverse = np.exp if settings["scale"] == "log" else float
    intervals = [
        (a, b, transform(b) - transform(a))
        for a, b in pairwise(tested)
        if transform(b) - transform(a) > settings["tolerance"]
        and (not integer or b - a > 1)
    ]
    if not intervals:
        return []
    rankings = []
    for target in sorted(
        {tuple(d[field] for field in groups) for d in definitions.values()}
    ):
        for criterion in criteria:
            scores = {}
            for key, definition in definitions.items():
                if tuple(definition[field] for field in groups) != target:
                    continue
                group = frame.loc[frame.setting_id.eq(key)]
                if len(group) != len(seeds) or set(group.seed) != set(seeds):
                    raise ValueError("Incomplete adaptive search development evidence")
                objective = group[criterion["objective"]]
                feasible = np.isfinite(objective).all()
                for metric, bounds in criterion.get("constraints", {}).items():
                    metric_values = group[metric]
                    feasible &= np.isfinite(metric_values).all()
                    if "min" in bounds:
                        feasible &= (metric_values >= bounds["min"]).all()
                    if "max" in bounds:
                        feasible &= (metric_values <= bounds["max"]).all()
                scores[definition[parameter]] = (
                    float(objective.mean()) if feasible else -np.inf
                )
            rankings.append(
                sorted(
                    intervals,
                    key=lambda x: (-max(scores[x[0]], scores[x[1]]), -x[2], x[0]),
                )
            )
    count = min(settings["initial_points"], remaining, len(intervals))
    # A widest-interval trial keeps exploration alive even on flat objectives.
    selected = [max(intervals, key=lambda x: (x[2], -x[0]))]
    ranks = {
        interval: [ranking.index(interval) for ranking in rankings]
        for interval in intervals
    }
    promising = sorted(
        intervals,
        key=lambda x: (
            min(ranks[x]),
            -sum(1 / (rank + 1) for rank in ranks[x]),
            -x[2],
            x[0],
        ),
    )
    selected.extend(interval for interval in promising if interval not in selected)
    selected = selected[:count]
    values = []
    for a, b, _ in selected:
        midpoint = float(inverse((transform(a) + transform(b)) / 2))
        if integer:
            midpoint = max(int(a) + 1, min(int(b) - 1, round(midpoint)))
        values.append(midpoint)
    return sorted(values)
