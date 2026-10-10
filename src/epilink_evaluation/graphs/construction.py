import igraph as ig
import numpy as np


def selected_pairs(values, spec, threshold, empty=False):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite pair score")
    if empty:
        return np.zeros(len(values), dtype=bool)
    return values >= threshold if spec.higher_is_better else values <= threshold


def build_graph(
    observations,
    n_cases,
    values,
    spec,
    threshold,
    empty=False,
    *,
    full=False,
):
    values = np.asarray(values, dtype=float)
    if full:
        if threshold is not None or empty:
            raise ValueError("Full graphs have no cutoff or empty-selection setting")
        if not np.all(np.isfinite(values)):
            raise ValueError("Non-finite pair score")
        keep = np.ones(len(values), dtype=bool)
    else:
        keep = selected_pairs(values, spec, threshold, empty)
    if spec.family not in ("epilink", "logistic", "genetic"):
        raise ValueError(f"Unknown graph scorer family: {spec.family}")
    if np.any(np.asarray(values) < 0):
        raise ValueError("Weights cannot be negative")
    graph = ig.Graph(
        n=n_cases,
        edges=np.column_stack(
            (observations.a.to_numpy()[keep], observations.b.to_numpy()[keep])
        ),
    )
    graph.es["weight"] = (
        np.ones(int(keep.sum())) if spec.family == "genetic" else values[keep]
    ).tolist()
    return graph
