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
    observations, n_cases, values, spec, threshold, policy="binary", empty=False,
    *, full=False,
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
    if policy == "native":
        if spec.family not in ("epilink", "logistic"):
            raise ValueError(
                "Native weights require nonnegative compatibility/probability scores"
            )
        if np.any(np.asarray(values) < 0):
            raise ValueError("Native weights cannot be negative")
        if not full:
            keep &= values > 0
    elif policy != "binary":
        raise ValueError(f"Unknown graph weight policy: {policy}")
    graph = ig.Graph(
        n=n_cases,
        edges=np.column_stack(
            (observations.a.to_numpy()[keep], observations.b.to_numpy()[keep])
        ),
    )
    graph.es["weight"] = (
        np.asarray(values)[keep] if policy == "native" else np.ones(int(keep.sum()))
    ).tolist()
    return graph
