"""Offspring heterogeneity and structure of the fixed transmission backbone."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..truth import TreeIndex

DEFAULTS = {
    "superspreading_quantile": 0.99,
    "bootstrap_replicates": 0,
    "bootstrap_seed": 67001,
}


def backbone_settings(settings=None):
    """Resolve diagnostic-only settings without changing the shared experiment."""
    settings = settings or {}
    if set(settings) - set(DEFAULTS):
        raise ValueError("Unknown backbone diagnostic settings")
    result = {**DEFAULTS, **settings}
    q = result["superspreading_quantile"]
    if not np.isfinite(q) or not 0 < q < 1:
        raise ValueError("superspreading_quantile must be in (0, 1)")
    for field in ("bootstrap_replicates", "bootstrap_seed"):
        if type(result[field]) is not int or result[field] < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    return result


def _counts(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not x.size:
        raise ValueError("Offspring counts must be a nonempty one-dimensional array")
    if np.any(~np.isfinite(x)) or np.any(x < 0) or np.any(x != np.rint(x)):
        raise ValueError("Offspring counts must be finite nonnegative integers")
    return x.astype(np.int64)


def _fit_dispersion(x):
    """Profile the NB likelihood at its exact empirical-mean MLE."""
    mean = float(x.mean())
    if mean == 0:
        return np.nan, "degenerate", "No transmissions"
    if len(x) < 2:
        return np.nan, "insufficient_cases", "Dispersion requires at least two cases"
    variance = float(x.var())
    if variance <= mean:
        return np.nan, "poisson_limit", "No finite overdispersion estimate"
    values, frequencies = np.unique(x, return_counts=True)

    def negative_log_likelihood(log_k):
        k = np.exp(log_k)
        return -float(frequencies @ stats.nbinom.logpmf(values, k, k / (k + mean)))

    upper = np.log(1e6)
    try:
        fit = optimize.minimize_scalar(
            negative_log_likelihood,
            bounds=(np.log(1e-8), upper),
            method="bounded",
            options={"xatol": 1e-8},
        )
        if not fit.success or not np.isfinite(fit.fun):
            raise RuntimeError("Negative-binomial optimisation failed")
        poisson_nll = -float(frequencies @ stats.poisson.logpmf(values, mean))
        if fit.x >= upper - 1e-3 or fit.fun >= poisson_nll - 1e-8:
            return np.nan, "poisson_limit", "Poisson likelihood boundary"
        return float(np.exp(fit.x)), "mle", ""
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        # Retain the archived method-of-moments fallback and make it visible.
        k = mean**2 / (float(x.var(ddof=1)) - mean)
        return float(k), "moments_fallback", str(exc)


def _bootstrap(x, settings):
    columns = ["replicate", "mean_offspring", "dispersion_k", "fit_method"]
    rows = []
    rng = np.random.default_rng(settings["bootstrap_seed"])
    if len(x) >= 2:
        for replicate in range(settings["bootstrap_replicates"]):
            sample = x[rng.integers(0, len(x), len(x))]
            k, method, _ = _fit_dispersion(sample)
            rows.append((replicate, float(sample.mean()), k, method))
    frame = pd.DataFrame(rows, columns=columns)
    metadata = {
        "requested": settings["bootstrap_replicates"],
        "completed": len(frame),
        "seed": settings["bootstrap_seed"],
        "interpretation": "Exploratory IID case-resampling intervals, not independent epidemic replicates",
    }
    for column in ("mean_offspring", "dispersion_k"):
        finite = frame[column].dropna().to_numpy(float)
        metadata[f"{column}_kept"] = len(finite)
        metadata[f"{column}_interval95"] = (
            np.percentile(finite, [2.5, 97.5]).tolist()
            if len(finite)
            else [np.nan, np.nan]
        )
    return frame, metadata


def offspring_statistics(counts, settings=None):
    """Summarise all cases, including zeros, using the inclusive Poisson rule."""
    settings = backbone_settings(settings)
    x = _counts(counts)
    n, total = len(x), int(x.sum())
    mean = float(x.mean())
    k, method, notes = _fit_dispersion(x)
    cutoff = int(stats.poisson.ppf(settings["superspreading_quantile"], mean))
    # Preserve the archived zero-transmission exception: no events can occur.
    superspreaders = x >= cutoff if total else np.zeros(n, dtype=bool)
    ranked = np.sort(x)[::-1]
    cumulative = np.r_[0, np.cumsum(ranked)]
    n80 = int(np.searchsorted(cumulative, 0.8 * total)) if total else None
    top20 = int(np.ceil(0.2 * n))
    bootstrap, intervals = _bootstrap(x, settings)
    summary = {
        "n_cases": n,
        "n_transmissions": total,
        "mean_offspring": mean,
        "offspring_variance": float(x.var()),
        "offspring_sample_variance": float(x.var(ddof=1)) if n > 1 else np.nan,
        "variance_to_mean": float(x.var() / mean) if mean else np.nan,
        "max_offspring": int(x.max()),
        "n_zero_offspring": int((x == 0).sum()),
        "zero_offspring_fraction": float((x == 0).mean()),
        "dispersion_k": k,
        "fit_method": method,
        "fit_notes": notes,
        "superspreading_quantile": settings["superspreading_quantile"],
        "superspreading_reference": "backbone_mean_offspring",
        "superspreading_operator": ">=",
        "poisson_percentile": cutoff,
        "minimum_superspreading_offspring": cutoff if total else None,
        "n_superspreaders": int(superspreaders.sum()),
        "superspreader_fraction": float(superspreaders.mean()),
        "superspreader_transmissions": int(x[superspreaders].sum()),
        "superspreader_transmission_fraction": float(x[superspreaders].sum() / total)
        if total
        else np.nan,
        "n_cases_for_80_percent": n80,
        "fraction_for_80_percent": n80 / n if total else np.nan,
        "top20_n_cases": top20,
        "top20_case_fraction": top20 / n,
        "top20_transmission_fraction": float(ranked[:top20].sum() / total)
        if total
        else np.nan,
        "bootstrap": intervals,
    }
    concentration = pd.DataFrame(
        {
            "rank": np.arange(n + 1),
            "case_fraction": np.arange(n + 1) / n,
            "cumulative_transmissions": cumulative,
            "transmission_fraction": cumulative / total
            if total
            else np.full(n + 1, np.nan),
        }
    )
    return summary, superspreaders, concentration, bootstrap


def backbone_diagnostics(tree, settings=None):
    """Characterise a single-parent forest once, independently of observation seeds."""
    index = TreeIndex(tree)
    ids = [str(node) for node in index.nodes]
    if len(set(ids)) != len(ids):
        raise ValueError("Backbone case identifiers must remain unique as strings")
    x = np.array([tree.out_degree(node) for node in index.nodes], dtype=np.int64)
    summary, flags, concentration, bootstrap = offspring_statistics(x, settings)
    descendants = dict.fromkeys(index.nodes, 0)
    for node in reversed(list(nx.topological_sort(tree))):
        descendants[node] = sum(
            1 + descendants[child] for child in tree.successors(node)
        )
    nodes = pd.DataFrame(
        {
            "node_index": np.arange(len(tree)),
            "case_id": ids,
            "root_case_id": [ids[i] for i in index.root],
            "depth": index.depth,
            "offspring": x,
            "descendant_count": [descendants[node] for node in index.nodes],
            "is_superspreader": flags,
        }
    )
    components = (
        nodes.groupby("root_case_id", sort=True)
        .agg(
            n_cases=("case_id", "size"),
            n_transmissions=("offspring", "sum"),
            max_depth=("depth", "max"),
        )
        .reset_index()
    )
    generations = (
        nodes.groupby("depth", sort=True).size().rename("n_cases").reset_index()
    )
    generations["case_fraction"] = generations.n_cases / len(tree)
    siblings = sum(int(z) * (int(z) - 1) // 2 for z in x)
    pair_universe = len(tree) * (len(tree) - 1) // 2
    summary.update(
        {
            "n_roots": len(components),
            "n_components": len(components),
            "largest_component_cases": int(components.n_cases.max()),
            "largest_component_fraction": float(components.n_cases.max() / len(tree)),
            "mean_depth": float(index.depth.mean()),
            "max_depth": int(index.depth.max()),
            "n_direct_transmission_pairs": tree.number_of_edges(),
            "n_shared_infector_pairs": siblings,
            "n_M0_pairs": tree.number_of_edges() + siblings,
            "M0_prevalence": (tree.number_of_edges() + siblings) / pair_universe
            if pair_universe
            else np.nan,
            "scope": "all backbone cases, including zero offspring; no observation-seed replication",
        }
    )
    support = np.arange(max(int(x.max()), summary["poisson_percentile"]) + 1)
    empirical = np.bincount(x, minlength=len(support))
    poisson = stats.poisson.pmf(support, summary["mean_offspring"])
    nb = (
        stats.nbinom.pmf(
            support,
            summary["dispersion_k"],
            summary["dispersion_k"]
            / (summary["dispersion_k"] + summary["mean_offspring"]),
        )
        if np.isfinite(summary["dispersion_k"])
        else poisson
        if summary["fit_method"] == "poisson_limit"
        else np.full(len(support), np.nan)
    )
    offspring = pd.DataFrame(
        {
            "offspring": support,
            "n_cases": empirical,
            "case_fraction": empirical / len(tree),
            "poisson_probability": poisson,
            "negative_binomial_probability": nb,
            "qualifies_superspreading": support >= summary["poisson_percentile"]
            if summary["n_transmissions"]
            else False,
        }
    )
    return summary, {
        "nodes.parquet": nodes,
        "offspring.csv": offspring,
        "concentration.csv": concentration,
        "generations.csv": generations,
        "components.csv": components,
        "bootstrap.csv": bootstrap,
    }
