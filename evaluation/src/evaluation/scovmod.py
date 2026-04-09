from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from numpy.random import default_rng
import pandas as pd

from networkx.algorithms.tree.branchings import maximum_spanning_arborescence

try:
    from .config import (
        configure_logging,
        get_config_value,
        load_config,
        resolve_configured_output_path,
        resolve_configured_path,
    )
    from .heterogeneity import heterogeneity
    from .specs import DEFAULT_SEED
except ImportError:  # pragma: no cover - support direct script execution
    from config import (
        configure_logging,
        get_config_value,
        load_config,
        resolve_configured_output_path,
        resolve_configured_path,
    )
    from heterogeneity import heterogeneity
    from specs import DEFAULT_SEED


LOGGER = logging.getLogger(__name__)


# -----------------------------
# SCoVMod parsing and tree building
# -----------------------------

def parse_scovmod_outputs(filepath: Path) -> pd.DataFrame:
    """
    Parse a SCoVMod CSV with ragged ID lists.

    Parameters
    ----------
    filepath : Path
        CSV with columns ``TimeStep``, ``Location``, and a ragged list of IDs.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``TimeStep``, ``Location``, and ``Ids``.

    Notes
    -----
    SCoVMod exports may spread the ID list across multiple CSV columns, so a
    custom parser is used instead of ``pandas.read_csv``.
    """
    data = []

    with filepath.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        _ = next(reader, None)  # skip header

        for row in reader:
            if not row:
                continue

            time_step = int(row[0])
            location = int(row[1])

            # Join the remainder and evaluate as a Python list
            raw_list_str = ",".join(row[2:])
            ids = list(set(ast.literal_eval(raw_list_str)))  # de-duplicate

            data.append((time_step, location, ids))

    return pd.DataFrame(data, columns=["TimeStep", "Location", "Ids"])


def build_transmission_network(
        trans_events_df: pd.DataFrame,
        infect_hist_df: pd.DataFrame,
        rng_seed: int = 12345,
) -> nx.DiGraph:
    """
    Build a directed transmission network from event and history tables.

    Parameters
    ----------
    trans_events_df : pandas.DataFrame
        Transmission events with columns ``TimeStep``, ``Location``, ``Ids``.
    infect_hist_df : pandas.DataFrame
        Infection history with columns ``TimeStep``, ``Location``, ``Ids``.
    rng_seed : int, optional
        Seed for uniform infector sampling.

    Returns
    -------
    networkx.DiGraph
        Directed graph with edge attributes ``timeStep`` and ``location``.

    Notes
    -----
    For each exposed individual at (t, L), one infector is sampled uniformly
    from the infectious pool at (t, L), excluding self-loops.
    """
    rng = default_rng(rng_seed)

    # Build fast lookup: (TimeStep, Location) -> List[IDs]
    infection_lookup = {(int(row.TimeStep), int(row.Location)): row.Ids for row in infect_hist_df.itertuples()}

    edges = []

    for row in trans_events_df.itertuples():
        t = int(row.TimeStep)
        loc = int(row.Location)
        exposed = row.Ids

        potential = infection_lookup.get((t, loc), [])
        if not potential:
            continue

        # Convert to numpy array once for faster filtering
        potential = np.asarray(potential, dtype=int)

        for infectee in exposed:
            infectee = int(infectee)
            valid = potential[potential != infectee]
            if valid.size == 0:
                continue

            infector = int(rng.choice(valid))
            edges.append((infector, infectee, {"timeStep": t, "location": loc}))

    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G


def remove_reinfections(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Resolve nodes with multiple parents by keeping the earliest incoming edge.

    Parameters
    ----------
    graph : networkx.DiGraph
        Raw transmission network.

    Returns
    -------
    networkx.DiGraph
        Copy of the graph with at most one parent per node.

    Notes
    -----
    The earliest edge is defined by the smallest ``timeStep`` attribute; missing
    values are treated as +inf. This is a pragmatic simplification for a single
    outbreak backbone.
    """
    cleaned = graph.copy()
    nodes_multi = [n for n, d in cleaned.in_degree(cleaned.nodes) if d > 1]

    for node in nodes_multi:
        in_edges = list(cleaned.in_edges(node, data=True))
        in_edges_sorted = sorted(in_edges, key=lambda x: x[2].get("timeStep", np.inf))
        # Keep earliest, remove the rest
        for u, v, _ in in_edges_sorted[1:]:
            cleaned.remove_edge(u, v)

    return cleaned


def select_target_component(graph: nx.DiGraph, target_size: int) -> nx.DiGraph:
    """
    Select a weakly connected component closest to the target size.

    Parameters
    ----------
    graph : networkx.DiGraph
        Input graph with one or more weakly connected components.
    target_size : int
        Desired component size in nodes.

    Returns
    -------
    networkx.DiGraph
        Subgraph induced by the selected component.
    """
    comps = list(nx.weakly_connected_components(graph))
    comps.sort(key=len, reverse=True)
    selected = min(comps, key=lambda c: abs(len(c) - target_size))

    return nx.DiGraph(graph.subgraph(selected).copy())


def build_msa_tree(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Compute a Maximum Spanning Arborescence (MSA) to enforce a rooted, acyclic tree.

    Parameters
    ----------
    graph : networkx.DiGraph
        Input directed graph for a single connected component.

    Returns
    -------
    networkx.DiGraph
        A directed arborescence with preserved edge attributes.

    Notes
    -----
    Edge weights are set to ``-timeStep`` so earlier transmission edges are
    preferred when cycles must be broken.
    """
    weighted = graph.copy()
    for u, v, d in weighted.edges(data=True):
        d["weight"] = -int(d.get("timeStep", 0))

    msa = maximum_spanning_arborescence(weighted, attr="weight", preserve_attrs=True)
    # Ensure type is DiGraph
    return nx.DiGraph(msa)


def _degree_rows(graph: nx.DiGraph, label: str) -> list[dict[str, object]]:
    """
    Expand in/out degrees into row-wise records.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph to summarize.
    label : str
        Graph label stored in each row.

    Returns
    -------
    list of dict
        Each dict has keys ``graph``, ``degree_type``, and ``value``.
    """
    in_deg = np.array([d for _, d in graph.in_degree(graph.nodes)], dtype=int)
    out_deg = np.array([d for _, d in graph.out_degree(graph.nodes)], dtype=int)

    rows: list[dict[str, object]] = []
    rows.extend({"graph": label, "degree_type": "in", "value": int(v)} for v in in_deg)
    rows.extend({"graph": label, "degree_type": "out", "value": int(v)} for v in out_deg)
    return rows


def summarise_graph(graph: nx.DiGraph, label: str) -> dict[str, Any]:
    """
    Summarize degree and component statistics for a directed graph.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph to summarize.
    label : str
        Label stored in the summary row.

    Returns
    -------
    dict
        Summary metrics keyed by name, including node/edge counts and degree
        statistics.
    """
    in_degs = np.array([d for _, d in graph.in_degree(graph.nodes)], dtype=int)
    out_degs = np.array([d for _, d in graph.out_degree(graph.nodes)], dtype=int)

    summary = {
        "label": label,
        "n_nodes": int(graph.number_of_nodes()),
        "n_edges": int(graph.number_of_edges()),
        "n_components": int(nx.number_weakly_connected_components(graph)),
        "max_in_degree": int(in_degs.max()) if in_degs.size else 0,
        "max_out_degree": int(out_degs.max()) if out_degs.size else 0,
        "mean_out_degree": float(out_degs.mean()) if out_degs.size else 0.0,
        "prop_in_degree_gt1": float(np.mean(in_degs > 1)) if in_degs.size else 0.0,
        "prop_out_degree_ge10": float(np.mean(out_degs >= 10)) if out_degs.size else 0.0,
    }
    return summary


# -----------------------------
# Main execution
# -----------------------------


def main(config_path: str | Path = "config.yaml") -> None:
    configure_logging()
    LOGGER.info("scovmod: starting")
    config = load_config(config_path)
    workflow = get_config_value(config, "workflows.scovmod", default={})
    infection_path = resolve_configured_path(config, "paths.scovmod.infection_path")
    transmission_path = resolve_configured_path(config, "paths.scovmod.transmission_path")
    target_component_size = int(workflow.get("target_component_size"))
    rng_seed = int(get_config_value(config, "rng_seed", default=DEFAULT_SEED))

    if not infection_path.exists():
        raise FileNotFoundError(f"Missing infection history file: {infection_path}")
    if not transmission_path.exists():
        raise FileNotFoundError(f"Missing transmission events file: {transmission_path}")

    out_dir = resolve_configured_output_path(config, "outputs.scovmod.directory")
    tree_path = resolve_configured_output_path(config, "outputs.scovmod.tree_path")
    heterogeneity_path = resolve_configured_output_path(config, "outputs.scovmod.heterogeneity_path")
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("scovmod: parsing inputs")
    # Parse
    infect_df = parse_scovmod_outputs(infection_path)
    trans_df = parse_scovmod_outputs(transmission_path)

    LOGGER.info("scovmod: building tree")
    # Build raw network
    raw_G = build_transmission_network(trans_df, infect_df, rng_seed=rng_seed)

    # Raw network diagnostics
    raw_components = [len(c) for c in nx.weakly_connected_components(raw_G)]

    # Clean multiple parents
    clean_G = remove_reinfections(raw_G)
    clean_components = [len(c) for c in nx.weakly_connected_components(clean_G)]

    component_sizes = pd.DataFrame({
        "kind": ["raw"] * len(raw_components) + ["cleaned"] * len(clean_components),
        "value": raw_components + clean_components
    })


    # Select target component
    comp_G = select_target_component(
        clean_G,
        target_size=target_component_size,
    )

    degree_rows: list[dict[str, object]] = []
    degree_rows.extend(_degree_rows(raw_G, "raw"))
    degree_rows.extend(_degree_rows(clean_G, "cleaned"))
    degree_rows.extend(_degree_rows(comp_G, "selected_component"))
    degree_df = pd.DataFrame(degree_rows)

    # Build tree (MSA)
    tree_G = build_msa_tree(comp_G)

    # Save graph
    nx.write_gml(tree_G, tree_path)

    offspring_counts = np.array(list(dict(clean_G.out_degree(clean_G.nodes)).values()))
    results = heterogeneity(offspring_counts)
    heterogeneity_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Save summary stats
    summaries = [
        summarise_graph(raw_G, "raw"),
        summarise_graph(clean_G, "cleaned"),
        summarise_graph(comp_G, "selected_component"),
        summarise_graph(tree_G, "final_tree"),
    ]

    summary_df = pd.DataFrame(summaries)

    summary_df.to_parquet(out_dir / "tree_summary.parquet", index=False)
    component_sizes.to_parquet(out_dir / "component_sizes.parquet", index=False)
    degree_df.to_parquet(out_dir / "degree_distributions.parquet", index=False)
    LOGGER.info("scovmod: done")

if __name__ == "__main__":
    main()
