"""Known transmission-hop topology for TreeCluster controls."""

from __future__ import annotations

import networkx as nx
import numpy as np
from Bio.Phylo.BaseTree import Clade, Tree

from ..truth.relationships import TreeIndex


def transmission_hop_tree(tree: nx.DiGraph, case_ids) -> Tree:
    """Represent sampled transmission nodes as zero-length named terminal tips.

    ``TreeIndex`` validates the transmission topology. Exactly one root is
    required, even if samples occupy only one component of a forest. Each
    retained transmission node becomes an unlabeled internal clade; each
    transmission edge has length one. Only branches without any sampled
    descendants are removed: unary paths, unsampled intermediates, the original
    root, and multifurcations are retained. Thus all sampled-node hop distances
    and depths from the transmission root are preserved.

    Case IDs match tree nodes by their unique string representations, as in
    ``reference_memberships``. Their order does not affect the result. At least
    one observed case is required; a single-node isolate is supported. The
    result can be serialized with ``Bio.Phylo.write``. It is a known-topology
    control in transmission-hop units, not an inferred molecular genealogy.
    """
    index = TreeIndex(tree)
    roots = [i for i, node in enumerate(index.nodes) if tree.in_degree(node) == 0]
    if len(roots) != 1:
        raise ValueError(
            "Transmission hop trees require a single rooted transmission tree; "
            "multi-root forests have no defined between-component hop distances"
        )
    if isinstance(case_ids, (str, bytes)):
        raise ValueError("case_ids must be a collection of case identifiers")
    labels = list(map(str, case_ids))
    if not labels:
        raise ValueError("At least one observed case is required")
    if any(not label for label in labels) or len(labels) != len(set(labels)):
        raise ValueError("Observed case identifiers must be nonempty and unique")
    lookup = {str(node): i for i, node in enumerate(index.nodes)}
    if len(lookup) != len(index.nodes):
        raise ValueError("Transmission node identifiers are ambiguous as strings")
    missing = set(labels) - lookup.keys()
    if missing:
        raise ValueError(
            f"Observed cases are absent from the transmission tree: {sorted(missing)}"
        )
    sampled = {lookup[label]: label for label in labels}
    keep = np.zeros(len(index.nodes), dtype=bool)
    parents = index.up[0]
    # Each retained ancestor is visited at most once, even for many samples.
    for i in sampled:
        while not keep[i]:
            keep[i] = True
            i = parents[i]
    root = roots[0]
    clades = {
        i: Clade(branch_length=0.0 if i == root else 1.0)
        for i in range(len(index.nodes))
        if keep[i]
    }
    for i, clade in clades.items():
        if i != root:
            clades[parents[i]].clades.append(clade)
        if i in sampled:
            clade.clades.append(Clade(name=sampled[i], branch_length=0.0))
    return Tree(root=clades[root], rooted=True)
