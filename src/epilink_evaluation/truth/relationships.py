"""Binary-lifting LCA relationships, extracted and checked against graph paths."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


class TreeIndex:
    def __init__(self, tree):
        if (
            not tree.is_directed()
            or not len(tree)
            or not nx.is_directed_acyclic_graph(tree)
            or any(degree > 1 for _, degree in tree.in_degree())
        ):
            raise ValueError("Truth requires a nonempty, acyclic single-parent forest")
        self.nodes = list(tree)
        self.lookup = {node: i for i, node in enumerate(self.nodes)}
        self.depth = np.zeros(len(tree), dtype=np.int32)
        self.root = np.zeros(len(tree), dtype=np.int32)
        parent = np.arange(len(tree), dtype=np.int32)
        for node in nx.topological_sort(tree):
            i = self.lookup[node]
            parents = list(tree.predecessors(node))
            if parents:
                p = self.lookup[parents[0]]
                parent[i] = p
                self.depth[i] = self.depth[p] + 1
                self.root[i] = self.root[p]
            else:
                self.root[i] = i
        self.up = [parent]
        for _ in range(max(1, int(self.depth.max()).bit_length())):
            self.up.append(self.up[-1][self.up[-1]])

    def classify(self, a, b):
        a, b = np.asarray(a, dtype=np.int32), np.asarray(b, dtype=np.int32)
        if np.any(a == b):
            raise ValueError("Self-pairs have no pairwise truth label")
        same = self.root[a] == self.root[b]
        x, y = a.copy(), b.copy()
        swap = self.depth[x] < self.depth[y]
        x[swap], y[swap] = y[swap], x[swap]
        difference = self.depth[x] - self.depth[y]
        for k, up in enumerate(self.up):
            mask = ((difference >> k) & 1).astype(bool)
            x[mask] = up[x[mask]]
        for up in reversed(self.up):
            mask = up[x] != up[y]
            x[mask], y[mask] = up[x[mask]], up[y[mask]]
        lca = np.where(x == y, x, self.up[0][x])
        da, db = self.depth[a] - self.depth[lca], self.depth[b] - self.depth[lca]
        ad = same & ((da == 0) | (db == 0))
        ca = same & ~ad
        result = pd.DataFrame({"AD": ad.astype(np.int8), "CA": ca.astype(np.int8)})
        for name, values, valid in (
            ("m", da + db - 1, ad),
            ("m1", da - 1, ca),
            ("m2", db - 1, ca),
            ("tree_hops", da + db, same),
        ):
            values = pd.array(values, dtype="Int32")
            values[~valid] = pd.NA
            result[name] = values
        result["M"] = result.m.where(ad, result.m1 + result.m2)
        return result


def relationship_table(tree):
    index = TreeIndex(tree)
    a, b = np.triu_indices(len(tree), k=1)
    chunks = []
    for start in range(0, len(a), 500_000):
        chunks.append(
            index.classify(a[start : start + 500_000], b[start : start + 500_000])
        )
    if not chunks:
        raise ValueError("At least two transmission-tree cases are required")
    frame = pd.concat(chunks, ignore_index=True)
    frame.insert(0, "pair_id", np.arange(len(frame), dtype=np.int64))
    frame["node_a"], frame["node_b"] = a.astype(np.int32), b.astype(np.int32)
    return frame


def reference_memberships(tree, case_ids=None):
    """Each case belongs to its own and its infector's neighborhood.

    Reference labels retain unsampled infectors; only the evaluated case universe
    is restricted. String identifiers work for synthetic and empirical inputs.
    """
    labels = {node: i for i, node in enumerate(tree)}
    selected = None if case_ids is None else set(map(str, case_ids))
    return {
        str(node): {
            labels[node],
            *(labels[parent] for parent in tree.predecessors(node)),
        }
        for node in tree
        if selected is None or str(node) in selected
    }
