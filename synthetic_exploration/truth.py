"""Exact full-tree relationships; never remove unsampled intermediates."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .common import digest_file, fingerprint, write_json, log

RELATIONSHIPS = (
    "direct", "shared_infector", "indirect_ancestor", "other_shared_ancestry",
    "separate_introductions",
)
ENCODING_COLUMNS = ["AD", "CA", "m", "m1", "m2", "M"]


def canonical_encoding(frame):
    """Return unordered relationship keys, without changing case-level depths.

    CA(a,b) and CA(b,a) are one relationship. Summary keys use m1 <= m2;
    pair tables retain m1/m2 aligned with CaseID1/CaseID2. Returning only the
    encoding columns prevents sorted branches being mistaken for case depths.
    """
    encoded = frame[ENCODING_COLUMNS].copy()
    swap = (encoded.CA.eq(1) & encoded.m1.gt(encoded.m2)).fillna(False)
    first = encoded.loc[swap, "m1"].copy()
    encoded.loc[swap, "m1"] = encoded.loc[swap, "m2"]
    encoded.loc[swap, "m2"] = first
    return encoded


class TreeIndex:
    """Binary-lifting LCA index supporting a single-parent directed forest."""

    def __init__(self, tree):
        if not nx.is_directed_acyclic_graph(tree) or any(d > 1 for _, d in tree.in_degree()):
            raise ValueError("Truth must be an acyclic single-parent transmission forest.")
        self.nodes = list(tree)
        self.lookup = {node: i for i, node in enumerate(self.nodes)}
        n = len(self.nodes)
        self.depth = np.zeros(n, dtype=np.int32)
        self.root = np.zeros(n, dtype=np.int32)
        parent = np.arange(n, dtype=np.int32)
        for node in nx.topological_sort(tree):
            i = self.lookup[node]
            predecessors = list(tree.predecessors(node))
            if predecessors:
                p = self.lookup[predecessors[0]]
                parent[i] = p
                self.depth[i] = self.depth[p] + 1
                self.root[i] = self.root[p]
            else:
                self.root[i] = i
        self.up = [parent]
        for _ in range(max(1, int(self.depth.max(initial=0)).bit_length())):
            self.up.append(self.up[-1][self.up[-1]])

    def classify(self, a, b):
        a, b = np.asarray(a, dtype=np.int32), np.asarray(b, dtype=np.int32)
        if np.any(a == b):
            raise ValueError("Self-pairs are excluded from the pairwise investigation.")
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
        lca = np.where(x == y, x, self.up[0][x]).astype(np.int32)
        da, db = self.depth[a] - self.depth[lca], self.depth[b] - self.depth[lca]
        hops = da + db
        ancestor = (da == 0) | (db == 0)
        rel = np.full(len(a), 3, dtype=np.int8)
        rel[ancestor] = 2
        rel[(da == 1) & (db == 1)] = 1
        rel[ancestor & (hops == 1)] = 0
        rel[~same] = 4
        lca[~same] = -1
        da[~same], db[~same], hops[~same] = -1, -1, -1
        return lca, da, db, hops, rel

    def annotate(self, frame, batch_size=500_000):
        a = frame.CaseA.map(self.lookup).to_numpy(dtype=np.int32)
        b = frame.CaseB.map(self.lookup).to_numpy(dtype=np.int32)
        results = [np.empty(len(frame), dtype=np.int32) for _ in range(5)]
        for start in range(0, len(frame), batch_size):
            stop = start + batch_size
            for dest, values in zip(results, self.classify(a[start:stop], b[start:stop])):
                dest[start:stop] = values
        frame["index_a"], frame["index_b"] = a, b
        for column, values in zip(
            ("lca_index", "lca_steps_a", "lca_steps_b", "tree_hops"), results[:4]
        ):
            frame[column] = values
        frame["relationship"] = pd.Categorical.from_codes(results[4], RELATIONSHIPS)
        if "IsRelated" in frame:
            np.testing.assert_array_equal(frame.IsRelated.to_numpy(bool), results[4] < 2)
        frame["IsRelated"] = results[4] < 2
        frame = encode_epilink(frame)
        return frame


def encode_epilink(frame):
    """EpiLink depths count intermediates, not transmission edges.

    m1 is always the branch to the first case; m2 to the second case. For an
    AD pair the ancestor can be either case (identified by lca_index).
    M combines the active class: m for AD, m1+m2 for CA. It excludes the
    shared ancestor for CA, and therefore differs from tree edge distance.
    Missing values mean not applicable, never zero intermediates.
    """
    frame = frame.rename(columns={"CaseA": "CaseID1", "CaseB": "CaseID2"})
    a = frame.lca_steps_a.to_numpy()
    b = frame.lca_steps_b.to_numpy()
    connected = frame.tree_hops.to_numpy() >= 0
    ad = connected & ((a == 0) | (b == 0))
    ca = connected & (a > 0) & (b > 0)
    frame["AD"] = ad.astype(np.int8)
    frame["CA"] = ca.astype(np.int8)
    for name, values, valid in [
        ("m", frame.tree_hops.to_numpy() - 1, ad),
        ("m1", a - 1, ca), ("m2", b - 1, ca),
    ]:
        encoded = pd.array(values, dtype="Int32")
        encoded[~valid] = pd.NA
        frame[name] = encoded
    frame["M"] = frame["m"].where(ad, frame["m1"] + frame["m2"])
    if np.any((ad.astype(int) + ca.astype(int))[connected] != 1):
        raise AssertionError("Every connected, non-self pair has exactly one EpiLink class.")
    front = ["CaseID1", "CaseID2", "AD", "CA", "m", "m1", "m2", "M"]
    return frame[front + [c for c in frame if c not in front]]


def prepare_relationships(tree_path, output_root):
    """Persist a seed-independent truth table once per full transmission tree."""
    import json
    from pathlib import Path
    signature = {"schema": 3, "tree_sha256": digest_file(tree_path),
                 "truth_code_sha256": digest_file(__file__)}
    key = fingerprint(signature)
    directory = Path(output_root) / "trees" / key[:20]
    manifest = directory / "manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["fingerprint"] != key or digest_file(directory / "relationships.parquet") != saved["sha256"]:
            raise ValueError("Shared tree relationship cache was modified.")
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    tree = nx.read_gml(tree_path)
    log(f"Encoding all relationships once for the {len(tree):,}-case tree")
    nodes = np.asarray(list(tree))
    a, b = np.triu_indices(len(nodes), k=1)
    frame = TreeIndex(tree).annotate(pd.DataFrame({"CaseA": nodes[a], "CaseB": nodes[b]}))
    frame.insert(8, "pair_id", np.arange(len(frame), dtype=np.int64))
    frame.to_parquet(directory / "relationships.parquet", index=False)
    # Examples retain case-aligned depths; summary counts merge branch swaps.
    examples = frame.drop_duplicates(["AD", "CA", "m", "m1", "m2"])
    examples.to_csv(directory / "encoding_examples.csv", index=False)
    counts = canonical_encoding(frame).groupby(ENCODING_COLUMNS, dropna=False, observed=True).size().rename("n_pairs").reset_index()
    counts.to_csv(directory / "encoding_counts.csv", index=False)
    write_json(manifest, {"fingerprint": key, "signature": signature,
        "n_cases": len(tree), "n_pairs": len(frame),
        "sha256": digest_file(directory / "relationships.parquet"),
        "columns": {"CaseID1": "First case (tree input order, not transmission direction)",
                    "CaseID2": "Second case", "AD": "Ancestor-descendant indicator",
                    "CA": "Common-ancestor indicator, neither case ancestral to the other",
                    "m": "AD: number of intermediates between cases",
                    "m1": "CA: intermediates from common ancestor to CaseID1",
                    "m2": "CA: intermediates from common ancestor to CaseID2",
                    "M": "Total intermediates: m for AD, m1+m2 for CA; null for separate trees"},
        "edge_distance": "tree_hops = M+1 for AD and M+2 for CA",
        "relationship_grouping": "CA(a,b)=CA(b,a); encoding_counts.csv sorts CA depths m1<=m2. Pair tables and examples retain case-aligned depths.",
        "missing_counts": "Null for the inactive class; zero is a valid active-class count.",
        "disconnected": "AD=CA=0 and all counts null (not present in this baseline tree)."})
    return directory
