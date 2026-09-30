"""Reproducible SCoVMod backbone preparation from the preserved raw CSVs."""

from __future__ import annotations

import ast
import csv
from pathlib import Path

import networkx as nx
import numpy as np

from ..provenance import digest_file, write_json
from ..truth import TreeIndex


def parse_scovmod(path):
    with Path(path).open(newline="") as handle:
        rows = csv.reader(handle)
        next(rows, None)
        return [
            (int(row[0]), int(row[1]), sorted(set(ast.literal_eval(",".join(row[2:])))))
            for row in rows
            if row
        ]


def build_tree(infections, transmissions, target_size, seed):
    rng = np.random.default_rng(seed)
    lookup = {(time, location): ids for time, location, ids in infections}
    graph = nx.DiGraph()
    for time, location, exposed in transmissions:
        pool = lookup.get((time, location), [])
        for infectee in exposed:
            possible = [i for i in pool if i != infectee]
            if possible:
                graph.add_edge(
                    int(rng.choice(possible)),
                    infectee,
                    timeStep=time,
                    location=location,
                )
    for node in list(graph):
        incoming = sorted(
            graph.in_edges(node, data=True),
            key=lambda edge: (edge[2]["timeStep"], edge[0]),
        )
        graph.remove_edges_from((a, b) for a, b, _ in incoming[1:])
    components = sorted(
        nx.weakly_connected_components(graph),
        key=lambda c: (abs(len(c) - target_size), -len(c), min(c)),
    )
    if not components:
        raise ValueError("No transmission components could be constructed")
    component = graph.subgraph(components[0]).copy()
    for _, _, attributes in component.edges(data=True):
        attributes["weight"] = -attributes["timeStep"]
    tree = nx.maximum_spanning_arborescence(
        component, attr="weight", preserve_attrs=True
    )
    TreeIndex(tree)
    return tree


def prepare_tree(config):
    settings = config["inputs"]
    path = Path(settings["tree_path"])
    if path.exists():
        TreeIndex(nx.read_gml(path))
        return path
    inputs = {key: settings[key] for key in ("infection_path", "transmission_path")}
    tree = build_tree(
        parse_scovmod(inputs["infection_path"]),
        parse_scovmod(inputs["transmission_path"]),
        settings["target_component_size"],
        settings["tree_seed"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(tree, path)
    write_json(
        path.with_suffix(".source.json"),
        {
            "inputs": {
                key: {"path": value, "sha256": digest_file(value)}
                for key, value in inputs.items()
            },
            "tree_sha256": digest_file(path),
            "n_cases": len(tree),
            "seed": settings["tree_seed"],
            "target_size": settings["target_component_size"],
            "implementation_sha256": digest_file(__file__),
            "tie_order": "sorted integer IDs; earliest time then infector ID",
        },
    )
    return path
