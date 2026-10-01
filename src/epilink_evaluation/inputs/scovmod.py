"""Reproducible SCoVMod backbone preparation from the preserved raw CSVs."""

from __future__ import annotations

import ast
import csv
from os.path import relpath
from pathlib import Path

import networkx as nx
import numpy as np

from ..provenance import complete_artifact, digest_file, valid_artifact, write_json
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


def prepare_scovmod_inputs(config):
    """Build or reuse the configured backbone and its provenance artifact."""
    settings = config["inputs"]
    tree_path = Path(settings["tree_path"])
    output = tree_path.parent
    source_json = Path(
        settings.get("tree_source_path") or tree_path.with_suffix(".source.json")
    )
    inputs = {
        key: {"path": str(settings[key]), "sha256": digest_file(settings[key])}
        for key in ("infection_path", "transmission_path")
    }
    signature = {
        "kind": "scovmod-inputs-v2",
        "inputs": inputs,
        "tree_seed": settings["tree_seed"],
        "target_component_size": settings["target_component_size"],
        "tree_path": str(tree_path),
        "tree_source_path": str(source_json),
        "implementation": digest_file(__file__),
    }
    if valid_artifact(output, signature):
        TreeIndex(nx.read_gml(tree_path))
        return output
    tree = build_tree(
        parse_scovmod(settings["infection_path"]),
        parse_scovmod(settings["transmission_path"]),
        settings["target_component_size"],
        settings["tree_seed"],
    )
    output.mkdir(parents=True, exist_ok=True)
    nx.write_gml(tree, tree_path)
    write_json(
        source_json,
        {
            "inputs": inputs,
            "tree_sha256": digest_file(tree_path),
            "n_cases": len(tree),
            "seed": settings["tree_seed"],
            "target_size": settings["target_component_size"],
            "implementation_sha256": digest_file(__file__),
            "tie_order": "sorted integer IDs; earliest time then infector ID",
        },
    )
    complete_artifact(
        output,
        signature,
        [tree_path.name, relpath(source_json, output)],
        n_cases=len(tree),
        tree_sha256=digest_file(tree_path),
    )
    return output


def prepare_tree(config):
    """Use a supplied tree or prepare a managed SCoVMod input artifact."""
    settings = config["inputs"]
    path = Path(settings["tree_path"])
    # Explicit prebuilt trees need not have raw SCoVMod inputs or a manifest.
    if path.exists() and not (path.parent / "manifest.json").exists():
        TreeIndex(nx.read_gml(path))
        return path
    prepare_scovmod_inputs(config)
    return path
