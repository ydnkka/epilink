from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Phylo

from ..provenance import complete_artifact, digest_file, fingerprint, valid_artifact
from ..schemas import DISTANCES
from .external import command_identity, run_command


def validate_tree(path, case_ids):
    tree = Phylo.read(path, "newick")
    names = [str(tip.name) for tip in tree.get_terminals()]
    if len(names) != len(set(names)) or set(names) != set(map(str, case_ids)):
        raise ValueError("Tree tips do not exactly match the sampled case universe")
    for node in tree.find_clades():
        if node.branch_length is not None and (
            not np.isfinite(node.branch_length) or node.branch_length < 0
        ):
            raise ValueError("Tree contains non-finite or negative branch lengths")
    return tree


def root_signature(tree):
    return sorted(
        [
            sorted(str(tip.name) for tip in clade.get_terminals())
            for clade in tree.root.clades
        ]
    )


def raw_tree(config, observations, cases, process, dataset_id, implementation):
    settings = config["treecluster"]
    if (
        settings["raw_rooting"] != "midpoint"
        or settings["negative_branches"] != "clip_zero"
    ):
        raise ValueError(
            "Supported raw-tree policies are midpoint rooting and explicit clip_zero"
        )
    tool = command_identity(settings["executables"]["fastme"])
    signature = {
        "kind": "raw-tree-v1",
        "dataset": dataset_id,
        "process": process,
        "fastme": tool,
        "method": settings["fastme_method"],
        "rooting": settings["raw_rooting"],
        "negative_branches": settings["negative_branches"],
        "sequence_length": config["simulation"]["sequence_length"],
        "implementation": implementation,
    }
    directory = (
        Path(config["output_directory"])
        / "artifacts/trees"
        / fingerprint(signature)[:20]
    )
    path = directory / "raw.nwk"
    if valid_artifact(directory, signature):
        validate_tree(path, cases.case_id)
        return path
    directory.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((len(cases), len(cases)), dtype=float)
    distances = (
        observations[DISTANCES[process]].to_numpy(float)
        / config["simulation"]["sequence_length"]
    )
    a, b = observations.a.to_numpy(), observations.b.to_numpy()
    matrix[a, b] = distances
    matrix[b, a] = distances
    matrix_path = directory / "distances.phy"
    # Internal labels avoid PHYLIP truncation of arbitrary case identifiers.
    aliases = {f"s{i:08d}": str(case) for i, case in enumerate(cases.case_id)}
    with matrix_path.open("w") as handle:
        handle.write(f"{len(cases)}\n")
        for i, alias in enumerate(aliases):
            handle.write(f"{alias} ")
            np.savetxt(handle, matrix[i : i + 1], fmt="%.12g")
    original_path = directory / "fastme.nwk"
    argv = [
        tool["path"],
        "-i",
        str(matrix_path),
        "-o",
        str(original_path),
        "-m",
        settings["fastme_method"],
    ]
    run_command(argv, directory, "fastme", settings["command_timeout_seconds"])
    tree = Phylo.read(original_path, "newick")
    for tip in tree.get_terminals():
        tip.name = aliases[tip.name]
    negative = 0
    for node in tree.find_clades():
        if node.branch_length is not None and node.branch_length < 0:
            negative += 1
            node.branch_length = 0.0
    rooted = "midpoint"
    if tree.total_branch_length() > 0:
        tree.root_at_midpoint()
    else:
        rooted = "zero-length tree; equivalent original root retained"
    Phylo.write(tree, path, "newick", format_branch_length="%.12g")
    validate_tree(path, cases.case_id)
    complete_artifact(
        directory,
        signature,
        ["raw.nwk", "fastme.nwk", "distances.phy"],
        units="substitutions_per_site",
        negative_branches_clipped=negative,
        rooting=rooted,
        root_split=root_signature(tree),
        command=argv,
    )
    return path


def dated_tree(config, raw_path, cases, process, dataset_id, implementation):
    settings = config["treecluster"]
    tool = command_identity(settings["executables"]["treetime"])
    signature = {
        "kind": "dated-tree-v1",
        "dataset": dataset_id,
        "process": process,
        "raw_sha256": digest_file(raw_path),
        "treetime": tool,
        "clock_filter": settings["clock_filter"],
        "sequence_length": config["simulation"]["sequence_length"],
        "implementation": implementation,
    }
    directory = (
        Path(config["output_directory"])
        / "artifacts/trees"
        / fingerprint(signature)[:20]
    )
    path = directory / "dated.nwk"
    if valid_artifact(directory, signature):
        validate_tree(path, cases.case_id)
        return path
    directory.mkdir(parents=True, exist_ok=True)
    dates = pd.DataFrame(
        {
            "name": cases.case_id.astype(str),
            "date": [
                (date(2020, 1, 1) + timedelta(days=int(np.rint(day)))).isoformat()
                for day in cases.sample_date
            ],
        }
    )
    dates.to_csv(directory / "dates.csv", index=False)
    argv = [
        tool["path"],
        "--tree",
        str(raw_path),
        "--dates",
        str(directory / "dates.csv"),
        "--sequence-length",
        str(config["simulation"]["sequence_length"]),
        "--clock-filter",
        str(settings["clock_filter"]),
        "--outdir",
        str(directory / "treetime"),
    ]
    run_command(argv, directory, "treetime", settings["command_timeout_seconds"])
    tree = Phylo.read(directory / "treetime/timetree.nexus", "nexus")
    Phylo.write(tree, path, "newick", format_branch_length="%.12g")
    validate_tree(path, cases.case_id)
    original = validate_tree(raw_path, cases.case_id)
    complete_artifact(
        directory,
        signature,
        ["dated.nwk", "dates.csv", "treetime/timetree.nexus"],
        units="calendar_years",
        command=argv,
        rooting="TreeTime clock root",
        root_split=root_signature(tree),
        root_changed=root_signature(tree) != root_signature(original),
    )
    return path
