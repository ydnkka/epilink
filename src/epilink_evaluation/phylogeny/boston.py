"""Build uncensored Boston genetic and dated trees from the source alignment."""

from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Phylo, SeqIO

from ..provenance import complete_artifact, digest_file, fingerprint, valid_artifact
from .external import run_command
from .trees import root_signature, validate_tree


def alignment_length(path, cases):
    records = list(SeqIO.parse(path, "fasta"))
    names = [record.id for record in records]
    lengths = {len(record.seq) for record in records}
    if (
        len(names) != len(set(names))
        or set(names) != set(cases.case_id)
        or len(lengths) != 1
    ):
        raise ValueError(
            "Boston alignment needs exactly one equal-length sequence per case"
        )
    return lengths.pop()


def raw_boston_tree(root, alignment, cases, tools, settings):
    """TN93 all-pair distances -> FastME; never fill censored distances with zero."""
    alignment = Path(alignment)
    length = alignment_length(alignment, cases)
    signature = {
        "kind": "boston-raw-tree-v1",
        "alignment_sha256": digest_file(alignment),
        "case_ids": cases.case_id.tolist(),
        "sequence_length": length,
        "tn93_args": ["-t", "1", "-a", "resolve", "-g", "0.05", "-l", "500"],
        "fastme_method": settings["fastme_method"],
        "tools": {k: tools[k] for k in ("tn93", "fastme")},
        "rooting": "midpoint",
        "negative_branches": "clip_zero",
    }
    directory = Path(root) / "artifacts/trees" / fingerprint(signature)[:20]
    result = directory / "raw.nwk"
    if valid_artifact(directory, signature):
        validate_tree(result, cases.case_id)
        return result, length
    directory.mkdir(parents=True, exist_ok=True)
    pairs_path = directory / "all_pairs_tn93.csv"
    run_command(
        [
            tools["tn93"]["path"],
            "-q",
            *signature["tn93_args"],
            "-o",
            str(pairs_path),
            str(alignment),
        ],
        directory,
        "tn93",
        settings["command_timeout_seconds"],
    )
    pairs = pd.read_csv(pairs_path, dtype={"ID1": str, "ID2": str})
    if not {"ID1", "ID2", "Distance"}.issubset(pairs.columns):
        raise ValueError("Unexpected all-pair TN93 output schema")
    n = len(cases)
    order = pd.Series(np.arange(n), index=cases.case_id)
    a, b = pairs.ID1.map(order), pairs.ID2.map(order)
    distances = pairs.Distance.to_numpy(float)
    if (
        len(pairs) != n * (n - 1) // 2
        or a.isna().any()
        or b.isna().any()
        or (a == b).any()
        or not np.isfinite(distances).all()
        or (distances < 0).any()
    ):
        raise ValueError(
            "Uncensored TN93 must contain every distinct Boston pair with finite distances"
        )
    low, high = (
        np.minimum(a.to_numpy(int), b.to_numpy(int)),
        np.maximum(a.to_numpy(int), b.to_numpy(int)),
    )
    if pd.DataFrame({"a": low, "b": high}).duplicated().any():
        raise ValueError("Uncensored TN93 contains duplicate unordered pairs")
    matrix = np.zeros((n, n), dtype=float)
    matrix[low, high] = distances
    matrix[high, low] = distances
    aliases = {f"s{i:08d}": str(case_id) for i, case_id in enumerate(cases.case_id)}
    matrix_path = directory / "distances.phy"
    with matrix_path.open("w") as handle:
        handle.write(f"{n}\n")
        for i, alias in enumerate(aliases):
            handle.write(f"{alias} ")
            np.savetxt(handle, matrix[i : i + 1], fmt="%.15f")
    original = directory / "fastme.nwk"
    argv = [
        tools["fastme"]["path"],
        "-i",
        str(matrix_path),
        "-o",
        str(original),
        "-m",
        settings["fastme_method"],
    ]
    run_command(argv, directory, "fastme", settings["command_timeout_seconds"])
    tree = Phylo.read(original, "newick")
    for tip in tree.get_terminals():
        tip.name = aliases[tip.name]
    negative = 0
    for node in tree.find_clades():
        if node.branch_length is not None and node.branch_length < 0:
            negative += 1
            node.branch_length = 0.0
    if tree.total_branch_length() > 0:
        tree.root_at_midpoint()
    Phylo.write(tree, result, "newick", format_branch_length="%.12g")
    validate_tree(result, cases.case_id)
    complete_artifact(
        directory,
        signature,
        [
            "raw.nwk",
            "fastme.nwk",
            "distances.phy",
            "all_pairs_tn93.csv",
            "tn93.stdout.log",
            "tn93.stderr.log",
            "fastme.stdout.log",
            "fastme.stderr.log",
        ],
        n_cases=n,
        n_pairs=len(pairs),
        units="substitutions_per_site",
        negative_branches_clipped=negative,
        rooting="midpoint",
        root_split=root_signature(tree),
        command=argv,
    )
    return result, length


def dated_boston_tree(root, raw_path, cases, length, tools, settings):
    """TreeTime dates the new genetic tree using real Boston collection dates."""
    if cases.sample_date.isna().any():
        raise ValueError("Boston sample dates must be present to date the tree")
    signature = {
        "kind": "boston-dated-tree-v1",
        "raw_sha256": digest_file(raw_path),
        "dates": cases[["case_id", "sample_date"]].astype(str).to_dict("records"),
        "sequence_length": length,
        "treetime": tools["treetime"],
        "clock_filter": settings["clock_filter"],
        "rng_seed": settings["rng_seed"],
    }
    directory = Path(root) / "artifacts/trees" / fingerprint(signature)[:20]
    result = directory / "dated.nwk"
    if valid_artifact(directory, signature):
        validate_tree(result, cases.case_id)
        return result
    directory.mkdir(parents=True, exist_ok=True)
    dates = (
        cases[["case_id", "sample_date"]]
        .rename(columns={"case_id": "name", "sample_date": "date"})
        .copy()
    )
    dates["date"] = pd.to_datetime(dates.date).dt.strftime("%Y-%m-%d")
    dates.to_csv(directory / "dates.csv", index=False)
    argv = [
        tools["treetime"]["path"],
        "--tree",
        str(raw_path),
        "--dates",
        str(directory / "dates.csv"),
        "--sequence-length",
        str(length),
        "--clock-filter",
        str(settings["clock_filter"]),
        "--rng-seed",
        str(settings["rng_seed"]),
        "--outdir",
        str(directory / "treetime"),
    ]
    run_command(argv, directory, "treetime", settings["command_timeout_seconds"])
    tree = Phylo.read(directory / "treetime/timetree.nexus", "nexus")
    for node in tree.find_clades():
        if isinstance(node.confidence, str):
            node.name, node.confidence = node.confidence, None
        if node.comment and node.comment.startswith("[") and node.comment.endswith("]"):
            node.comment = node.comment[1:-1]
    Phylo.write(tree, result, "newick", format_branch_length="%.12g")
    validate_tree(result, cases.case_id)
    complete_artifact(
        directory,
        signature,
        [
            "dated.nwk",
            "dates.csv",
            "treetime/timetree.nexus",
            "treetime.stdout.log",
            "treetime.stderr.log",
        ],
        units="calendar_years",
        command=argv,
        rooting="TreeTime clock root",
        root_split=root_signature(tree),
        root_changed=root_signature(tree)
        != root_signature(validate_tree(raw_path, cases.case_id)),
    )
    return result
