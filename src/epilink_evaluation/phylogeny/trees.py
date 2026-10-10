from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Phylo
from Bio.Phylo.BaseTree import Tree

from ..inputs.synthetic import load_tree_inputs
from ..provenance import complete_artifact, digest_file, fingerprint, valid_artifact
from .external import command_identity


def validate_tree(source, case_ids):
    """Validate a parsed tree or Newick file against the sampled case universe."""
    tree = source if isinstance(source, Tree) else Phylo.read(source, "newick")
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
    """Return sorted list of tip sets under each root clade."""
    return sorted(
        [
            sorted(str(tip.name) for tip in clade.get_terminals())
            for clade in tree.root.clades
        ]
    )


def _prune_reference_tip(tree, reference_name):
    """Remove reference tip from tree. Returns pruned tree (does not modify input)."""
    from copy import deepcopy
    tree = deepcopy(tree)
    for clade in list(tree.find_clades()):
        if clade.name == reference_name and clade.is_terminal():
            tree.prune(clade)
            break
    return tree


def prepare_phylogeny(config, observation_dir, process, dataset_id, implementation):
    """Build IQ-TREE genetic + dated trees from persisted FASTA.
    
    Parameters
    ----------
    config : dict
        Full study configuration with phylogeny block
    observation_dir : Path
        Observation artifact directory (v3 with FASTA/reference/dates)
    process : str
        Sequence process: "deterministic" or "stochastic"
    dataset_id : str
        Dataset identifier for artifact signature
    implementation : dict
        Implementation signature for provenance
        
    Returns
    -------
    Path
        Tree artifact directory containing:
        - raw.nwk (reference-pruned, subs/site)
        - dated.nwk (reference-excluded, days)
        - node_dates.tsv
        - phylogeny.json (metadata)
        - manifest.json
    """
    inputs = load_tree_inputs(observation_dir)
    fasta_path = (
        inputs["deterministic_fasta"]
        if process == "deterministic"
        else inputs["stochastic_fasta"]
    )
    
    phylo_config = config.get("phylogeny", {})
    model = phylo_config.get("model", "JC")
    threads = phylo_config.get("threads", 1)
    seed = phylo_config.get("seed", 2026)
    clock_rate = phylo_config.get("clock_rate")
    iqtree_executable = phylo_config.get("executable", phylo_config.get("iqtree_executable", "iqtree"))
    timeout = phylo_config.get("timeout", 1800)
    
    signature = {
        "kind": "phylogeny-v1",
        "dataset": dataset_id,
        "process": process,
        "fasta_sha256": digest_file(fasta_path),
        "reference_sha256": digest_file(inputs["reference_fasta"]),
        "dates_sha256": digest_file(inputs["sampling_dates"]),
        "n_cases": inputs["n_cases"],
        "iqtree_model": model,
        "iqtree_threads": threads,
        "iqtree_seed": seed,
        "clock_rate": clock_rate,
        "iqtree_executable": command_identity(iqtree_executable),
        "implementation": implementation,
    }
    
    directory = (
        Path(config["output_directory"])
        / "artifacts/trees"
        / fingerprint(signature)[:20]
    )
    
    if valid_artifact(directory, signature):
        validate_tree(directory / "raw.nwk", pd.read_parquet(inputs["cases_parquet"]).case_id)
        validate_tree(directory / "dated.nwk", pd.read_parquet(inputs["cases_parquet"]).case_id)
        return directory
    
    directory.mkdir(parents=True, exist_ok=True)
    
    from epilink import build_phylogenetic_tree_from_fasta, PhylogenyError
    
    try:
        result = build_phylogenetic_tree_from_fasta(
            alignment_fasta=str(fasta_path),
            reference_fasta=str(inputs["reference_fasta"]),
            dates=str(inputs["sampling_dates"]),
            dated=True,
            output_dir=str(directory / "backend"),
            model=model,
            threads=threads,
            seed=seed,
            clock_rate=clock_rate,
            iqtree_executable=iqtree_executable,
            timeout=timeout,
        )
    except PhylogenyError as exc:
        raise RuntimeError(f"IQ-TREE inference failed: {exc}")
    
    raw_tree = Phylo.read(str(result.output_paths["raw_tree"]), "newick")
    reference_name = result.reference_name
    case_ids = pd.read_parquet(inputs["cases_parquet"]).case_id
    
    raw_pruned = _prune_reference_tip(raw_tree, reference_name)
    validate_tree(raw_pruned, case_ids)
    Phylo.write(raw_pruned, directory / "raw.nwk", "newick", format_branch_length="%.12g")
    
    dated_tree = Phylo.read(str(result.output_paths["dated_tree"]), "newick")
    validate_tree(dated_tree, case_ids)
    Phylo.write(dated_tree, directory / "dated.nwk", "newick", format_branch_length="%.12g")
    
    result.node_dates.to_csv(directory / "node_dates.tsv", sep="\t", index=False)
    
    phylogeny_meta = {
        "model": model,
        "threads": threads,
        "seed": seed,
        "clock_rate": result.clock_rate,
        "date_origin": str(result.date_origin) if result.date_origin else None,
        "reference_name": reference_name,
        "units": {
            "raw": "substitutions_per_site",
            "dated": "days",
        },
        "backend_paths": {k: str(v) for k, v in result.output_paths.items()},
    }
    pd.DataFrame([phylogeny_meta]).to_json(directory / "phylogeny.json", orient="records", indent=2)
    
    complete_artifact(
        directory,
        signature,
        ["raw.nwk", "dated.nwk", "node_dates.tsv", "phylogeny.json"],
        units="substitutions_per_site (raw) / days (dated)",
        rooting="IQ-TREE midpoint (raw) / LSD2 clock (dated)",
        root_split=root_signature(raw_pruned),
        dated_root_split=root_signature(dated_tree),
    )
    
    return directory
