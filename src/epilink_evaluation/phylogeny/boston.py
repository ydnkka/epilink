"""Build Boston genetic and dated trees using IQ-TREE from aligned FASTA."""

from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Phylo._io import read as read_phylo
from Bio.Phylo._io import write as write_phylo

from ..provenance import complete_artifact, digest_file, fingerprint, valid_artifact
from .external import command_identity
from .trees import _prune_reference_tip, root_signature, validate_tree


def alignment_length(path, cases):
    """Validate a sample-only alignment against the empirical case universe."""
    identifiers, lengths = [], set()
    for record in SeqIO.parse(path, "fasta"):
        identifiers.append(record.id)
        lengths.add(len(record.seq))
    if (
        not identifiers or len(set(identifiers)) != len(identifiers)
        or set(identifiers) != set(cases.case_id.astype(str))
    ):
        raise ValueError("Boston alignment IDs must match the unique case universe")
    if len(lengths) != 1 or next(iter(lengths)) <= 0:
        raise ValueError("Boston alignment must contain equal-length nonempty sequences")
    return next(iter(lengths))


def prepare_boston_phylogeny(root, alignment_path, reference_path, cases, phylo_config, implementation):
    """Build IQ-TREE trees from Boston alignment.
    
    Parameters
    ----------
    root : Path
        Study output root
    alignment_path : Path
        Boston aligned FASTA (e.g., MGH_DPH_98percent_772samples_aligned.fasta)
    reference_path : Path
        Reference FASTA (e.g., data/sars-cov-2/reference.fasta)
    cases : pd.DataFrame
        Cases with case_id and sample_date columns
    phylo_config : dict
        Phylogeny configuration block
    implementation : dict
        Implementation signature for provenance
        
    Returns
    -------
    tuple
        (tree_directory, alignment_length)
    """
    alignment_path = Path(alignment_path)
    reference_path = Path(reference_path)
    
    sequence_length = alignment_length(alignment_path, cases)
    case_ids = cases.case_id.tolist()
    iqtree_executable = phylo_config.get(
        "executable", phylo_config.get("iqtree_executable", "iqtree")
    )
    iqtree_tool = command_identity(iqtree_executable)
    
    signature = {
        "kind": "boston-phylogeny-v1",
        "alignment_sha256": digest_file(alignment_path),
        "reference_sha256": digest_file(reference_path),
        "case_ids": case_ids,
        "alignment_length": sequence_length,
        "iqtree_model": phylo_config.get("model", "MFP"),
        "iqtree_threads": phylo_config.get("threads", 1),
        "iqtree_seed": phylo_config.get("seed", 2026),
        "clock_rate": phylo_config.get("clock_rate"),
        "iqtree_executable": iqtree_tool,
        "implementation": implementation,
    }
    
    directory = Path(root) / "artifacts/trees" / fingerprint(signature)[:20]
    
    if valid_artifact(directory, signature):
        validate_tree(directory / "raw.nwk", cases.case_id)
        validate_tree(directory / "dated.nwk", cases.case_id)
        return directory, sequence_length
    
    directory.mkdir(parents=True, exist_ok=True)
    
    dates_df = cases[["case_id", "sample_date"]].copy()
    dates_df["sample_date"] = pd.to_datetime(dates_df["sample_date"]).dt.strftime("%Y-%m-%d")
    dates_path = directory / "sampling_dates.csv"
    dates_df.to_csv(dates_path, index=False)
    
    from epilink import PhylogenyError, build_phylogenetic_tree_from_fasta
    
    try:
        result = build_phylogenetic_tree_from_fasta(
            alignment_fasta=str(alignment_path),
            reference_fasta=str(reference_path),
            dates=str(dates_path),
            dated=True,
            output_dir=str(directory / "backend"),
            model=phylo_config.get("model", "MFP"),
            threads=phylo_config.get("threads", 1),
            seed=phylo_config.get("seed", 2026),
            clock_rate=phylo_config.get("clock_rate"),
            iqtree_executable=iqtree_tool["path"],
            timeout=phylo_config.get("timeout", 7200),
        )
    except PhylogenyError as exc:
        raise RuntimeError(f"IQ-TREE inference failed: {exc}")
    
    raw_tree = read_phylo(str(result.output_paths["raw_tree"]), "newick")
    reference_name = result.reference_name
    raw_pruned = _prune_reference_tip(raw_tree, reference_name)
    validate_tree(raw_pruned, cases.case_id)
    write_phylo(raw_pruned, directory / "raw.nwk", "newick", format_branch_length="%.12g")
    
    dated_tree = read_phylo(str(result.output_paths["dated_tree"]), "newick")
    validate_tree(dated_tree, cases.case_id)
    write_phylo(dated_tree, directory / "dated.nwk", "newick", format_branch_length="%.12g")
    
    result.node_dates.to_csv(directory / "node_dates.tsv", sep="\t", index=False)
    
    phylogeny_meta = {
        "model": phylo_config.get("model", "MFP"),
        "threads": phylo_config.get("threads", 1),
        "seed": phylo_config.get("seed", 2026),
        "clock_rate": result.clock_rate,
        "date_origin": str(result.date_origin) if result.date_origin else None,
        "reference_name": reference_name,
        "alignment_length": sequence_length,
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
        rooting="IQ-TREE reference outgroup, pruned (raw) / LSD2 clock (dated)",
        root_split=root_signature(raw_pruned),
        dated_root_split=root_signature(dated_tree),
    )
    
    return directory, sequence_length
