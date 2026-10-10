"""Content-addressed truth and observation artifacts; no scorer dependencies."""

from __future__ import annotations

import textwrap
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from epilink import (
    InfectiousnessToTransmission,
    build_pairwise_case_table,
    simulate_epidemic_dates,
    simulate_genomic_sequences,
)

from ..natural_history import natural_history
from ..provenance import (
    complete_artifact,
    digest_file,
    fingerprint,
    generation_signature,
    valid_artifact,
)
from ..schemas import validate_pairs
from ..truth import TreeIndex, relationship_table
from .scovmod import prepare_tree

_PACK_SHIFTS = np.arange(62, -2, -2, dtype=np.uint64)


def load_backbone(config):
    path = prepare_tree(config)
    tree = nx.read_gml(path)
    count = config["inputs"].get("smoke_cases")
    if count is not None:
        if count < 4:
            raise ValueError("Smoke backbone needs at least four cases")
        # Topological prefix keeps all ancestors; never prune internal paths.
        keep = list(nx.topological_sort(tree))[:count]
        tree = tree.subgraph(keep).copy()
    TreeIndex(tree)
    return tree


def prepare_truth(config, tree, implementation):
    signature = {
        "kind": "truth-v1",
        "source_sha256": digest_file(config["inputs"]["tree_path"]),
        "nodes": list(tree),
        "edges": list(tree.edges()),
        "implementation": implementation["evaluation"]["truth/relationships.py"],
    }
    directory = (
        Path(config["output_directory"])
        / "artifacts/truth"
        / fingerprint(signature)[:20]
    )
    if not valid_artifact(directory, signature):
        directory.mkdir(parents=True, exist_ok=True)
        frame = relationship_table(tree)
        frame.to_parquet(directory / "relationships.parquet", index=False)
        pd.DataFrame(
            {"node_index": np.arange(len(tree)), "case_id": list(tree)}
        ).to_parquet(directory / "nodes.parquet", index=False)
        complete_artifact(
            directory,
            signature,
            ["relationships.parquet", "nodes.parquet"],
            n_cases=len(tree),
            n_pairs=len(frame),
        )
    return directory


def _export_sampled_fasta(packed_data, sampled_node_ids, filepath):
    """Export only sampled cases to FASTA file.
    
    Parameters
    ----------
    packed_data : PackedGenomicData
        Packed genomic data container with node_to_idx mapping
    sampled_node_ids : list
        List of sampled case IDs (strings) matching packed_data node identifiers
    filepath : Path
        Output FASTA file path
    """
    base_lookup = np.array([packed_data.bases_map[idx] for idx in range(4)], dtype="<U1")
    
    with open(filepath, "w") as f:
        for node_id in sampled_node_ids:
            idx = packed_data.node_to_idx[node_id]
            blocks = packed_data.packed_u64[idx]
            L = packed_data.original_length
            unpacked = np.empty((len(blocks), 32), dtype=np.int8)
            
            for position, shift in enumerate(_PACK_SHIFTS):
                unpacked[:, position] = ((blocks >> shift) & np.uint64(3)).astype(np.int8)
            
            seq = unpacked.reshape(len(blocks) * 32)[:L]
            seq_str = "".join(base_lookup[seq])
            
            header = f">{node_id}"
            body = textwrap.fill(seq_str, width=100)
            f.write(f"{header}\n{body}\n")


def _export_reference(reference_sequence_string, filepath):
    """Export ancestral reference sequence to FASTA file.
    
    Parameters
    ----------
    reference_sequence_string : str
        A/C/G/T reference sequence string
    filepath : Path
        Output FASTA file path
    """
    with open(filepath, "w") as f:
        f.write(">ancestral_reference\n")
        f.write(textwrap.fill(reference_sequence_string, width=100) + "\n")


def _export_sampling_dates(cases, filepath):
    """Export sampling dates as TSV file.
    
    Parameters
    ----------
    cases : pd.DataFrame
        Cases DataFrame with case_id and sample_date columns
    filepath : Path
        Output TSV file path
    """
    cases[["case_id", "sample_date"]].to_csv(
        filepath, sep="\t", index=False, na_rep="NA"
    )


def prepare_observations(config, tree, truth_directory, seed, implementation):
    signature = {
        "kind": "observations-v3",
        "truth": Path(truth_directory).name,
        "generation": config["generation"],
        "simulation": config["simulation"],
        "seed": seed,
        "implementation": generation_signature(implementation),
    }
    directory = (
        Path(config["output_directory"])
        / "artifacts/observations"
        / fingerprint(signature)[:20]
    )
    if valid_artifact(directory, signature):
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    profile = InfectiousnessToTransmission(
        parameters=natural_history(config["generation"]), rng_seed=seed
    )
    populated = simulate_epidemic_dates(
        profile, tree, config["simulation"]["fraction_sampled"]
    )
    genomes = simulate_genomic_sequences(
        profile, populated, config["simulation"]["sequence_length"]
    )
    original = build_pairwise_case_table(genomes["packed"], populated)
    nodes = list(populated)
    full_lookup = {str(node): i for i, node in enumerate(nodes)}
    cases = pd.DataFrame(
        [
            {
                "case_id": str(node),
                "node_index": i,
                "sample_date": attributes["sample_date"],
                "exposure_date": attributes["exposure_date"],
            }
            for i, (node, attributes) in enumerate(populated.nodes(data=True))
            if attributes["sampled"]
        ]
    )
    if len(cases) < 2:
        raise ValueError("Simulation produced fewer than two sampled cases")
    selected = original.loc[original.BothSampled].copy()
    node_a = selected.CaseA.astype(str).map(full_lookup).to_numpy(np.int64)
    node_b = selected.CaseB.astype(str).map(full_lookup).to_numpy(np.int64)
    low, high = np.minimum(node_a, node_b), np.maximum(node_a, node_b)
    rank = {int(index): i for i, index in enumerate(cases.node_index)}
    observations = (
        pd.DataFrame(
            {
                "pair_id": low * (2 * len(tree) - low - 1) // 2 + high - low - 1,
                "a": [rank[int(i)] for i in low],
                "b": [rank[int(i)] for i in high],
                "TD": np.rint(
                    np.abs(selected.SamplingDateDistanceDays.to_numpy(float))
                ),
                "GD_deterministic": selected.DeterministicDistance.to_numpy(),
                "GD_stochastic": selected.StochasticDistance.to_numpy(),
            }
        )
        .sort_values(["a", "b"])
        .reset_index(drop=True)
    )
    validate_pairs(observations, cases)
    truth = pd.read_parquet(
        Path(truth_directory) / "relationships.parquet", columns=["pair_id", "M"]
    )
    truth = truth.set_index("pair_id").loc[observations.pair_id]
    original_labels = pd.Series(
        selected.IsRelated.to_numpy(bool),
        index=low * (2 * len(tree) - low - 1) // 2 + high - low - 1,
    ).loc[observations.pair_id]
    np.testing.assert_array_equal(truth.M.eq(0).fillna(False), original_labels)
    
    sampled_case_ids = cases.case_id.tolist()
    _export_sampled_fasta(
        genomes.packed.deterministic,
        sampled_case_ids,
        directory / "sampled_deterministic.fasta"
    )
    _export_sampled_fasta(
        genomes.packed.stochastic,
        sampled_case_ids,
        directory / "sampled_stochastic.fasta"
    )
    _export_reference(
        genomes.reference_sequence_string,
        directory / "reference.fasta"
    )
    _export_sampling_dates(cases, directory / "sampling_dates.tsv")
    
    observations.to_parquet(directory / "pairs.parquet", index=False)
    cases.to_parquet(directory / "cases.parquet", index=False)
    complete_artifact(
        directory,
        signature,
        [
            "pairs.parquet",
            "cases.parquet",
            "sampled_deterministic.fasta",
            "sampled_stochastic.fasta",
            "reference.fasta",
            "sampling_dates.tsv",
        ],
        truth_directory=str(truth_directory),
        n_cases=len(cases),
        n_pairs=len(observations),
        units={
            "GD": "Hamming substitutions",
            "TD": "absolute rounded sampling days",
            "mutation_count_genome_length": config["generation"]["genome_length"],
            "simulated_sequence_length": config["simulation"]["sequence_length"],
        },
    )
    return directory


def load_observations(directory):
    directory = Path(directory)
    observations = pd.read_parquet(directory / "pairs.parquet")
    cases = pd.read_parquet(directory / "cases.parquet")
    validate_pairs(observations, cases)
    return observations, cases


def load_tree_inputs(directory):
    """Return paths to FASTA, reference, and dates for tree inference.
    
    Parameters
    ----------
    directory : Path or str
        Observation artifact directory
        
    Returns
    -------
    dict
        Paths to sampled_deterministic.fasta, sampled_stochastic.fasta,
        reference.fasta, sampling_dates.tsv, plus validated case count
    """
    directory = Path(directory)
    required = [
        "sampled_deterministic.fasta",
        "sampled_stochastic.fasta",
        "reference.fasta",
        "sampling_dates.tsv",
        "cases.parquet",
    ]
    missing = [f for f in required if not (directory / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Observation bundle missing files: {missing}. "
            f"Expected v3 observation artifact at {directory}"
        )
    cases = pd.read_parquet(directory / "cases.parquet")
    return {
        "deterministic_fasta": directory / "sampled_deterministic.fasta",
        "stochastic_fasta": directory / "sampled_stochastic.fasta",
        "reference_fasta": directory / "reference.fasta",
        "sampling_dates": directory / "sampling_dates.tsv",
        "cases_parquet": directory / "cases.parquet",
        "n_cases": len(cases),
    }


def load_truth(directory, pair_ids):
    truth = pd.read_parquet(Path(directory) / "relationships.parquet")
    return truth.set_index("pair_id").loc[pair_ids].reset_index()


def analysis_table(observations, cases, truth, scores=None):
    """Convenient joined view; score columns remain explicit model identifiers."""
    result = observations.merge(truth, on="pair_id", validate="one_to_one")
    result["CaseID1"] = cases.case_id.to_numpy()[result.a]
    result["CaseID2"] = cases.case_id.to_numpy()[result.b]
    if scores is not None:
        result = result.merge(scores, on="pair_id", validate="one_to_one")
    return result
