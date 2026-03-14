"""Genomic simulation utilities."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.stats import poisson

from ..model.clock import MolecularClock
from ..sequence.packing import PackedGenomicData


def simulate_genomic_sequences(
    clock: MolecularClock,
    tree: nx.DiGraph,
    return_raw: bool = False,
) -> dict[str, Any]:
    """Simulate genomic evolution along a transmission tree.

    Parameters
    ----------
    clock : MolecularClock
        Molecular clock used to sample mutation rates.
    tree : networkx.DiGraph
        Directed transmission tree with epidemic-date annotations.
    return_raw : bool, default=False
        If ``True``, include the unpacked integer sequence matrices in the
        returned dictionary.

    Returns
    -------
    dict
        Dictionary with packed genomic data and, optionally, raw sequences.
    """

    bases = np.array([0, 1, 2, 3], dtype=np.int8)
    base_lookup = {0: "A", 1: "C", 2: "G", 3: "T"}
    nodes = list(tree.nodes())
    node_to_index = {node: index for index, node in enumerate(nodes)}
    num_nodes = len(nodes)

    linear_sequences = np.zeros((num_nodes, clock.genome_length), dtype=np.int8)
    poisson_sequences = np.zeros((num_nodes, clock.genome_length), dtype=np.int8)
    reference_sequence = clock.rng.choice(bases, size=clock.genome_length)

    def mutate_sequence(
        sequence: npt.NDArray[np.int8],
        num_mutations: int,
    ) -> npt.NDArray[np.int8]:
        num_mutations = min(int(num_mutations), clock.genome_length)
        if num_mutations <= 0:
            return sequence.copy()

        mutated_sequence = sequence.copy()
        mutation_sites = np.array(
            clock.rng.choice(clock.genome_length, size=num_mutations, replace=False)
        )
        for site_index in mutation_sites:
            current_base = mutated_sequence[site_index]
            mutated_sequence[site_index] = clock.rng.choice(bases[bases != current_base])
        return mutated_sequence

    roots = [node for node, degree in tree.in_degree(tree.nodes) if degree == 0]
    for root in roots:
        root_index = node_to_index[root]
        max_root_drift = min(35, clock.genome_length + 1)
        root_drift_mutations = int(clock.rng.integers(1, max_root_drift))
        root_sequence = mutate_sequence(reference_sequence, root_drift_mutations)
        linear_sequences[root_index] = root_sequence
        poisson_sequences[root_index] = root_sequence

    for root in roots:
        for parent, child in nx.dfs_edges(tree, source=root):
            parent_index = node_to_index[parent]
            child_index = node_to_index[child]

            try:
                transmission_date = tree.nodes[child]["exposure_date"]
                parent_sample_date = tree.nodes[parent]["sample_date"]
                child_sample_date = tree.nodes[child]["sample_date"]
            except KeyError as error:
                raise ValueError("Missing dates in tree. Run epidemic simulation first.") from error

            branch_length_days = abs(parent_sample_date - transmission_date) + abs(
                child_sample_date - transmission_date
            )
            substitution_rate_per_day = clock.sample_substitution_rate_per_day().item()
            expected_mutations = substitution_rate_per_day * branch_length_days

            num_linear_mutations = int(round(expected_mutations))
            linear_sequences[child_index] = mutate_sequence(
                linear_sequences[parent_index],
                num_linear_mutations,
            )

            num_poisson_mutations = int(poisson.rvs(expected_mutations, random_state=clock.rng))
            poisson_sequences[child_index] = mutate_sequence(
                poisson_sequences[parent_index],
                num_poisson_mutations,
            )

    packed_sequences = {
        "linear": PackedGenomicData(
            linear_sequences,
            clock.genome_length,
            node_to_index,
            base_lookup,
        ),
        "poisson": PackedGenomicData(
            poisson_sequences,
            clock.genome_length,
            node_to_index,
            base_lookup,
        ),
    }
    raw_sequences = {"linear": linear_sequences, "poisson": poisson_sequences}

    return {"packed": packed_sequences, "raw": raw_sequences if return_raw else None}


__all__ = ["simulate_genomic_sequences"]
