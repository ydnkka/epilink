from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd

from ..model import InfectiousnessToTransmission
from .genome import PackedGenomicData


def simulate_epidemic_dates(
    transmission_profile: InfectiousnessToTransmission,
    tree: nx.DiGraph,
    fraction_sampled: float = 1.0,
) -> nx.DiGraph:
    """Populate a transmission tree with simulated epidemic dates.

    Parameters
    ----------
    transmission_profile : InfectiousnessToTransmission
        Transmission-time model used to sample latent, presymptomatic, and
        transmission intervals.
    tree : networkx.DiGraph
        Directed transmission tree.
    fraction_sampled : float, default=1.0
        Fraction of nodes marked as sampled.
        Shape parameter for the Gamma-distributed testing delay.

    Returns
    -------
    networkx.DiGraph
        Copy of ``tree`` with simulated epidemic annotations.
    """

    tree = tree.copy()
    rng = transmission_profile.rng

    num_nodes = tree.number_of_nodes()
    num_sampled = int(round(fraction_sampled * num_nodes))
    sampled_node_ids = set(rng.choice(list(tree.nodes()), size=num_sampled, replace=False))
    nx.set_node_attributes(tree, {node: (node in sampled_node_ids) for node in tree}, "sampled")

    def sample_stage_intervals() -> tuple[float, float, float]:
        latent = transmission_profile.sample_latent_periods().item()
        presymptomatic = transmission_profile.sample_presymptomatic_periods().item()
        testing = transmission_profile.sample_testing_delays().item()

        return latent, presymptomatic, testing

    roots = [n for n, d in tree.in_degree() if d == 0]

    for root in roots:
        exposure_date = int(rng.choice(range(30)))
        latent_period, presymptomatic_period, testing_delay = sample_stage_intervals()

        tree.nodes[root].update(
            {
                "exposure_date": exposure_date,
                "date_infectious": exposure_date + latent_period,
                "date_symptom_onset": (exposure_date + latent_period + presymptomatic_period),
                "sample_date": (
                    exposure_date + latent_period + presymptomatic_period + testing_delay
                ),
                "seed": True,
            }
        )

        for parent, child in nx.dfs_edges(tree, source=root):
            transmission_time = transmission_profile.rvs().item()
            latent_period, presymptomatic_period, testing_delay = sample_stage_intervals()
            parent_infectious_date = tree.nodes[parent]["date_infectious"]

            child_exposure_date = parent_infectious_date + transmission_time
            child_infectious_date = child_exposure_date + latent_period
            child_symptom_onset_date = child_infectious_date + presymptomatic_period
            child_sample_date = child_symptom_onset_date + testing_delay

            tree.nodes[child].update(
                {
                    "exposure_date": child_exposure_date,
                    "date_infectious": child_infectious_date,
                    "date_symptom_onset": child_symptom_onset_date,
                    "sample_date": child_sample_date,
                    "seed": False,
                }
            )

    return tree


def simulate_genomic_sequences(
    transmission_profile: InfectiousnessToTransmission,
    tree: nx.DiGraph,
    genome_length: int = 1000,
    return_raw: bool = False,
) -> dict[str, Any]:
    """Simulate genomic evolution along a transmission tree.

    Parameters
    ----------
    transmission_profile : InfectiousnessToTransmission
        Transmission-time model used to sample latent, presymptomatic, and
        transmission intervals.
    tree : networkx.DiGraph
        Directed transmission tree with epidemic-date annotations.
    genome_length : int, default=1000
        Length of the simulated genome sequences.
    return_raw : bool, default=False
        If ``True``, include the unpacked integer sequence matrices in the
        returned dictionary.

    Returns
    -------
    dict
        Dictionary with packed genomic data and, optionally, raw sequences.
    """

    base_lookup = {0: "A", 1: "C", 2: "G", 3: "T"}
    num_bases = len(base_lookup)
    nodes = list(tree.nodes())
    node_to_index = {node: index for index, node in enumerate(nodes)}
    num_nodes = len(nodes)

    rng = transmission_profile.rng

    linear_sequences = np.zeros((num_nodes, genome_length), dtype=np.int8)
    poisson_sequences = np.zeros((num_nodes, genome_length), dtype=np.int8)
    reference_sequence = rng.integers(0, num_bases, size=genome_length, dtype=np.int8)

    def mutate_sequence(
        sequence: npt.NDArray[np.int8],
        num_mutations: int,
    ) -> npt.NDArray[np.int8]:
        num_mutations = min(int(num_mutations), genome_length)
        if num_mutations <= 0:
            return sequence.copy()

        mutated_sequence = sequence.copy()
        mutation_sites = np.asarray(
            rng.choice(genome_length, size=num_mutations, replace=False),
            dtype=np.intp,
        )
        mutation_offsets = rng.integers(1, num_bases, size=num_mutations, dtype=np.int8)
        mutated_sequence[mutation_sites] = (
            mutated_sequence[mutation_sites] + mutation_offsets
        ) % num_bases
        return mutated_sequence

    roots = [node for node, degree in tree.in_degree(tree.nodes) if degree == 0]
    for root in roots:
        root_index = node_to_index[root]
        max_root_drift = min(35, genome_length + 1)
        root_drift_mutations = int(rng.integers(1, max_root_drift))
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

            expected_mutations = transmission_profile.expected_mutations(branch_length_days)

            num_linear_mutations = int(round(expected_mutations.item()))
            linear_sequences[child_index] = mutate_sequence(
                linear_sequences[parent_index],
                num_linear_mutations,
            )

            num_poisson_mutations = int(rng.poisson(expected_mutations))
            poisson_sequences[child_index] = mutate_sequence(
                poisson_sequences[parent_index],
                num_poisson_mutations,
            )

    packed_sequences = {
        "linear": PackedGenomicData(
            linear_sequences,
            genome_length,
            node_to_index,
            base_lookup,
        ),
        "poisson": PackedGenomicData(
            poisson_sequences,
            genome_length,
            node_to_index,
            base_lookup,
        ),
    }
    raw_sequences = {"linear": linear_sequences, "poisson": poisson_sequences}

    return {"packed": packed_sequences, "raw": raw_sequences if return_raw else None}


def build_pairwise_case_table(
    packed_genomic_data: Mapping[str, PackedGenomicData], tree: nx.DiGraph
) -> pd.DataFrame:
    """Generate a long-format pairwise distance table.

    Parameters
    ----------
    packed_genomic_data : mapping
        Mapping containing packed genomic data for the ``"linear"`` and
        ``"poisson"`` simulation outputs.
    tree : networkx.DiGraph
        Directed transmission tree with epidemic annotations.

    Returns
    -------
    pandas.DataFrame
        Pairwise table of genetic distances, temporal distances, and simple
        topological relationships.
    """

    packed_linear = packed_genomic_data["linear"]
    packed_poisson = packed_genomic_data["poisson"]
    node_map = packed_linear.node_to_idx
    n_nodes = packed_linear.n_seqs
    idx_to_node = {v: k for k, v in node_map.items()}

    mat_linear = packed_linear.compute_hamming_distances()
    mat_poisson = packed_poisson.compute_hamming_distances()

    sample_dates = np.zeros(n_nodes)
    for node, idx in node_map.items():
        sample_dates[idx] = tree.nodes[node].get("sample_date", np.nan)

    diff_matrix = sample_dates[:, np.newaxis] - sample_dates
    mat_temporal = np.abs(diff_matrix).round()

    mat_related = np.zeros((n_nodes, n_nodes), dtype=bool)

    for u, v in tree.edges():
        if u in node_map and v in node_map:
            i, j = node_map[u], node_map[v]
            mat_related[i, j] = True
            mat_related[j, i] = True

    for node in tree.nodes():
        children = list(tree.successors(node))
        if len(children) > 1:
            child_indices = [node_map[c] for c in children if c in node_map]
            if child_indices:
                grid_x, grid_y = np.meshgrid(child_indices, child_indices)
                mat_related[grid_x, grid_y] = True

    rows, cols = np.triu_indices(n_nodes, k=1)
    id_array = np.array([idx_to_node[i] for i in range(n_nodes)])
    sampled_status = np.array([tree.nodes[node].get("sampled", False) for node in id_array])

    return pd.DataFrame(
        {
            "CaseA": id_array[rows],
            "CaseB": id_array[cols],
            "IsRelated": mat_related[rows, cols],
            "BothSampled": sampled_status[rows] & sampled_status[cols],
            "DeterministicDistance": mat_linear[rows, cols],
            "StochasticDistance": mat_poisson[rows, cols],
            "SamplingDateDistanceDays": mat_temporal[rows, cols],
        }
    )


__all__ = ["simulate_genomic_sequences", "build_pairwise_case_table", "simulate_epidemic_dates"]
