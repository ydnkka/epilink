"""
Simulate epidemic dates and genomic sequences along a transmission tree.

Provides utilities to populate epidemic dates on a transmission tree, simulate
genomic evolution along branches, and generate pairwise genetic and temporal
distance tables.

Classes
-------
SequencePacker64
    64-bit sequence packer and Hamming distance utilities.
PackedGenomicData
    Container for packed sequences and metadata.

Functions
---------
populate_epidemic_data
simulate_genomic_data
generate_pairwise_data
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import njit, prange
from scipy.stats import gamma, poisson

from .infectiousness_profile import TOIT, MolecularClock

NDArrayInt8 = npt.NDArray[np.int8]
NDArrayUInt64 = npt.NDArray[np.uint64]
NDArrayFloat32 = npt.NDArray[np.float32]


def populate_epidemic_data(
    toit: TOIT,
    tree: nx.DiGraph,
    prop_sampled: float = 1.0,
    sampling_scale: float = 1.0,
    sampling_shape: float = 3.0,
    root_start_range: int = 30,
) -> nx.DiGraph:
    """
    Populate a transmission tree with simulated epidemic dates and sampling status.

    Parameters
    ----------
    toit : TOIT
        Infectiousness profile used to sample generation intervals and stage durations.
    tree : networkx.DiGraph
        Directed transmission tree (roots -> infections).
    prop_sampled : float, default=1.0
        Proportion of nodes marked as sampled.
    sampling_scale : float, default=1.0
        Scale parameter for the Gamma distribution of sampling delay.
        If <= 0, use deterministic mean values.
    sampling_shape : float, default=3.0
        Shape parameter for the Gamma distribution of sampling delay.
    root_start_range : int, default=30
        Upper bound (exclusive) for the random start date of the index case.

    Returns
    -------
    graph : networkx.DiGraph
        Copy of the input graph with added node attributes.
    """
    # Create a shallow copy to avoid mutating the original object unexpectedly
    G = tree.copy()
    rng = toit.rng  # Use the random state from the TOIT object for consistency

    # --- 1. Assign Sampling Status ---
    # We select all sampled nodes at once for efficiency
    n_nodes = G.number_of_nodes()
    n_sampled = int(round(prop_sampled * n_nodes))

    # Efficiently choose nodes without replacement
    sampled_node_ids = set(rng.choice(list(G.nodes()), size=n_sampled, replace=False))

    # Batch update sampling attribute
    nx.set_node_attributes(G, {n: (True if n in sampled_node_ids else False) for n in G}, "sampled")

    # --- 2. Helper for Interval Sampling ---
    def get_intervals() -> tuple[float, float, float]:
        """Return latent period, presymptomatic period, and sampling delay."""
        if sampling_scale <= 0:
            # Deterministic mode (useful for testing/control)
            # Expected value of a gamma distribution
            yE = toit.params.latent_shape * toit.params.incubation_scale
            yP = toit.params.presymptomatic_shape * toit.params.incubation_scale
            test_delay = 0
        else:
            # Stochastic mode
            yE = toit.sample_latent().item()
            yP = toit.sample_presymptomatic().item()
            # Gamma distribution for delay between symptom onset and test
            test_delay = gamma.rvs(a=sampling_shape, scale=sampling_scale, random_state=rng)

        return yE, yP, test_delay

    # --- 3. Traverse and Assign Dates ---
    # Identify root(s)
    roots = [n for n, d in G.in_degree(G.nodes) if d == 0]

    for root in roots:
        # A. Initialize Root Node
        # Roots are seeded at a random time within the start range
        exp_date = int(rng.choice(range(root_start_range))) if root_start_range > 0 else 0
        latent_period, pre_sym_inf, sym_test = get_intervals()

        G.nodes[root].update(
            {
                "exposure_date": exp_date,
                "date_infectious": exp_date + latent_period,
                "date_symptom_onset": exp_date + latent_period + pre_sym_inf,
                "sample_date": exp_date + latent_period + pre_sym_inf + sym_test,
                "seed": True,
            }
        )

        # B. Propagate to Successors (DFS Traversal)
        # Using dfs_edges ensures we always process a parent before their children
        for parent, child in nx.dfs_edges(G, source=root):
            # Sample Generation Interval (TOIT)
            # This is the time from Parent becoming infectious to Child being exposed
            toit_value = 0 if sampling_scale <= 0 else toit.rvs().item()

            # Sample biological intervals for the child
            latent_period, pre_sym_inf, sym_test = get_intervals()

            # Retrieve Parent's infectious date
            parent_inf_date = G.nodes[parent]["date_infectious"]

            # Calculate Child's dates
            child_exp_date = parent_inf_date + toit_value
            child_inf_date = child_exp_date + latent_period
            child_sym_date = child_inf_date + pre_sym_inf
            child_sample_date = child_sym_date + sym_test

            # Update Child Node
            G.nodes[child].update(
                {
                    "exposure_date": child_exp_date,
                    "date_infectious": child_inf_date,
                    "date_symptom_onset": child_sym_date,
                    "sample_date": child_sample_date,
                    "seed": False,
                }
            )

    return G


class SequencePacker64:
    """
    Utility engine for 64-bit packing and Hamming distance calculation.

    Notes
    -----
    Uses a 2-bit encoding mapping: A=0, C=1, G=2, T=3.
    """

    # 2-bit encoding mapping for reference
    # A=0, C=1, G=2, T=3

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def pack_u64(arr: NDArrayInt8) -> NDArrayUInt64:
        """
        Pack 2-bit nucleotide arrays into 64-bit blocks.

        Parameters
        ----------
        arr : numpy.ndarray
            Array of shape (N, L) with values in {0, 1, 2, 3}.

        Returns
        -------
        packed : numpy.ndarray
            Packed array of shape (N, ceil(L / 32)) with dtype uint64.
        """
        N, L = arr.shape
        B = (L + 31) // 32  # number of 64-bit blocks needed
        out = np.zeros((N, B), dtype=np.uint64)

        for i in prange(N):
            for b in range(B):
                start = b * 32
                end = min(start + 32, L)
                word = np.uint64(0)

                # First nucleotide occupies the most significant bits (left-aligned)
                shift = 62

                for k in range(start, end):
                    # Cast to uint64 to keep the bitwise ops in unsigned space.
                    word |= np.uint64(np.uint64(arr[i, k]) << shift)
                    shift -= 2

                out[i, b] = word

        return out

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def hamming64(packed: NDArrayUInt64) -> NDArrayFloat32:
        """
        Compute pairwise Hamming distances for packed sequences.

        Parameters
        ----------
        packed : numpy.ndarray
            Packed array of shape (N, B) with dtype uint64.

        Returns
        -------
        distances : numpy.ndarray
            Pairwise Hamming distance matrix of shape (N, N).
        """
        # Bitmasks for SWAR Hamming distance
        M55 = np.uint64(0x5555555555555555)  # 0101...
        M33 = np.uint64(0x3333333333333333)  # 0011...
        M0F = np.uint64(0x0F0F0F0F0F0F0F0F)  # 00001111...
        M01 = np.uint64(0x0101010101010101)  # accumulator

        N, B = packed.shape
        d = np.zeros((N, N), dtype=np.float32)

        for i in prange(N):
            for j in range(i + 1, N):
                total = 0
                for k in range(B):
                    # Step 1: XOR identifies differences
                    x = packed[i, k] ^ packed[j, k]

                    # Step 2: Vertical collapse (2-bit diff -> 1-bit count)
                    # If bits are 00 or 11 -> 0. If 01 or 10 -> 1.
                    # This maps nucleotide diffs to binary 1s.
                    diff = (x & M55) | ((x >> 1) & M55)

                    # Step 3: Standard 64-bit Population Count (SWAR)
                    c = diff
                    c = (c & M33) + ((c >> 2) & M33)
                    c = (c & M0F) + ((c >> 4) & M0F)
                    # Multiplication acts as a parallel adder for bytes
                    c = (c * M01) >> 56

                    total += int(c)

                d[i, j] = total
                d[j, i] = total

        return d


def _ensure_pyfunc(func: Any) -> None:
    """Ensure a .py_func attribute for non-numba callables (test compatibility)."""
    if not hasattr(func, "py_func"):
        func.py_func = func


_ensure_pyfunc(SequencePacker64.pack_u64)
_ensure_pyfunc(SequencePacker64.hamming64)


class PackedGenomicData:
    """
    Container for packed genomic sequences and lookup metadata.

    Parameters
    ----------
    int8_matrix : numpy.ndarray
        Array of shape (N, L) with values in {0, 1, 2, 3}.
    original_length : int
        Original length of the sequences (L).
    node_map : dict[str, int]
        Mapping from node names to row indices.
    base_map : dict[int, str]
        Mapping from integer codes to base characters (e.g., 0 -> "A").

    Attributes
    ----------
    packed_u64 : numpy.ndarray
        Packed array of shape (N, B) with dtype uint64.
    original_length : int
        Original length of the sequences (L).
    n_seqs : int
        Number of sequences (N).
    node_to_idx : dict[str, int]
        Mapping from node names to row indices.
    idx_to_node : dict[int, str]
        Mapping from row indices to node names.
    bases_map : dict[int, str]
        Mapping from integer codes to base characters.
    """

    def __init__(
        self,
        int8_matrix: NDArrayInt8,
        original_length: int,
        node_map: dict[str, int],
        base_map: dict[int, str],
    ):
        """Initialize packed genomic data and metadata."""
        self.n_seqs: int
        self.original_length: int
        self.node_to_idx: dict[str, int]
        self.idx_to_node: dict[int, str]
        self.bases_map: dict[int, str]
        self.packed_u64: NDArrayUInt64

        self.n_seqs, L = int8_matrix.shape
        self.original_length = original_length
        self.node_to_idx = node_map
        self.idx_to_node = {v: k for k, v in node_map.items()}
        self.bases_map = base_map
        self.packed_u64 = SequencePacker64.pack_u64(int8_matrix)

    def compute_hamming_distances(self) -> NDArrayFloat32:
        """
        Compute the pairwise Hamming distance matrix.

        Returns
        -------
        distances : numpy.ndarray
            Distance matrix of shape (N, N).
        """
        print(f"Computing 64-bit Hamming distances for {self.n_seqs} sequences...")
        return SequencePacker64.hamming64(self.packed_u64)

    def write_fasta(self, filepath: str) -> None:
        """
        Write unpacked sequences to a FASTA file.

        Parameters
        ----------
        filepath : str
            Output FASTA path.

        Returns
        -------
        None
        """
        print(f"Exporting FASTA to {filepath}...")

        with open(filepath, "w") as f:
            for i in range(self.n_seqs):
                blocks = self.packed_u64[i]
                L = self.original_length

                # Temporary buffer for unpacking
                unpacked = np.zeros(len(blocks) * 32, dtype=np.int8)

                idx = 0
                for w in blocks:
                    # Extract 32 nucleotides from the 64-bit word
                    for shift in range(62, -1, -2):  # 62, 60, 58, ..., 2, 0
                        unpacked[idx] = (w >> shift) & 3
                        idx += 1

                # Trim padding and convert to string
                seq = unpacked[:L]
                seq_str = "".join(self.bases_map[int(b)] for b in seq)

                # Write header and wrapped sequence
                header = f">{self.idx_to_node[i]}"
                body = textwrap.fill(seq_str, width=100)
                f.write(f"{header}\n{body}\n")


def simulate_genomic_data(
    clock: MolecularClock, tree: nx.DiGraph, return_raw: bool = False
) -> dict[str, Any]:
    """
    Simulate genomic evolution along a transmission tree.

    Parameters
    ----------
    clock : MolecularClock
        Molecular clock model for sampling substitution rates.
    tree : networkx.DiGraph
        Transmission tree with epidemic dates on nodes.
    return_raw : bool, default=False
        If True, include raw int8 mutation matrices in the output.

    Returns
    -------
    output : GenomicSimulationOutput
        Dictionary with "packed" mapping to PackedGenomicData objects ("linear"
        and "poisson") and "raw" mapping to int8 arrays when return_raw is True,
        otherwise None.
    """
    BASES = np.array([0, 1, 2, 3], dtype=np.int8)
    BASES_MAP = {0: "A", 1: "C", 2: "G", 3: "T"}
    nodes = list(tree.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n_nodes = len(nodes)

    # Use int8 for active simulation (easier to mutate)
    linear_mat = np.zeros((n_nodes, clock.gen_len), dtype=np.int8)
    poisson_mat = np.zeros((n_nodes, clock.gen_len), dtype=np.int8)

    # Generate Reference
    ref_seq = clock.rng.choice(BASES, size=clock.gen_len)

    def mutate(seq: NDArrayInt8, n_mut: int) -> NDArrayInt8:
        if n_mut <= 0:
            return seq.copy()
        new_seq = seq.copy()
        pos_indices = np.array(clock.rng.choice(clock.gen_len, size=n_mut, replace=False))
        for pos in pos_indices:
            current = new_seq[pos]
            # Choose any base except current
            new_seq[pos] = clock.rng.choice(BASES[BASES != current])
        return new_seq

    # Initialize Roots (Random drift from reference)
    roots = [n for n, d in tree.in_degree(tree.nodes) if d == 0]
    for root in roots:
        idx = node_to_idx[root]
        root_drift = int(clock.rng.choice(range(1, 35)))
        root_seq = mutate(ref_seq, root_drift)
        linear_mat[idx] = root_seq
        poisson_mat[idx] = root_seq

    # Traverse Tree
    print("Simulating mutations along transmission tree...")
    for root in roots:
        for parent, child in nx.dfs_edges(tree, source=root):
            par_idx = node_to_idx[parent]
            chi_idx = node_to_idx[child]

            # 1. Calculate Time Duration
            # branch_length = |Sample_Par - Trans| + |Sample_Chi - Trans|
            try:
                t_trans = tree.nodes[child]["exposure_date"]
                t_samp_par = tree.nodes[parent]["sample_date"]
                t_samp_chi = tree.nodes[child]["sample_date"]
            except KeyError:
                raise ValueError("Missing dates in tree. Run epidemic simulation first.")

            branch_length = abs(t_samp_par - t_trans) + abs(t_samp_chi - t_trans)

            # 2. Get Clock Rate from TOIT (Strict or Relaxed)
            rate_val = clock.sample_clock_rate_per_day().item()
            genetic_dist = rate_val * branch_length

            # 3. Mutate (Linear)
            n_lin = int(round(genetic_dist))
            linear_mat[chi_idx] = mutate(linear_mat[par_idx], n_lin)

            # 4. Mutate (Poisson)
            n_poi = int(poisson.rvs(genetic_dist, random_state=clock.rng))
            poisson_mat[chi_idx] = mutate(poisson_mat[par_idx], n_poi)

    print("Packing data into 2-bit format...")
    packed = {
        "linear": PackedGenomicData(linear_mat, clock.gen_len, node_to_idx, BASES_MAP),
        "poisson": PackedGenomicData(poisson_mat, clock.gen_len, node_to_idx, BASES_MAP),
    }
    raw = {"linear": linear_mat, "poisson": poisson_mat}

    out = {"packed": packed, "raw": raw if return_raw else None}

    return out


def generate_pairwise_data(
    packed_genomic_data: Mapping[str, PackedGenomicData], tree: nx.DiGraph
) -> pd.DataFrame:
    """
    Generate a long-format DataFrame with genetic and temporal distances.

    Parameters
    ----------
    packed_genomic_data : Mapping[str, PackedGenomicData]
        Packed genomic data containing "linear" and "poisson" entries.
    tree : networkx.DiGraph
        Transmission tree with "sample_date" node attributes.

    Returns
    -------
    df : pandas.DataFrame
        Columns: ["NodeA", "NodeB", "Related", "Sampled", "LinearDist",
        "PoissonDist", "TemporalDist"].
    """
    # 1. Retrieve Data & Map
    # We assume both linear/poisson share the same node map (they should)
    packed_linear = packed_genomic_data["linear"]
    packed_poisson = packed_genomic_data["poisson"]
    node_map = packed_linear.node_to_idx
    n_nodes = packed_linear.n_seqs

    # Invert map for labeling
    idx_to_node = {v: k for k, v in node_map.items()}

    print("Computing genetic distances...")
    mat_linear = packed_linear.compute_hamming_distances()
    mat_poisson = packed_poisson.compute_hamming_distances()

    # 2. Compute Temporal Distances (Vectorized)
    print("Computing temporal distances...")
    # Create an array of sample dates in the order of the matrix indices
    # We use a default of NaN or 0 if missing, but they should exist.
    sample_dates = np.zeros(n_nodes)
    for node, idx in node_map.items():
        sample_dates[idx] = tree.nodes[node].get("sample_date", np.nan)

    # Calculate absolute difference |Date_A - Date_B|
    # Broadcasting: (N, 1) - (1, N) creates (N, N) matrix
    diff_matrix = sample_dates[:, np.newaxis] - sample_dates
    mat_temporal = np.abs(diff_matrix).round()

    # 3. Determine 'Related' Status (Topology)
    print("Mapping topological relationships...")
    mat_related = np.zeros((n_nodes, n_nodes), dtype=bool)

    # A. Direct Links
    for u, v in tree.edges():
        if u in node_map and v in node_map:
            i, j = node_map[u], node_map[v]
            mat_related[i, j] = True
            mat_related[j, i] = True

    # B. Siblings
    for node in tree.nodes():
        children = list(tree.successors(node))
        if len(children) > 1:
            child_indices = [node_map[c] for c in children if c in node_map]
            if child_indices:
                grid_x, grid_y = np.meshgrid(child_indices, child_indices)
                mat_related[grid_x, grid_y] = True

    # 4. Extract Upper Triangle (Unique Pairs)
    print("Constructing DataFrame...")
    rows, cols = np.triu_indices(n_nodes, k=1)

    # Create ID lookup array
    # Ensure the order matches indices 0..N-1
    id_array = np.array([idx_to_node[i] for i in range(n_nodes)])
    sampled_status = np.array([tree.nodes[node].get("sampled", False) for node in id_array])

    df = pd.DataFrame(
        {
            "NodeA": id_array[rows],
            "NodeB": id_array[cols],
            "Related": mat_related[rows, cols],
            "Sampled": sampled_status[rows] & sampled_status[cols],
            "LinearDist": mat_linear[rows, cols],
            "PoissonDist": mat_poisson[rows, cols],
            "TemporalDist": mat_temporal[rows, cols],
        }
    )

    return df
