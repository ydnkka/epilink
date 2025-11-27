"""
Simulate epidemic dates and genomic sequences along a transmission tree.

This script exposes three main functions:
- populate_epidemic_data(tree, toit, ...)
- simulate_genomic_data(tree, gen_length, toit, ...)
- generate_pairwise_data(packed_genomic_data, tree)

"""

from __future__ import annotations

import textwrap
from typing import Dict

import networkx as nx
import numpy as np
from scipy.stats import gamma, poisson
from tqdm.auto import tqdm

from numba import njit, prange

from epilink.infectiousness_profile import TOIT


def populate_epidemic_data(
    tree: nx.DiGraph,
    toit: TOIT,
    prop_sampled: float = 1.0,
    sampling_delay: float = 2.0,
    sampling_shape: float = 3.0,
    root_start_range: int = 30,
) -> nx.DiGraph:
    """
    Populates a transmission tree with simulated epidemic dates and sampling status.

    Args:
        tree: The directed transmission tree (roots -> infections).
        toit: Instance of the TOIT class for generation interval sampling.
        prop_sampled: proportion (0-1) nodes 'sampled' (observed).
        sampling_delay: Scale parameter for the Gamma distribution of sampling delay.
                        If <= 0, uses deterministic mean values.
        sampling_shape: Shape parameter for the Gamma distribution of sampling delay.
        root_start_range: Upper bound for the random start date of the index case.

    Returns:
        The modified NetworkX graph with added node attributes.
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
    nx.set_node_attributes(G, {n: (1 if n in sampled_node_ids else 0) for n in G}, "sampled")

    # --- 2. Helper for Interval Sampling ---
    def get_intervals():
        """Returns (Latent Period, Pre-symptomatic Period, Sampling Delay)"""
        if sampling_delay <= 0:
            # Deterministic mode (useful for testing/control)
            # Expected value of a gamma distribution
            yE = toit.params.k_E * toit.params.scale_inc
            yP = toit.params.k_P * toit.params.scale_inc
            test_delay = 0
        else:
            # Stochastic mode
            yE = toit.sample_E().item()
            yP = toit.sample_P().item()
            # Gamma distribution for delay between symptom onset and test
            test_delay = gamma.rvs(a=sampling_shape, scale=sampling_delay, random_state=rng)

        return yE, yP, test_delay

    # --- 3. Traverse and Assign Dates ---
    # Identify root(s)
    roots = [n for n, d in G.in_degree() if d == 0]

    for root in roots:
        # A. Initialize Root Node
        # Roots are seeded at a random time within the start range
        exp_date = int(rng.choice(range(root_start_range))) if root_start_range > 0 else 0
        latent_period, pre_sym_inf, sym_test = get_intervals()

        G.nodes[root].update({
            "exposure_date": exp_date,
            "date_infectious": exp_date + latent_period,
            "date_symptom_onset": exp_date + latent_period + pre_sym_inf,
            "sample_date": exp_date + latent_period + pre_sym_inf + sym_test,
            "seed": True,
        })

        # B. Propagate to Successors (DFS Traversal)
        # Using dfs_edges ensures we always process a parent before their children
        for parent, child in tqdm(nx.dfs_edges(G, source=root), total=G.number_of_edges(), desc="Simulating Epidemic Data"):
            # Sample Generation Interval (TOIT)
            # This is the time from Parent becoming infectious to Child being exposed
            toit_value = 0 if sampling_delay <= 0 else toit.rvs().item()

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
            G.nodes[child].update({
                "exposure_date": child_exp_date,
                "date_infectious": child_inf_date,
                "date_symptom_onset": child_sym_date,
                "sample_date": child_sample_date,
                "seed": False,
            })

    return G


class SequencePacker64:
    """
    Utility engine providing 64-bit packing for 2-bit nucleotide sequences
    and a fast Hamming distance calculator using SWAR (SIMD Within A Register).
    """

    # 2-bit encoding mapping for reference
    # A=0, C=1, G=2, T=3

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def pack_u64(arr: np.ndarray) -> np.ndarray:
        """
        Packs an (N x L) array of 2-bit integers (0,1,2,3) into 64-bit blocks.
        Each block encodes 32 nucleotides (32 * 2 = 64 bits).
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
                    word |= np.uint64(arr[i, k]) << shift
                    shift -= 2

                out[i, b] = word

        return out

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def hamming64(packed: np.ndarray) -> np.ndarray:
        """
        Computes pairwise Hamming distances on 64-bit packed sequences.
        Uses SWAR logic to count bit differences in parallel chunks.
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


class PackedGenomicData:
    """
    High-level container for genomic sequences.

    Attributes:
        packed_u64 (np.ndarray): The N x B uint64 array of packed sequences.
        original_length (int): Original length of the sequences (L).
    """

    def __init__(
        self,
        int8_matrix: np.ndarray,
        original_length: int,
        node_map: Dict,
        base_map: Dict,
    ):
        """
        Args:
            int8_matrix: Numpy array (N, L) of integers (0-3).
            original_length: Integer L.
            node_map: Dictionary mapping Node Names -> Index.
            base_map: Dictionary mapping Integer -> Character (e.g. 0->'A').
        """
        self.n_seqs, L = int8_matrix.shape
        self.original_length = original_length
        self.node_to_idx = node_map
        self.idx_to_node = {v: k for k, v in node_map.items()}
        self.BASES_MAP = base_map
        self.packed_u64 = SequencePacker64.pack_u64(int8_matrix)

    def compute_hamming_distances(self):
        """Calculates the N x N distance matrix using the packed representation."""
        print(f"Computing 64-bit Hamming distances for {self.n_seqs} sequences...")
        return SequencePacker64.hamming64(self.packed_u64)

    def write_fasta(self, filepath):
        """
        Unpack directly from 64-bit blocks and write to FASTA file.
        Useful for verifying integrity or exporting data.
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
                    shift = np.uint64(62)
                    # Extract 32 nucleotides from the 64-bit word
                    for _ in range(32):
                        unpacked[idx] = (w >> shift) & 3
                        idx += 1
                        shift -= 2

                # Trim padding and convert to string
                seq = unpacked[:L]
                seq_str = "".join(self.BASES_MAP[int(b)] for b in seq)

                # Write header and wrapped sequence
                header = f">{self.idx_to_node[i]}"
                body = textwrap.fill(seq_str, width=100)
                f.write(f"{header}\n{body}\n")


def simulate_genomic_data(
    tree: nx.DiGraph,
        gen_length: int,
        toit: TOIT,
        return_raw: bool = False
) -> Dict:
    """
    Simulates genomic evolution and returns efficient PackedGenomicData objects.
    """
    BASES = np.array([0, 1, 2, 3], dtype=np.int8)
    BASES_MAP = {0: "A", 1: "C", 2: "G", 3: "T"}
    nodes = list(tree.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n_nodes = len(nodes)

    # Use int8 for active simulation (easier to mutate)
    linear_mat = np.zeros((n_nodes, gen_length), dtype=np.int8)
    poisson_mat = np.zeros((n_nodes, gen_length), dtype=np.int8)

    # Generate Reference
    ref_seq = toit.rng.choice(BASES, size=gen_length)

    def mutate(seq, n_mut):
        if n_mut <= 0:
            return seq.copy()
        new_seq = seq.copy()
        pos_indices = np.array(toit.rng.choice(gen_length, size=n_mut, replace=False))
        for pos in pos_indices:
            current = new_seq[pos]
            # Choose any base except current
            new_seq[pos] = toit.rng.choice(BASES[BASES != current])
        return new_seq

    # Initialize Roots (Random drift from reference)
    roots = [n for n, d in tree.in_degree() if d == 0]
    for root in roots:
        idx = node_to_idx[root]
        root_drift = int(toit.rng.choice(range(1, 35)))
        root_seq = mutate(ref_seq, root_drift)
        linear_mat[idx] = root_seq
        poisson_mat[idx] = root_seq

    # Traverse Tree
    print("Simulating mutations along transmission tree...")
    for root in roots:
        for parent, child in tqdm(nx.dfs_edges(tree, source=root), total=tree.number_of_edges(), desc="Evolving"):
            par_idx = node_to_idx[parent]
            chi_idx = node_to_idx[child]

            # 1. Calculate Time Duration (Psi)
            # Psi = |Sample_Par - Trans| + |Sample_Chi - Trans|
            try:
                t_trans = tree.nodes[child]["exposure_date"]
                t_samp_par = tree.nodes[parent]["sample_date"]
                t_samp_chi = tree.nodes[child]["sample_date"]
            except KeyError:
                raise ValueError("Missing dates in tree. Run epidemic simulation first.")

            psi = abs(t_samp_par - t_trans) + abs(t_samp_chi - t_trans)

            # 2. Get Clock Rate from TOIT (Strict or Relaxed)
            rate_val = toit.sample_clock_rate_per_day().item()
            genetic_dist = rate_val * psi

            # 3. Mutate (Linear)
            n_lin = int(round(genetic_dist))
            linear_mat[chi_idx] = mutate(linear_mat[par_idx], n_lin)

            # 4. Mutate (Poisson)
            n_poi = int(poisson.rvs(genetic_dist, random_state=toit.rng))
            poisson_mat[chi_idx] = mutate(poisson_mat[par_idx], n_poi)

    print("Packing data into 2-bit format...")
    packed = {
        "linear": PackedGenomicData(linear_mat, gen_length, node_to_idx, BASES_MAP),
        "poisson": PackedGenomicData(poisson_mat, gen_length, node_to_idx, BASES_MAP),
    }
    raw = {"linear": linear_mat, "poisson": poisson_mat}

    out = {"packed": packed, "raw": raw if return_raw else None}

    return out


def generate_pairwise_data(
        packed_genomic_data: dict[str, PackedGenomicData],
        tree: nx.DiGraph
) -> pd.DataFrame:
    """
    Generates a long-format DataFrame with Linear, Poisson, and Temporal distances.

    Args:
        packed_genomic_data: Output dict from simulate_genomic_data containing:
                         - 'linear': PackedGenomicData object
                         - 'poisson': PackedGenomicData object
        tree: NetworkX DiGraph with 'sample_date' node attributes.

    Returns:
        pd.DataFrame: Columns ['NodeA', 'NodeB', 'LinearDist', 'PoissonDist', 'TemporalDist', 'Related']
    """
    # 1. Retrieve Data & Map
    # We assume both linear/poisson share the same node map (they should)
    packed_linear = packed_genomic_data['linear']
    packed_poisson = packed_genomic_data['poisson']
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

    df = pd.DataFrame({
        'NodeA': id_array[rows],
        'NodeB': id_array[cols],
        'LinearDist': mat_linear[rows, cols],
        'PoissonDist': mat_poisson[rows, cols],
        'TemporalDist': mat_temporal[rows, cols],
        'Related': mat_related[rows, cols]
    })

    return df
