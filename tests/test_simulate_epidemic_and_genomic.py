"""
Comprehensive tests for simulate_epidemic_and_genomic utilities.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from epilink import (
    TOIT,
    MolecularClock,
    PackedGenomicData,
    SequencePacker64,
    generate_pairwise_data,
    populate_epidemic_data,
    simulate_genomic_data,
)


def _naive_hamming(int8_matrix: np.ndarray) -> np.ndarray:
    n = int8_matrix.shape[0]
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.sum(int8_matrix[i] != int8_matrix[j]))
            out[i, j] = dist
            out[j, i] = dist
    return out


def _row_for_pair(df, node_a: str, node_b: str):
    mask = ((df["NodeA"] == node_a) & (df["NodeB"] == node_b)) | (
        (df["NodeA"] == node_b) & (df["NodeB"] == node_a)
    )
    return df[mask].iloc[0]


def test_populate_epidemic_data_deterministic_dates_and_sampling():
    tree = nx.DiGraph()
    tree.add_edges_from([("A", "B"), ("A", "C")])
    toit = TOIT(rng_seed=7)

    out = populate_epidemic_data(
        toit=toit,
        tree=tree,
        prop_sampled=0.5,
        sampling_scale=0.0,
        sampling_shape=3.0,
        root_start_range=0,
    )

    latent = toit.params.latent_shape * toit.params.incubation_scale
    presymp = toit.params.presymptomatic_shape * toit.params.incubation_scale

    assert np.isclose(out.nodes["A"]["exposure_date"], 0.0)
    assert np.isclose(out.nodes["A"]["date_infectious"], latent)
    assert np.isclose(out.nodes["A"]["date_symptom_onset"], latent + presymp)
    assert np.isclose(out.nodes["A"]["sample_date"], latent + presymp)
    assert out.nodes["A"]["seed"] is True

    for child in ("B", "C"):
        assert np.isclose(out.nodes[child]["exposure_date"], out.nodes["A"]["date_infectious"])
        assert np.isclose(out.nodes[child]["date_infectious"], out.nodes[child]["exposure_date"] + latent)
        assert np.isclose(
            out.nodes[child]["date_symptom_onset"],
            out.nodes[child]["date_infectious"] + presymp,
        )
        assert np.isclose(out.nodes[child]["sample_date"], out.nodes[child]["date_symptom_onset"])
        assert out.nodes[child]["seed"] is False

    sampled_flags = [out.nodes[n]["sampled"] for n in out.nodes]
    assert sum(sampled_flags) == 2


def test_sequence_packer_hamming_matches_naive():
    rng = np.random.default_rng(0)
    int8_matrix = rng.integers(0, 4, size=(3, 35), dtype=np.int8)

    packed = SequencePacker64.pack_u64(int8_matrix)
    hamming = SequencePacker64.hamming64(packed)
    expected = _naive_hamming(int8_matrix)

    np.testing.assert_allclose(hamming, expected, rtol=0.0, atol=0.0)


def test_packed_genomic_data_write_fasta(tmp_path):
    int8_matrix = np.array(
        [
            [0, 1, 2, 3, 0, 0],
            [3, 3, 2, 2, 1, 1],
        ],
        dtype=np.int8,
    )
    node_map = {"n1": 0, "n2": 1}
    base_map = {0: "A", 1: "C", 2: "G", 3: "T"}

    packed = PackedGenomicData(int8_matrix, original_length=6, node_map=node_map, base_map=base_map)
    out_path = tmp_path / "out.fasta"
    packed.write_fasta(str(out_path))

    contents = out_path.read_text()
    assert contents == ">n1\nACGTAA\n>n2\nTTGGCC\n"


def test_simulate_genomic_data_zero_branch_length():
    tree = nx.DiGraph()
    tree.add_edge("A", "B")
    tree.nodes["A"]["sample_date"] = 0.0
    tree.nodes["B"]["sample_date"] = 0.0
    tree.nodes["B"]["exposure_date"] = 0.0

    clock = MolecularClock(relax_rate=False, gen_len=64, rng_seed=5)
    out = simulate_genomic_data(clock=clock, tree=tree, return_raw=True)

    linear_raw = out["raw"]["linear"]
    poisson_raw = out["raw"]["poisson"]
    idx_map = out["packed"]["linear"].node_to_idx
    idx_a = idx_map["A"]
    idx_b = idx_map["B"]

    assert np.array_equal(linear_raw[idx_a], linear_raw[idx_b])
    assert np.array_equal(poisson_raw[idx_a], poisson_raw[idx_b])
    assert np.array_equal(linear_raw[idx_a], poisson_raw[idx_a])
    assert np.all((linear_raw >= 0) & (linear_raw <= 3))


def test_generate_pairwise_data_relationships():
    tree = nx.DiGraph()
    tree.add_edges_from([("A", "B"), ("A", "C"), ("B", "D")])
    tree.nodes["A"]["sample_date"] = 0
    tree.nodes["B"]["sample_date"] = 2
    tree.nodes["C"]["sample_date"] = 5
    tree.nodes["D"]["sample_date"] = 7
    tree.nodes["A"]["sampled"] = True
    tree.nodes["B"]["sampled"] = True
    tree.nodes["C"]["sampled"] = False
    tree.nodes["D"]["sampled"] = True

    int8_matrix = np.array(
        [
            [0, 0, 0, 0],  # A
            [0, 0, 0, 1],  # B
            [0, 0, 1, 1],  # C
            [1, 0, 0, 1],  # D
        ],
        dtype=np.int8,
    )
    node_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    base_map = {0: "A", 1: "C", 2: "G", 3: "T"}
    packed_linear = PackedGenomicData(int8_matrix, original_length=4, node_map=node_map, base_map=base_map)
    packed_poisson = PackedGenomicData(int8_matrix, original_length=4, node_map=node_map, base_map=base_map)

    df = generate_pairwise_data({"linear": packed_linear, "poisson": packed_poisson}, tree)

    assert set(df.columns) == {
        "NodeA",
        "NodeB",
        "Related",
        "Sampled",
        "LinearDist",
        "PoissonDist",
        "TemporalDist",
    }
    assert len(df) == 6

    row_bc = _row_for_pair(df, "B", "C")
    assert bool(row_bc["Related"]) is True

    row_cd = _row_for_pair(df, "C", "D")
    assert bool(row_cd["Related"]) is False

    row_ab = _row_for_pair(df, "A", "B")
    assert bool(row_ab["Sampled"]) is True
    assert row_ab["LinearDist"] == 1
    assert row_ab["PoissonDist"] == 1

    row_ac = _row_for_pair(df, "A", "C")
    assert bool(row_ac["Sampled"]) is False

    row_bd = _row_for_pair(df, "B", "D")
    assert row_bd["TemporalDist"] == 5
