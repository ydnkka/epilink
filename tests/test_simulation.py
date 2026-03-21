from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from epilink import (  # noqa: E402
    PackedGenomicData,
    SequencePacker64,
    SimulationResult,
    SimulationSequenceSet,
    build_pairwise_case_table,
    simulate_epidemic_dates,
    simulate_genomic_sequences,
)

BASE_MAP = {0: "A", 1: "C", 2: "G", 3: "T"}


class DeterministicSimulationProfile:
    def __init__(
        self,
        *,
        latent: float = 2.0,
        presymptomatic: float = 3.0,
        testing: float = 4.0,
        transmission: float = 5.0,
        expected_mutation_scale: float = 0.0,
        rng_seed: int = 123,
    ) -> None:
        self.latent = latent
        self.presymptomatic = presymptomatic
        self.testing = testing
        self.transmission = transmission
        self.expected_mutation_scale = expected_mutation_scale
        self.rng = np.random.default_rng(rng_seed)

    @staticmethod
    def _full(
        size: int | tuple[int, ...] = 1,
        value: float = 0.0,
    ) -> np.ndarray:
        sample_shape = (size,) if isinstance(size, int) else size
        return np.full(sample_shape, value, dtype=float)

    def sample_latent_periods(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        return self._full(size, self.latent)

    def sample_presymptomatic_periods(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        return self._full(size, self.presymptomatic)

    def sample_testing_delays(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        return self._full(size, self.testing)

    def rvs(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        return self._full(size, self.transmission)

    def expected_mutations(self, branch_length_days: float | np.ndarray) -> np.ndarray:
        return np.asarray(branch_length_days, dtype=float) * self.expected_mutation_scale


class TestSequencePacker64(unittest.TestCase):
    def test_pack_u64_and_hamming64_match_direct_hamming_distances(self) -> None:
        sequence_a = np.tile(np.array([0, 1, 2, 3], dtype=np.int8), 9)
        sequence_b = sequence_a.copy()
        sequence_b[[0, 17, 35]] = np.array([1, 2, 0], dtype=np.int8)
        sequence_c = np.full(36, 3, dtype=np.int8)
        sequences = np.vstack([sequence_a, sequence_b, sequence_c])

        packed = SequencePacker64.pack_u64(sequences)
        distances = SequencePacker64.hamming64(packed)
        expected = np.count_nonzero(
            sequences[:, np.newaxis, :] != sequences[np.newaxis, :, :],
            axis=2,
        ).astype(np.int32)

        self.assertEqual(packed.shape, (3, 2))
        self.assertEqual(packed.dtype, np.uint64)
        np.testing.assert_array_equal(distances, expected)

    def test_hamming64_matches_direct_distances_across_multiple_blocks(self) -> None:
        rng = np.random.default_rng(7)
        sequences = rng.integers(0, 4, size=(6, 97), dtype=np.int8)

        packed = SequencePacker64.pack_u64(sequences)
        distances = SequencePacker64.hamming64(packed, block_size=2)
        expected = np.count_nonzero(
            sequences[:, np.newaxis, :] != sequences[np.newaxis, :, :],
            axis=2,
        ).astype(np.int32)

        np.testing.assert_array_equal(distances, expected)


class TestPackedGenomicData(unittest.TestCase):
    def test_validates_input_matrix_dtype_and_value_range(self) -> None:
        with self.assertRaises(TypeError):
            PackedGenomicData(
                np.array([[0.0, 1.0]], dtype=float),
                original_length=2,
                node_map={"case-1": 0},
                base_map=BASE_MAP,
            )

        with self.assertRaises(ValueError):
            PackedGenomicData(
                np.array([[0, 4]], dtype=np.int8),
                original_length=2,
                node_map={"case-1": 0},
                base_map=BASE_MAP,
            )

    def test_compute_hamming_distances_handles_non_contiguous_packed_storage(self) -> None:
        base_pattern = np.tile(np.array([0, 1, 2, 3], dtype=np.int8), 10)
        sequences = np.vstack(
            [
                base_pattern,
                np.roll(base_pattern, 1),
                np.roll(base_pattern, 5),
            ]
        )
        packed = PackedGenomicData(
            np.asfortranarray(sequences),
            original_length=sequences.shape[1],
            node_map={"A": 0, "B": 1, "C": 2},
            base_map=BASE_MAP,
        )
        packed.packed_u64 = packed.packed_u64[:, ::-1]
        expected = np.count_nonzero(
            sequences[:, np.newaxis, :] != sequences[np.newaxis, :, :],
            axis=2,
        ).astype(np.int32)

        self.assertFalse(packed.packed_u64.flags["C_CONTIGUOUS"])
        np.testing.assert_array_equal(packed.compute_hamming_distances(block_size=1), expected)

    def test_write_fasta_outputs_wrapped_sequences_with_node_headers(self) -> None:
        sequence_a = np.concatenate(
            [
                np.zeros(100, dtype=np.int8),
                np.array([1, 2, 3, 0, 1], dtype=np.int8),
            ]
        )
        sequence_b = np.full(105, 3, dtype=np.int8)
        packed = PackedGenomicData(
            np.vstack([sequence_a, sequence_b]),
            original_length=105,
            node_map={"case-1": 0, "case-2": 1},
            base_map=BASE_MAP,
        )

        with TemporaryDirectory() as temp_dir:
            fasta_path = Path(temp_dir) / "simulated.fasta"
            packed.write_fasta(str(fasta_path))
            lines = fasta_path.read_text().splitlines()

        self.assertEqual(lines[0], ">case-1")
        self.assertEqual(lines[1], "A" * 100)
        self.assertEqual(lines[2], "CGTAC")
        self.assertEqual(lines[3], ">case-2")
        self.assertEqual(lines[4], "T" * 100)
        self.assertEqual(lines[5], "T" * 5)


class TestSimulationHelpers(unittest.TestCase):
    def test_simulate_epidemic_dates_adds_dates_without_mutating_input_tree(self) -> None:
        tree = nx.DiGraph([("root", "child")])
        tree.add_node("other-root")
        profile = DeterministicSimulationProfile()

        simulated = simulate_epidemic_dates(profile, tree, fraction_sampled=0.5)

        self.assertNotIn("sampled", tree.nodes["root"])
        self.assertEqual(sum(simulated.nodes[node]["sampled"] for node in simulated.nodes), 2)

        root = simulated.nodes["root"]
        child = simulated.nodes["child"]
        other_root = simulated.nodes["other-root"]

        self.assertTrue(root["seed"])
        self.assertFalse(child["seed"])
        self.assertTrue(other_root["seed"])
        self.assertGreaterEqual(root["exposure_date"], 0)
        self.assertLess(root["exposure_date"], 30)
        self.assertEqual(root["date_infectious"] - root["exposure_date"], 2.0)
        self.assertEqual(root["date_symptom_onset"] - root["date_infectious"], 3.0)
        self.assertEqual(root["sample_date"] - root["date_symptom_onset"], 4.0)
        self.assertEqual(child["exposure_date"], root["date_infectious"] + 5.0)
        self.assertEqual(child["date_infectious"] - child["exposure_date"], 2.0)
        self.assertEqual(child["date_symptom_onset"] - child["date_infectious"], 3.0)
        self.assertEqual(child["sample_date"] - child["date_symptom_onset"], 4.0)

    def test_simulate_epidemic_dates_rejects_invalid_sampling_fraction(self) -> None:
        tree = nx.DiGraph([("root", "child")])

        with self.assertRaises(ValueError):
            simulate_epidemic_dates(
                DeterministicSimulationProfile(),
                tree,
                fraction_sampled=1.5,
            )

    def test_simulate_genomic_sequences_requires_epidemic_dates(self) -> None:
        tree = nx.DiGraph([("root", "child")])

        with self.assertRaises(ValueError):
            simulate_genomic_sequences(
                DeterministicSimulationProfile(),
                tree,
                genome_length=8,
            )

    def test_simulate_genomic_sequences_returns_raw_and_packed_outputs(self) -> None:
        tree = nx.DiGraph([("root", "child")])
        tree.nodes["root"].update(exposure_date=0.0, sample_date=6.0)
        tree.nodes["child"].update(exposure_date=2.0, sample_date=8.0)

        result = simulate_genomic_sequences(
            DeterministicSimulationProfile(expected_mutation_scale=0.0, rng_seed=9),
            tree,
            genome_length=12,
            return_raw=True,
        )

        self.assertIsInstance(result, SimulationResult)
        self.assertIsInstance(result.packed, SimulationSequenceSet)
        self.assertEqual(set(result["packed"]), {"linear", "poisson"})
        self.assertEqual(set(result["raw"]), {"linear", "poisson"})
        self.assertEqual(result.packed.linear.node_to_idx, {"root": 0, "child": 1})
        self.assertEqual(result["packed"]["linear"].node_to_idx, {"root": 0, "child": 1})
        self.assertEqual(result["packed"]["linear"].original_length, 12)
        self.assertEqual(result["raw"]["linear"].shape, (2, 12))
        self.assertTrue(np.all((result["raw"]["linear"] >= 0) & (result["raw"]["linear"] <= 3)))
        np.testing.assert_array_equal(result["raw"]["linear"][0], result["raw"]["linear"][1])
        np.testing.assert_array_equal(result["raw"]["poisson"][0], result["raw"]["poisson"][1])
        np.testing.assert_array_equal(
            result["packed"]["linear"].compute_hamming_distances(),
            np.zeros((2, 2), dtype=np.int32),
        )

        without_raw = simulate_genomic_sequences(
            DeterministicSimulationProfile(expected_mutation_scale=0.0, rng_seed=9),
            tree,
            genome_length=12,
            return_raw=False,
        )
        self.assertIsNone(without_raw["raw"])

    def test_simulate_genomic_sequences_rejects_non_positive_genome_length(self) -> None:
        tree = nx.DiGraph([("root", "child")])
        tree.nodes["root"].update(exposure_date=0.0, sample_date=6.0)
        tree.nodes["child"].update(exposure_date=2.0, sample_date=8.0)

        with self.assertRaises(ValueError):
            simulate_genomic_sequences(
                DeterministicSimulationProfile(expected_mutation_scale=0.0, rng_seed=9),
                tree,
                genome_length=0,
            )

    def test_simulate_genomic_sequences_linear_model_mutates_selected_sites(self) -> None:
        tree = nx.DiGraph([("root", "child")])
        tree.nodes["root"].update(exposure_date=0.0, sample_date=6.0)
        tree.nodes["child"].update(exposure_date=2.0, sample_date=8.0)

        result = simulate_genomic_sequences(
            DeterministicSimulationProfile(expected_mutation_scale=0.5, rng_seed=9),
            tree,
            genome_length=4,
            return_raw=True,
        )

        linear_sequences = result["raw"]["linear"]
        self.assertEqual(np.count_nonzero(linear_sequences[0] != linear_sequences[1]), 4)
        self.assertTrue(np.all((linear_sequences >= 0) & (linear_sequences <= 3)))

    def test_build_pairwise_case_table_reports_relationships_sampling_and_distances(self) -> None:
        tree = nx.DiGraph([("A", "B"), ("A", "C")])
        tree.add_node("D")
        nx.set_node_attributes(
            tree,
            {
                "A": {"sample_date": 10.0, "sampled": True},
                "B": {"sample_date": 12.0, "sampled": True},
                "C": {"sample_date": 17.0, "sampled": False},
                "D": {"sample_date": 20.0, "sampled": True},
            },
        )

        packed = {
            "linear": PackedGenomicData(
                np.array(
                    [
                        [0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [2, 0, 0, 0],
                        [3, 3, 3, 3],
                    ],
                    dtype=np.int8,
                ),
                original_length=4,
                node_map={"A": 0, "B": 1, "C": 2, "D": 3},
                base_map=BASE_MAP,
            ),
            "poisson": PackedGenomicData(
                np.array(
                    [
                        [0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [2, 2, 0, 0],
                        [3, 3, 3, 0],
                    ],
                    dtype=np.int8,
                ),
                original_length=4,
                node_map={"A": 0, "B": 1, "C": 2, "D": 3},
                base_map=BASE_MAP,
            ),
        }

        table = build_pairwise_case_table(packed, tree)
        rows = {(row.CaseA, row.CaseB): row for row in table.itertuples(index=False)}

        self.assertEqual(len(table), 6)
        self.assertTrue(rows[("A", "B")].IsRelated)
        self.assertTrue(rows[("A", "B")].BothSampled)
        self.assertEqual(rows[("A", "B")].DeterministicDistance, 1)
        self.assertEqual(rows[("A", "B")].StochasticDistance, 2)
        self.assertEqual(rows[("A", "B")].SamplingDateDistanceDays, 2.0)

        self.assertTrue(rows[("B", "C")].IsRelated)
        self.assertFalse(rows[("B", "C")].BothSampled)
        self.assertEqual(rows[("B", "C")].DeterministicDistance, 2)
        self.assertEqual(rows[("B", "C")].StochasticDistance, 3)
        self.assertEqual(rows[("B", "C")].SamplingDateDistanceDays, 5.0)

        self.assertFalse(rows[("A", "D")].IsRelated)
        self.assertTrue(rows[("A", "D")].BothSampled)
        self.assertEqual(rows[("A", "D")].DeterministicDistance, 4)
        self.assertEqual(rows[("A", "D")].StochasticDistance, 3)
        self.assertEqual(rows[("A", "D")].SamplingDateDistanceDays, 10.0)


if __name__ == "__main__":
    unittest.main()
