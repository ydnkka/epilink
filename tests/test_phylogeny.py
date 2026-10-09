from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd

from epilink import (
    ConfigurationError,
    PackedGenomicData,
    PhylogenyError,
    PhylogenyResult,
    SimulationResult,
    SimulationSequenceSet,
    build_phylogenetic_tree,
)

try:
    from Bio import Phylo, SeqIO
except ImportError:
    Phylo = SeqIO = None

BASE_MAP = {0: "A", 1: "C", 2: "G", 3: "T"}
IQTREE = os.environ.get("EPILINK_IQTREE") or next(
    (path for name in ("iqtree3", "iqtree2", "iqtree") if (path := shutil.which(name))),
    None,
)


def _fixture(last_label="sample two"):
    reference = np.tile(np.array([0, 1, 2, 3], dtype=np.int8), 17)[:65]
    sequences = np.tile(reference, (4, 1))
    sequences[1, [0, 32, 64]] = (sequences[1, [0, 32, 64]] + 1) % 4
    sequences[2, [4, 35, 62]] = (sequences[2, [4, 35, 62]] + 2) % 4
    sequences[3, [8, 39, 60]] = (sequences[3, [8, 39, 60]] + 3) % 4
    # Row order deliberately differs from graph order; stochastic rows differ too.
    mapping = {last_label: 2, "unsampled": 0, 42: 3, "reference": 1}
    packed = SimulationSequenceSet(
        deterministic=PackedGenomicData(sequences, 65, mapping.copy(), BASE_MAP),
        stochastic=PackedGenomicData((sequences + 1) % 4, 65, mapping.copy(), BASE_MAP),
    )
    tree = nx.DiGraph()
    tree.add_node("unsampled", sampled=False, sample_date=np.nan)
    tree.add_node("reference", sampled=True, sample_date=10.0)
    tree.add_node(42, sampled=True, sample_date=20.0)
    tree.add_node(last_label, sampled=True, sample_date=30.0)
    tree.add_edges_from([("unsampled", 42), (42, "reference"), ("reference", last_label)])
    return SimulationResult(packed=packed, raw=None, reference_sequence=reference), tree, sequences


def _write_backend_outputs(command, *, cwd, stdout, **kwargs):
    cwd = Path(cwd)
    stdout.write("IQ-TREE test backend\n")
    (cwd / "iqtree.iqtree").write_text("Substitution model: JC\n", encoding="utf-8")
    (cwd / "iqtree.treefile").write_text(
        "(epilink_reference:0.01,(epilink_tip_0:0.02,"
        "(epilink_tip_1:0.03,epilink_tip_2:0.04):0.05):0.06);\n",
        encoding="utf-8",
    )
    if "--date" in command:
        dates = [
            float(line.split()[1]) for line in (cwd / "sampling_dates.txt").read_text().splitlines()
        ]
        root_date = min(dates) - 5
        internal_date = min(dates[1:]) - 5
        time_tree = (
            f'(epilink_tip_0[&date="{dates[0]}"]:{dates[0] - root_date},'
            f'(epilink_tip_1[&date="{dates[1]}"]:{dates[1] - internal_date},'
            f'epilink_tip_2[&date="{dates[2]}"]:{dates[2] - internal_date})'
            f'[&date="{internal_date}"]:{internal_date - root_date})[&date="{root_date}"];'
        )
        (cwd / "iqtree.timetree.nex").write_text(
            f"#NEXUS\nBegin trees;\ntree 1 = {time_tree}\nEnd;\n", encoding="utf-8"
        )
        # Real LSD2 writes substitution lengths to this misleadingly named file.
        (cwd / "iqtree.timetree.nwk").write_text(
            "(epilink_tip_0:0.01,(epilink_tip_1:0.01,epilink_tip_2:0.03):0.02);\n",
            encoding="utf-8",
        )
        (cwd / "iqtree.timetree.lsd").write_text(
            "*RESULTS:\n- Dating results:\n rate 0.002, tMRCA 5, objective function 1e-8\n",
            encoding="utf-8",
        )
    return subprocess.CompletedProcess(command, 0)


class TestPhylogenyDependencies(unittest.TestCase):
    def test_base_package_imports_without_biopython(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.modules['Bio'] = None; import epilink; "
                "assert callable(epilink.build_phylogenetic_tree)",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_missing_biopython_has_install_instructions(self):
        simulation, tree, _ = _fixture()
        with (
            patch.dict(sys.modules, {"Bio": None}),
            self.assertRaisesRegex(ImportError, r"epilink\[phylogeny\]"),
        ):
            build_phylogenetic_tree(simulation, tree)


@unittest.skipUnless(Phylo is not None, "Install epilink[phylogeny] for phylogeny tests")
class TestPhylogeneticInference(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output_dir = Path(self.temp.name) / "path with spaces"
        self.simulation, self.tree, self.sequences = _fixture()
        discovery = patch("epilink.simulation.phylogeny.shutil.which", return_value="/test/iqtree3")
        self.which = discovery.start()
        self.addCleanup(discovery.stop)
        backend = patch(
            "epilink.simulation.phylogeny.subprocess.run", side_effect=_write_backend_outputs
        )
        self.backend = backend.start()
        self.addCleanup(backend.stop)

    def build(self, **kwargs):
        return build_phylogenetic_tree(
            self.simulation, self.tree, output_dir=self.output_dir, **kwargs
        )

    def test_sampled_sequence_selection_dates_and_units(self):
        before_tree = self.tree.copy()
        before_packed = self.simulation.packed.stochastic.packed_u64.copy()
        before_reference = self.simulation.reference_sequence.copy()
        result = self.build(threads=2, seed=12, timeout=15)
        self.assertIsInstance(result, PhylogenyResult)
        self.assertEqual(result.sample_ids, ("reference", 42, "sample two"))
        self.assertEqual(result.reference_name, "reference_1")
        records = list(SeqIO.parse(result.output_paths["alignment"], "fasta"))
        expected = [(self.sequences[i] + 1) % 4 for i in (1, 3, 2)]
        expected.append(self.simulation.reference_sequence)
        self.assertEqual(len(records), 4)
        for record, sequence in zip(records, expected, strict=True):
            self.assertEqual(str(record.seq), "".join(BASE_MAP[int(base)] for base in sequence))
        dates = result.output_paths["sampling_dates"].read_text().splitlines()
        self.assertEqual(dates, ["epilink_tip_0\t10", "epilink_tip_1\t20", "epilink_tip_2\t30"])
        self.assertEqual(
            {tip.name for tip in result.raw_tree.get_terminals()},
            {"reference", "42", "sample two", "reference_1"},
        )
        self.assertTrue(result.raw_tree.rooted)
        self.assertIn("reference_1", [clade.name for clade in result.raw_tree.root.clades])
        self.assertAlmostEqual(result.raw_tree.distance("42", "sample two"), 0.07)
        self.assertEqual(
            {tip.name for tip in result.dated_tree.get_terminals()},
            {"reference", "42", "sample two"},
        )
        self.assertAlmostEqual(result.dated_tree.distance("42", "sample two"), 20.0)
        exported = Phylo.read(result.output_paths["dated_tree"], "newick")
        self.assertAlmostEqual(exported.distance("42", "sample two"), 20.0)
        nexus = Phylo.read(result.output_paths["dated_nexus"], "nexus")
        self.assertTrue(nexus.rooted)
        # Biopython's legacy NEXUS reader leaves quotes on labels with spaces.
        nexus_sample = next(
            tip for tip in nexus.get_terminals() if tip.name.strip("'") == "sample two"
        )
        self.assertAlmostEqual(nexus.distance("42", nexus_sample), 20.0)
        tips = result.node_dates[result.node_dates.is_tip].set_index("node")
        self.assertEqual(tips.loc["42", "case_id"], 42)
        self.assertEqual(tips.date.to_dict(), {"reference": 10.0, "42": 20.0, "sample two": 30.0})
        self.assertEqual(result.clock_rate, 0.002)
        pd.testing.assert_frame_equal(
            pd.read_csv(result.output_paths["node_dates"], sep="\t"),
            result.node_dates.assign(
                case_id=result.node_dates.case_id.map(
                    lambda value: str(value) if value is not None else np.nan
                )
            ),
            check_dtype=False,
        )
        for path in result.output_paths.values():
            self.assertTrue(path.is_file(), path)
        command = self.backend.call_args.args[0]
        self.assertEqual(command[command.index("-T") + 1], "2")
        self.assertEqual(command[command.index("-seed") + 1], "12")
        self.assertIn("-keep-ident", command)
        self.assertEqual(self.backend.call_args.kwargs["timeout"], 15)
        self.assertEqual(dict(before_tree.nodes(data=True)), dict(self.tree.nodes(data=True)))
        self.assertEqual(list(before_tree.edges), list(self.tree.edges))
        np.testing.assert_array_equal(before_packed, self.simulation.packed.stochastic.packed_u64)
        np.testing.assert_array_equal(before_reference, self.simulation.reference_sequence)
        self.assertIs(result.to_dict()["raw_tree"], result.raw_tree)

    def test_deterministic_genetic_tree_needs_no_dates(self):
        nx.set_node_attributes(self.tree, None, "sample_date")
        result = self.build(sequence_model="deterministic", dated=False, model="GTR+G")
        records = list(SeqIO.parse(result.output_paths["alignment"], "fasta"))
        self.assertEqual(
            str(records[0].seq), "".join(BASE_MAP[int(base)] for base in self.sequences[1])
        )
        self.assertIsNone(result.dated_tree)
        self.assertIsNone(result.clock_rate)
        self.assertTrue(result.node_dates.empty)
        self.assertNotIn("sampling_dates", result.output_paths)
        command = self.backend.call_args.args[0]
        self.assertNotIn("--date", command)
        self.assertEqual(command[command.index("-m") + 1], "GTR+G")

    def test_identical_tip_sequences_are_retained(self):
        identical = PackedGenomicData(
            np.tile(self.simulation.reference_sequence, (4, 1)),
            65,
            self.simulation.packed.stochastic.node_to_idx,
            BASE_MAP,
        )
        self.simulation = replace(
            self.simulation, packed=SimulationSequenceSet(identical, identical)
        )
        result = self.build(dated=False)
        records = list(SeqIO.parse(result.output_paths["alignment"], "fasta"))
        self.assertEqual(len({str(record.seq) for record in records}), 1)
        self.assertEqual(len(result.raw_tree.get_terminals()), 4)
        self.assertIn("-keep-ident", self.backend.call_args.args[0])

    def test_fixed_rate_allows_equal_sampling_dates(self):
        nx.set_node_attributes(self.tree, 10.0, "sample_date")
        with self.assertRaisesRegex(ConfigurationError, "Distinct sampling dates"):
            self.build()
        result = self.build(clock_rate=0.002)
        self.assertEqual(float(result.output_paths["clock_rate"].read_text()), 0.002)
        self.assertIn("-w clock_rate.txt", self.backend.call_args.args[0][-1])
        self.assertEqual(set(result.node_dates.loc[result.node_dates.is_tip, "date"]), {10.0})

    def test_repeated_runs_have_separate_artifacts(self):
        first = self.build(dated=False)
        content = first.output_paths["raw_tree"].read_text()
        second = self.build(dated=False)
        self.assertNotEqual(
            first.output_paths["raw_tree"].parent, second.output_paths["raw_tree"].parent
        )
        self.assertEqual(first.output_paths["raw_tree"].read_text(), content)

    def test_internal_names_do_not_collide_with_case_labels(self):
        self.simulation, self.tree, _ = _fixture("internal_0")
        result = self.build()
        self.assertTrue(result.node_dates.node.is_unique)
        self.assertIn("internal_0", [tip.name for tip in result.dated_tree.get_terminals()])

    def test_invalid_options_do_not_launch_backend(self):
        options = [
            {"sequence_model": "invalid"},
            {"threads": 0},
            {"threads": True},
            {"seed": -1},
            {"seed": 1.5},
            {"model": ""},
            {"model": "GTR +G"},
            {"clock_rate": 0},
            {"clock_rate": float("inf")},
            {"clock_rate": True},
            {"timeout": -1},
            {"timeout": float("nan")},
            {"dated": False, "clock_rate": 0.002},
        ]
        for kwargs in options:
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigurationError):
                self.build(**kwargs)
        self.backend.assert_not_called()
        self.assertFalse(self.output_dir.exists())

    def test_insufficient_samples_and_missing_sequences(self):
        self.tree.nodes[42]["sampled"] = False
        with self.assertRaisesRegex(ConfigurationError, "three sampled"):
            self.build()
        self.tree.nodes[42]["sampled"] = True
        for index in (None, -1, 4, 1, "row"):
            with self.subTest(index=index):
                self.simulation.packed.stochastic.node_to_idx[42] = index
                with self.assertRaisesRegex(ConfigurationError, "sequence mapping"):
                    self.build()
        self.backend.assert_not_called()

    def test_invalid_sample_dates(self):
        for date in (None, "2026-01-01", float("nan"), float("inf"), True):
            with self.subTest(date=date):
                self.tree.nodes[42]["sample_date"] = date
                with self.assertRaisesRegex(ConfigurationError, "finite numeric sample_date"):
                    self.build()
        self.backend.assert_not_called()

    def test_invalid_reference_data(self):
        for reference in (np.zeros(64, dtype=np.int8), np.full(65, 4, dtype=np.int8), np.zeros(65)):
            with self.subTest(reference=reference.dtype):
                self.simulation = replace(self.simulation, reference_sequence=reference)
                with self.assertRaisesRegex(ConfigurationError, "matching A/C/G/T"):
                    self.build()
        self.backend.assert_not_called()

    def test_ambiguous_or_multiline_labels_are_rejected(self):
        self.tree.add_node("42", sampled=True, sample_date=40.0)
        with self.assertRaisesRegex(ConfigurationError, "single-line labels"):
            self.build()
        self.tree.remove_node("42")
        self.tree = nx.relabel_nodes(self.tree, {"sample two": "multiline\nlabel"})
        with self.assertRaisesRegex(ConfigurationError, "single-line labels"):
            self.build()

    def test_missing_executable(self):
        self.which.return_value = None
        with self.assertRaisesRegex(PhylogenyError, "iqtree_executable"):
            self.build()
        self.assertEqual(self.which.call_count, 3)
        self.which.reset_mock()
        with self.assertRaises(PhylogenyError):
            self.build(iqtree_executable="/missing/iqtree")
        self.which.assert_called_once_with("/missing/iqtree")
        self.backend.assert_not_called()

    def test_process_failures_report_log_path(self):
        for error in (OSError("cannot execute"), subprocess.TimeoutExpired("iqtree", 1)):
            with self.subTest(error=type(error).__name__):
                self.backend.side_effect = error
                with self.assertRaisesRegex(PhylogenyError, "inference.log"):
                    self.build(timeout=1)
        self.backend.side_effect = None
        self.backend.return_value = subprocess.CompletedProcess([], 2)
        with self.assertRaisesRegex(PhylogenyError, "status 2"):
            self.build()

    def test_zero_exit_with_failed_dating_is_not_success(self):
        def failed_dating(command, **kwargs):
            completed = _write_backend_outputs(command, **kwargs)
            (Path(kwargs["cwd"]) / "iqtree.timetree.nex").unlink()
            return completed

        self.backend.side_effect = failed_dating
        with self.assertRaisesRegex(PhylogenyError, "Invalid IQ-TREE/LSD2 output"):
            self.build()
        self.assertEqual(len(list(self.output_dir.glob("run-*/raw_tree.nwk"))), 1)

    def test_corrupt_backend_outputs_are_rejected(self):
        for filename, content in (
            ("iqtree.treefile", "(epilink_reference:0.1,unexpected:0.2);\n"),
            ("iqtree.treefile", "(epilink_reference:0.1,(epilink_tip_0:0.1);\n"),
            (
                "iqtree.treefile",
                "(epilink_reference:0.1,(epilink_tip_0:0.1,epilink_tip_1:-0.1,epilink_tip_2:0.1):0.1);\n",
            ),
            (
                "iqtree.timetree.nex",
                "#NEXUS\nBegin trees; tree 1 = (epilink_tip_0:1,epilink_tip_1:1,epilink_tip_2:1); End;\n",
            ),
            ("iqtree.timetree.lsd", "Dating failed\n"),
            ("iqtree.timetree.lsd", " rate -0.002, tMRCA 5\n"),
        ):

            def corrupt(command, *, filename=filename, content=content, **kwargs):
                completed = _write_backend_outputs(command, **kwargs)
                (Path(kwargs["cwd"]) / filename).write_text(content, encoding="utf-8")
                return completed

            with self.subTest(filename=filename, content=content):
                self.backend.side_effect = corrupt
                with self.assertRaisesRegex(PhylogenyError, "inference.log"):
                    self.build()


@unittest.skipUnless(
    IQTREE and Phylo is not None, "Set EPILINK_IQTREE or install IQ-TREE for integration tests"
)
class TestIQTreeIntegration(unittest.TestCase):
    def test_real_inference_estimated_and_fixed_clock(self):
        base = np.tile(np.array([0, 1, 2, 3], dtype=np.int8), 250)
        sequences = np.tile(base, (5, 1))
        nodes = ["case 0", "case-1", "case-2", "case-3", "unsampled"]
        graph = nx.DiGraph()
        for index, node in enumerate(nodes):
            graph.add_node(node, sampled=index < 4, sample_date=(index + 1) * 10.0)
            if index < 4:
                sites = np.arange((index + 1) * 20)
                sequences[index, sites] = (sequences[index, sites] + 1) % 4
        packed = PackedGenomicData(
            sequences, 1000, dict(zip(nodes, range(5), strict=True)), BASE_MAP
        )
        simulation = SimulationResult(SimulationSequenceSet(packed, packed), None, base)
        with TemporaryDirectory() as temp:
            for rate in (None, 0.002):
                with self.subTest(clock_rate=rate):
                    result = build_phylogenetic_tree(
                        simulation,
                        graph,
                        output_dir=Path(temp) / "with spaces",
                        iqtree_executable=IQTREE,
                        clock_rate=rate,
                        timeout=60,
                    )
                    self.assertEqual(
                        {tip.name for tip in result.raw_tree.get_terminals()},
                        {*nodes[:4], "reference"},
                    )
                    self.assertEqual(
                        {tip.name for tip in result.dated_tree.get_terminals()}, set(nodes[:4])
                    )
                    tips = result.node_dates[result.node_dates.is_tip]
                    np.testing.assert_allclose(tips.date, tips.sample_date, atol=1e-4)
                    dates = result.node_dates.set_index("node").date
                    for parent in result.dated_tree.find_clades():
                        for child in parent.clades:
                            self.assertAlmostEqual(
                                child.branch_length,
                                dates[child.name] - dates[parent.name],
                                delta=1e-3,
                            )
                    self.assertGreater(result.clock_rate, 0)
                    if rate is not None:
                        self.assertAlmostEqual(result.clock_rate, rate)
                    exported = Phylo.read(result.output_paths["dated_tree"], "newick")
                    self.assertAlmostEqual(
                        exported.distance("case-2", "case-3"),
                        result.dated_tree.distance("case-2", "case-3"),
                    )


if __name__ == "__main__":
    unittest.main()
