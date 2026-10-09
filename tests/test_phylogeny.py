from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta
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
    build_phylogenetic_tree_from_fasta,
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
                "assert callable(epilink.build_phylogenetic_tree); "
                "assert callable(epilink.build_phylogenetic_tree_from_fasta)",
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

    def test_fasta_interface_imports_biopython_lazily(self):
        with (
            patch.dict(sys.modules, {"Bio": None}),
            self.assertRaisesRegex(ImportError, r"epilink\[phylogeny\]"),
        ):
            build_phylogenetic_tree_from_fasta("unused.fasta", reference_id="ref", dated=False)


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
        for sample_date in (None, "2026-01-01", float("nan"), float("inf"), True):
            with self.subTest(date=sample_date):
                self.tree.nodes[42]["sample_date"] = sample_date
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


@unittest.skipUnless(Phylo is not None, "Install epilink[phylogeny] for FASTA tests")
class TestFastaPhylogeneticInference(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.output_dir = self.directory / "output with spaces"
        self.alignment = self.directory / "aligned.fasta"
        self.sample_sequences = {
            "001": "acgtn-?ryswkmbdhv",
            "002": "ACGTN-?RYSWKMBDHV",
            "NA": "ACGTN-?RYSWKMBDHV",
        }
        self.reference_sequence = "ACGTACGTACGTACGTN"
        self.write_alignment(include_reference=True)
        self.dates = {"001": 10.5, "002": 20.5, "NA": 30.5}
        discovery = patch("epilink.simulation.phylogeny.shutil.which", return_value="/test/iqtree3")
        discovery.start()
        self.addCleanup(discovery.stop)
        backend = patch(
            "epilink.simulation.phylogeny.subprocess.run", side_effect=_write_backend_outputs
        )
        self.backend = backend.start()
        self.addCleanup(backend.stop)

    def write_alignment(self, *, include_reference):
        text = "".join(
            f">{name} sample description\n{seq}\n" for name, seq in self.sample_sequences.items()
        )
        if include_reference:
            text += f">ref reference description\n{self.reference_sequence}\n"
        self.alignment.write_text(text, encoding="utf-8")

    def build(self, **kwargs):
        options = {"reference_id": "ref", "dates": self.dates, "output_dir": self.output_dir}
        options.update(kwargs)
        return build_phylogenetic_tree_from_fasta(self.alignment, **options)

    def test_numeric_dates_and_ambiguous_alignment_are_preserved(self):
        before_alignment = self.alignment.read_text()
        before_dates = self.dates.copy()
        result = self.build()
        self.assertEqual(result.sample_ids, ("001", "002", "NA"))
        self.assertEqual(result.reference_name, "ref")
        self.assertIsNone(result.date_origin)
        self.assertNotIn("calendar_date", result.node_dates)
        self.assertEqual(
            result.node_dates[result.node_dates.is_tip].set_index("node").sample_date.to_dict(),
            self.dates,
        )
        sequences = [
            str(record.seq) for record in SeqIO.parse(result.output_paths["alignment"], "fasta")
        ]
        self.assertEqual(
            sequences,
            [seq.upper() for seq in self.sample_sequences.values()] + [self.reference_sequence],
        )
        self.assertEqual(
            {tip.name for tip in result.raw_tree.get_terminals()}, {"001", "002", "NA", "ref"}
        )
        self.assertEqual(
            {tip.name for tip in result.dated_tree.get_terminals()}, {"001", "002", "NA"}
        )
        command = self.backend.call_args.args[0]
        self.assertEqual(command[command.index("-m") + 1], "MFP")
        self.assertEqual(self.alignment.read_text(), before_alignment)
        self.assertEqual(self.dates, before_dates)

    def test_calendar_origin_leap_days_and_pre_origin_internal_dates(self):
        result = self.build(
            dates={
                "001": "2024-02-28",
                "002": date(2024, 3, 1),
                "NA": "2024-03-02",
                "ref": "1900-01-01",  # The outgroup date is not a calibration.
            }
        )
        self.assertEqual(result.date_origin, date(2024, 2, 28))
        tips = result.node_dates[result.node_dates.is_tip].set_index("node")
        self.assertEqual(tips.sample_date.to_dict(), {"001": 0.0, "002": 2.0, "NA": 3.0})
        self.assertEqual(tips.loc["002", "calendar_date"], pd.Timestamp("2024-03-01"))
        self.assertEqual(tips.loc["NA", "sample_calendar_date"], pd.Timestamp("2024-03-02"))
        root = result.node_dates.set_index("node").loc[result.dated_tree.root.name]
        self.assertEqual(root.date, -5.0)
        self.assertEqual(root.calendar_date, pd.Timestamp("2024-02-23"))
        self.assertTrue(pd.isna(root.sample_calendar_date))
        self.assertEqual(result.output_paths["date_origin"].read_text(), "2024-02-28\n")
        self.assertIn("2024-03-01", result.output_paths["node_dates"].read_text())
        self.assertIn("calendar_date=", result.output_paths["dated_nexus"].read_text())
        self.assertEqual(result.to_dict()["date_origin"], date(2024, 2, 28))

    def test_csv_and_tsv_dates_match_ids_not_row_order(self):
        for extension, separator in (("csv", ","), ("tsv", "\t")):
            for calendar in (False, True):
                with self.subTest(extension=extension, calendar=calendar):
                    path = self.directory / f"dates.{extension}"
                    values = (
                        ["2026-10-09", "2026-10-01", "2026-10-04", ""]
                        if calendar
                        else [30.5, 10.5, 20.5, ""]
                    )
                    pd.DataFrame(
                        {"case_id": ["NA", "001", "002", "ref"], "sample_date": values}
                    ).to_csv(path, sep=separator, index=False)
                    source = path.read_text()
                    result = self.build(dates=path)
                    self.assertEqual(result.sample_ids, ("001", "002", "NA"))
                    tips = result.node_dates[result.node_dates.is_tip].set_index("node")
                    self.assertEqual(
                        tips.sample_date.to_dict(),
                        {"001": 0.0, "002": 3.0, "NA": 8.0} if calendar else self.dates,
                    )
                    self.assertEqual(path.read_text(), source)

    def test_separate_reference_and_genetic_only_inference(self):
        self.write_alignment(include_reference=False)
        reference = self.directory / "reference.fasta"
        reference.write_text(f">external_ref\n{self.reference_sequence}\n", encoding="utf-8")
        result = self.build(reference_id=None, reference_fasta=reference)
        self.assertEqual(result.reference_name, "external_ref")
        self.assertIn("external_ref", [tip.name for tip in result.raw_tree.get_terminals()])
        genetic = self.build(
            reference_id=None,
            reference_fasta=reference,
            dated=False,
            dates=self.directory / "unused_dates.csv",
            model="GTR+G",
        )
        self.assertIsNone(genetic.dated_tree)
        self.assertIsNone(genetic.date_origin)
        self.assertTrue(genetic.node_dates.empty)
        self.assertNotIn("--date", self.backend.call_args.args[0])

    def test_same_day_calendar_samples_need_a_fixed_rate(self):
        dates = dict.fromkeys(self.dates, "2024-02-29")
        with self.assertRaisesRegex(ConfigurationError, "Distinct sampling dates"):
            self.build(dates=dates)
        result = self.build(dates=dates, clock_rate=0.002)
        self.assertEqual(result.date_origin, date(2024, 2, 29))
        self.assertEqual(set(result.node_dates.loc[result.node_dates.is_tip, "sample_date"]), {0.0})

    def test_reference_selection_and_separate_reference_validation(self):
        for kwargs in (
            {"reference_id": None},
            {"reference_fasta": self.alignment},
            {"reference_id": "missing"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigurationError):
                self.build(**kwargs)
        path = self.directory / "reference.fasta"
        for content in (
            f">a\n{self.reference_sequence}\n>b\n{self.reference_sequence}\n",
            f">ref\n{self.reference_sequence}\n",
            ">different\nACGT\n",
        ):
            with self.subTest(content=content):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ConfigurationError):
                    self.build(reference_id=None, reference_fasta=path)
        self.backend.assert_not_called()

    def test_invalid_fastas_fail_before_inference(self):
        for content in (
            "",
            ">ref\nACGT\n>001\nACG\n",
            ">ref\nACGT\n>001\nACGT\n",
            ">ref\nACGT\n>001\nACGT\n>001\nACGT\n",
            ">ref\nACGT\n>001\nACGZ\n",
            ">ref\n\n",
            ">\nACGT\n",
        ):
            with self.subTest(content=content):
                self.alignment.write_text(content, encoding="utf-8")
                with self.assertRaises(ConfigurationError):
                    self.build()
        with self.assertRaisesRegex(ConfigurationError, "Cannot read aligned FASTA"):
            build_phylogenetic_tree_from_fasta(self.directory / "missing.fasta", reference_id="ref")
        self.backend.assert_not_called()
        self.assertFalse(self.output_dir.exists())

    def test_invalid_or_mismatched_dates_fail_before_inference(self):
        sources = [
            None,
            {"001": 10},
            {**self.dates, "extra": 40},
            {1: 10, "002": 20, "NA": 30},
            {"": 10, "002": 20, "NA": 30},
        ]
        sources += [
            {**self.dates, "001": value}
            for value in (
                "2024-02-30",
                "2024-02-28",
                "not-a-date",
                float("nan"),
                "inf",
                True,
                None,
                datetime(2024, 2, 28, 12),
            )
        ]
        for source in sources:
            with self.subTest(source=source), self.assertRaises(ConfigurationError):
                self.build(dates=source)
        self.backend.assert_not_called()

    def test_invalid_dates_files_fail_before_inference(self):
        path = self.directory / "dates.csv"
        for content in (
            "",
            "id,date\n001,10\n",
            "case_id,sample_date\n001,10\n001,20\n002,20\nNA,30\n",
            'case_id,sample_date\n"unterminated',
        ):
            with self.subTest(content=content):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ConfigurationError):
                    self.build(dates=path)
        with self.assertRaisesRegex(ConfigurationError, "Cannot read sampling dates"):
            self.build(dates=self.directory / "missing.csv")
        self.backend.assert_not_called()


@unittest.skipUnless(
    IQTREE and Phylo is not None, "Set EPILINK_IQTREE or install IQ-TREE for integration tests"
)
class TestIQTreeIntegration(unittest.TestCase):
    def test_real_fasta_inference_with_calendar_and_separate_reference(self):
        base = "ACGT" * 250
        sample_ids = ["001", "002", "NA", "004"]
        sequences = {}
        for index, name in enumerate(sample_ids):
            sequence = list(base)
            for site in range((index + 1) * 20):
                sequence[site] = "ACGT"[("ACGT".index(sequence[site]) + 1) % 4]
            sequences[name] = "".join(sequence) + "N-?R"
        reference_sequence = base + "N-?R"
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            alignment = directory / "alignment.fasta"
            reference = directory / "reference.fasta"
            reference.write_text(f">ref\n{reference_sequence}\n", encoding="utf-8")
            for inside in (True, False):
                with self.subTest(reference_inside=inside):
                    text = "".join(f">{name}\n{seq}\n" for name, seq in sequences.items())
                    if inside:
                        text += f">ref\n{reference_sequence}\n"
                    alignment.write_text(text, encoding="utf-8")
                    numeric_dates = {name: float(i * 10) for i, name in enumerate(sample_ids)}
                    dates = (
                        {
                            name: (date(2024, 2, 28) + timedelta(days=value)).isoformat()
                            for name, value in numeric_dates.items()
                        }
                        if inside
                        else numeric_dates
                    )
                    result = build_phylogenetic_tree_from_fasta(
                        alignment,
                        reference_id="ref" if inside else None,
                        reference_fasta=None if inside else reference,
                        dates=dates,
                        model="JC",
                        clock_rate=0.002,
                        output_dir=directory / "output with spaces",
                        iqtree_executable=IQTREE,
                        timeout=60,
                    )
                    self.assertEqual(result.sample_ids, tuple(sample_ids))
                    self.assertEqual(
                        {tip.name for tip in result.dated_tree.get_terminals()}, set(sample_ids)
                    )
                    tips = result.node_dates[result.node_dates.is_tip].set_index("node")
                    np.testing.assert_allclose(tips.date, tips.sample_date, atol=1e-4)
                    self.assertEqual(tips.sample_date.to_dict(), numeric_dates)
                    self.assertAlmostEqual(result.clock_rate, 0.002)
                    self.assertEqual(result.date_origin, date(2024, 2, 28) if inside else None)
                    if inside:
                        self.assertEqual(
                            tips.loc["002", "sample_calendar_date"], pd.Timestamp("2024-03-09")
                        )

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
