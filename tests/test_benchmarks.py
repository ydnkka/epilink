from __future__ import annotations

import io
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
from docs import benchmark_api as benchmark
from docs import plot_benchmarks as plots

from epilink import InfectiousnessToTransmission, NaturalHistoryParameters

try:
    import matplotlib
except ImportError:
    matplotlib = None


def small_profile(length, seed):
    return InfectiousnessToTransmission(
        parameters=NaturalHistoryParameters(genome_length=length),
        rng_seed=seed,
        integration_grid_points=32,
        sampling_grid_points=16,
    )


SMALL = benchmark.BenchmarkConfig(
    mc_samples=32,
    maximum_depth=1,
    genome_length=64,
    mc_sizes=(16, 32),
    depths=(0, 1),
    batch_sizes=(1, 4),
    tree_sizes=(3, 7),
    genome_sizes=(32, 64),
    repeats=2,
    warmups=0,
    min_time=0.0001,
)


class TestBenchmarkMeasurement(unittest.TestCase):
    def test_timer_excludes_setup_and_calibrates_only_repeatable_calls(self):
        clock = [0.0]

        def setup():
            clock[0] += 100.0

            def operation():
                clock[0] += 0.02
                return object()

            return operation

        cases = [
            benchmark.BenchmarkCase(benchmark._parameters("constructor", "test"), setup),
            benchmark.BenchmarkCase(
                benchmark._parameters("warm", "test"), setup, repeat_calls=True
            ),
        ]
        config = replace(SMALL, min_time=0.05, repeats=3)
        with patch.object(benchmark.time, "perf_counter", side_effect=lambda: clock[0]):
            trials = benchmark.measure_cases(cases, config)
        np.testing.assert_allclose(trials.seconds_per_call, 0.02)
        self.assertEqual(set(trials.loc[trials.operation == "constructor", "calls"]), {1})
        self.assertEqual(set(trials.loc[trials.operation == "warm", "calls"]), {4})
        self.assertEqual(len(trials), 6)

    def test_final_result_is_alive_until_after_stop_timestamp(self):
        clock = [0.0]

        class Result:
            def __del__(self):
                clock[0] += 50

        def operation():
            clock[0] += 1
            return Result()

        with patch.object(benchmark.time, "perf_counter", side_effect=lambda: clock[0]):
            elapsed, result = benchmark._elapsed(operation, 1)
        self.assertEqual(elapsed, 1)
        self.assertEqual(clock[0], 1)
        del result
        self.assertEqual(clock[0], 51)

    def test_initialization_cache_boundaries_and_seeded_draws(self):
        with patch.object(benchmark, "_profile", side_effect=small_profile):
            cases = benchmark.initialization_cases(SMALL)
            by_operation = {case.parameters["operation"]: case for case in cases[:3]}
            cold = by_operation["cold_profile_model"].setup()()
            warm = by_operation["warm_model"].setup()()
            for label in cold.draws_by_scenario:
                for name in cold.draws_by_scenario[label]:
                    np.testing.assert_array_equal(
                        cold.draws_by_scenario[label][name], warm.draws_by_scenario[label][name]
                    )
            another_cold = by_operation["cold_profile_model"].setup()()
            self.assertIsNot(cold.profile, another_cold.profile)
            first_sort = by_operation["scorer_construction"].setup()
            second_sort = by_operation["scorer_construction"].setup()
            self.assertIsNot(first_sort(), second_sort())

    def test_scalar_batch_comparison_has_identical_work_and_outputs(self):
        with patch.object(benchmark, "_profile", side_effect=small_profile):
            cases = benchmark.scoring_cases(SMALL)
        for index in (0, 2):
            scalar, batch = cases[index : index + 2]
            self.assertEqual(scalar.parameters["observations"], batch.parameters["observations"])
            self.assertEqual(scalar.parameters["target_count"], batch.parameters["target_count"])
            np.testing.assert_allclose(scalar.setup()(), batch.setup()())
        self.assertEqual(cases[-1].parameters["operation"], "detailed_score_pair")
        self.assertEqual(cases[-1].parameters["scenario_count"], 5)

    def test_simulation_shapes_matching_genomes_and_reproducible_inputs(self):
        original = benchmark.simulate_genomic_sequences

        def checked_simulation(profile, tree, *, genome_length, return_raw):
            self.assertEqual(profile.parameters.genome_length, genome_length)
            return original(profile, tree, genome_length=genome_length, return_raw=return_raw)

        with (
            patch.object(benchmark, "_profile", side_effect=small_profile),
            patch.object(benchmark, "simulate_genomic_sequences", side_effect=checked_simulation),
        ):
            cases = benchmark.simulation_cases(SMALL)
            for case in cases:
                first = case.setup()()
                second = case.setup()()
                nodes = case.parameters["tree_nodes"]
                if case.parameters["operation"] == "genomic_sequences":
                    self.assertIsNone(first.raw)
                    self.assertEqual(
                        first.packed.stochastic.original_length, case.parameters["genome_length"]
                    )
                    self.assertEqual(first.packed.stochastic.n_seqs, nodes)
                    np.testing.assert_array_equal(
                        first.packed.stochastic.packed_u64, second.packed.stochastic.packed_u64
                    )
                elif case.parameters["operation"] == "pairwise_table":
                    self.assertEqual(len(first), nodes * (nodes - 1) // 2)
                    pd.testing.assert_frame_equal(first, second)
                else:
                    self.assertEqual(dict(first.nodes(data=True)), dict(second.nodes(data=True)))
        for nodes in (1, 13, 63):
            tree = benchmark._build_tree(nodes)
            self.assertEqual(len(tree), nodes)
            self.assertEqual(tree.number_of_edges(), nodes - 1)
            self.assertTrue(nx.is_arborescence(tree))

    def test_statistics_and_paired_speedups(self):
        parameters = benchmark._parameters(
            "target_batch", "observations", observations=6, work_items=6, work_unit="observations"
        )
        frame = pd.DataFrame(
            [{**parameters, "case_id": "a", "seconds_per_call": value} for value in (1, 2, 3, 4, 5)]
        )
        summary = benchmark.summarize_trials(frame).iloc[0]
        self.assertEqual(summary.median_seconds, 3)
        self.assertEqual(summary.q25_seconds, 2)
        self.assertEqual(summary.q75_seconds, 4)
        self.assertEqual(summary.work_items_per_second, 2)
        rows = []
        for repeat, (scalar, batch) in enumerate(((10, 1), (12, 3), (8, 2))):
            rows += [
                {
                    "repeat": repeat,
                    "observations": 6,
                    "operation": "target_scalar_loop",
                    "seconds_per_call": scalar,
                },
                {
                    "repeat": repeat,
                    "observations": 6,
                    "operation": "target_batch",
                    "seconds_per_call": batch,
                },
            ]
        ratios = plots.paired_speedups(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(ratios.median_seconds, 4)  # Ratio of medians would incorrectly give 5.

    def test_argument_validation_presets_and_legacy_flags(self):
        quick, _ = benchmark.parse_args([])
        full, _ = benchmark.parse_args(["--preset", "full", "--repeats", "7"])
        self.assertGreater(max(full.batch_sizes), max(quick.batch_sizes))
        self.assertEqual(full.repeats, 7)
        legacy, _ = benchmark.parse_args(["--tree-nodes", "13", "--grid-size", "10"])
        self.assertEqual(legacy.tree_sizes, (13,))
        self.assertEqual(legacy.batch_sizes, (100,))
        for args in (
            ["--repeats", "0"],
            ["--min-time", "nan"],
            ["--mc-sizes", "-1"],
            ["--warmups", "-1"],
            ["--tree-nodes", "3", "--tree-sizes", "7"],
        ):
            with (
                self.subTest(args=args),
                patch("sys.stderr", new_callable=io.StringIO),
                self.assertRaises(SystemExit),
            ):
                benchmark.parse_args(args)


@unittest.skipUnless(matplotlib is not None, "Install epilink[benchmark] for figure tests")
class TestBenchmarkArtifacts(unittest.TestCase):
    def test_end_to_end_cli_and_replot_preserve_measurements(self):
        with TemporaryDirectory() as temp:
            args = [
                "--output-dir",
                temp,
                "--mc-samples",
                "32",
                "--mc-sizes",
                "16",
                "32",
                "--maximum-depth",
                "1",
                "--depths",
                "0",
                "1",
                "--genome-length",
                "64",
                "--batch-sizes",
                "1",
                "4",
                "--tree-sizes",
                "3",
                "7",
                "--genome-sizes",
                "32",
                "64",
                "--repeats",
                "2",
                "--warmups",
                "0",
                "--min-time",
                "0.0001",
            ]
            with (
                patch.object(benchmark, "_profile", side_effect=small_profile),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                benchmark.main(args)
            run_dir = next(Path(temp).glob("run-*"))
            trials = pd.read_csv(run_dir / "raw_trials.csv")
            summary = pd.read_csv(run_dir / "summary.csv")
            metadata = json.loads((run_dir / "metadata.json").read_text())
            self.assertEqual(len(trials), len(summary) * 2)
            self.assertTrue((trials.calls >= 1).all())
            self.assertTrue((summary.median_seconds > 0).all())
            self.assertEqual(metadata["targets"], list(benchmark.TARGETS))
            self.assertEqual(metadata["config"]["repeats"], 2)
            self.assertIn("numpy", metadata["versions"])
            self.assertIn("thread_environment", metadata)
            self.assertEqual(metadata["profile_parameters"]["genome_length"], 64)
            self.assertEqual(metadata["profile_grid"]["sampling_points"], 16)
            before = {
                name: (run_dir / name).read_bytes()
                for name in ("raw_trials.csv", "summary.csv", "metadata.json")
            }
            output = Path(temp) / "replotted"
            files = plots.plot_run(run_dir, output, dpi=60)
            self.assertEqual(len(files), 6)
            plt = plots.require_matplotlib()
            for path in files:
                if path.suffix == ".png":
                    pixels = plt.imread(path)
                    self.assertGreater(pixels.shape[0], 100)
                    self.assertGreater(np.std(pixels), 0)
                else:
                    self.assertTrue(path.read_bytes().startswith(b"%PDF"))
            for name, content in before.items():
                self.assertEqual((run_dir / name).read_bytes(), content)
            self.assertEqual(plt.get_fignums(), [])

    def test_scoring_only_creates_only_its_figures(self):
        with TemporaryDirectory() as temp:
            with (
                patch.object(benchmark, "_profile", side_effect=small_profile),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                benchmark.main(
                    [
                        "--output-dir",
                        temp,
                        "--sections",
                        "scoring",
                        "--mc-samples",
                        "16",
                        "--mc-sizes",
                        "16",
                        "--genome-length",
                        "32",
                        "--batch-sizes",
                        "1",
                        "2",
                        "--repeats",
                        "1",
                        "--warmups",
                        "0",
                        "--min-time",
                        "0.0001",
                    ]
                )
            files = list(Path(temp).glob("run-*/figures/*"))
            self.assertEqual(
                {path.name for path in files},
                {"scoring_performance.png", "scoring_performance.pdf"},
            )


if __name__ == "__main__":
    unittest.main()
