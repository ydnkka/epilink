"""Reproducible API benchmarks with CSV measurements and automatic figures.

Run from the checkout: python -m docs.benchmark_api --preset quick
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from epilink import (
    EpiLink,
    InfectiousnessToTransmission,
    NaturalHistoryParameters,
    PairwiseCompatibilityModel,
    build_pairwise_case_table,
    simulate_epidemic_dates,
    simulate_genomic_sequences,
)

TARGETS = ("ad(0)", "ca(0,0)")
CASE_COLUMNS = [
    "case_id",
    "operation",
    "sweep",
    "mc_samples",
    "maximum_depth",
    "scenario_count",
    "target_count",
    "genome_length",
    "tree_nodes",
    "observations",
    "work_items",
    "work_unit",
]
PRESETS = {
    "quick": {
        "mc_sizes": (2000, 10000, 20000),
        "depths": (0, 1, 2, 3),
        "batch_sizes": (1, 10, 100, 1000),
        "tree_sizes": (15, 31, 63, 127),
        "genome_sizes": (500, 2000),
        "repeats": 5,
        "min_time": 0.05,
    },
    "full": {
        "mc_sizes": (2000, 10000, 50000, 100000),
        "depths": (0, 1, 2, 4, 6),
        "batch_sizes": (1, 10, 100, 1000, 10000, 100000),
        "tree_sizes": (15, 63, 255, 511),
        "genome_sizes": (500, 5000, 29903),
        "repeats": 10,
        "min_time": 0.2,
    },
}


@dataclass(frozen=True)
class BenchmarkConfig:
    preset: str = "quick"
    mc_samples: int = 20000
    maximum_depth: int = 2
    genome_length: int = 29903
    mc_sizes: tuple[int, ...] = (2000, 10000, 20000)
    depths: tuple[int, ...] = (0, 1, 2, 3)
    batch_sizes: tuple[int, ...] = (1, 10, 100, 1000)
    tree_sizes: tuple[int, ...] = (15, 31, 63, 127)
    genome_sizes: tuple[int, ...] = (500, 2000)
    repeats: int = 5
    warmups: int = 2
    min_time: float = 0.05
    rng_seed: int = 2026
    sections: tuple[str, ...] = ("initialization", "scoring", "simulation")


@dataclass(frozen=True)
class BenchmarkCase:
    parameters: dict[str, object]
    setup: Callable[[], Callable[[], object]]
    repeat_calls: bool = False


def _profile(genome_length: int, seed: int) -> InfectiousnessToTransmission:
    return InfectiousnessToTransmission(
        parameters=NaturalHistoryParameters(genome_length=genome_length), rng_seed=seed
    )


def _seed_profile(profile: InfectiousnessToTransmission, seed: int) -> InfectiousnessToTransmission:
    profile.rng = np.random.default_rng(seed)
    return profile


def _model(profile: InfectiousnessToTransmission, mc_samples: int, depth: int) -> EpiLink:
    return EpiLink(
        transmission_profile=profile,
        maximum_depth=depth,
        mc_samples=mc_samples,
        target=TARGETS,
        mutation_process="stochastic",
    )


def _build_tree(node_count: int) -> nx.DiGraph:
    """Construct exactly N nodes without an oversized intermediate tree."""
    names = [f"case-{i}" for i in range(node_count)]
    tree = nx.DiGraph()
    tree.add_nodes_from(names)
    tree.add_edges_from((names[(i - 1) // 2], names[i]) for i in range(1, node_count))
    return tree


def _parameters(operation: str, sweep: str, **values: object) -> dict[str, object]:
    return {
        "operation": operation,
        "sweep": sweep,
        "mc_samples": 0,
        "maximum_depth": 0,
        "scenario_count": 0,
        "target_count": len(TARGETS),
        "genome_length": 0,
        "tree_nodes": 0,
        "observations": 0,
        "work_items": 0,
        "work_unit": "",
        **values,
    }


def initialization_cases(config: BenchmarkConfig) -> list[BenchmarkCase]:
    cases = []
    points = [("mc_samples", m, config.maximum_depth) for m in config.mc_sizes]
    points += [("maximum_depth", config.mc_samples, d) for d in config.depths]
    for sweep, samples, depth in points:
        warm_profile = _profile(config.genome_length, config.rng_seed)
        warm_profile.cdf(0.0)  # Public API primes the numerical CDF without consuming RNG draws.
        common = {
            "mc_samples": samples,
            "maximum_depth": depth,
            "scenario_count": (depth + 1) * (depth + 4) // 2,
            "genome_length": config.genome_length,
        }

        def cold_setup(samples=samples, depth=depth):
            return lambda: _model(_profile(config.genome_length, config.rng_seed), samples, depth)

        def warm_setup(profile=warm_profile, samples=samples, depth=depth):
            profile = _seed_profile(profile, config.rng_seed)
            return lambda: _model(profile, samples, depth)

        def scorer_setup(profile=warm_profile, samples=samples, depth=depth):
            model = _model(_seed_profile(profile, config.rng_seed), samples, depth)
            return model.pairwise_model  # Fresh model: the sorting cache is empty on every trial.

        for operation, setup in (
            ("cold_profile_model", cold_setup),
            ("warm_model", warm_setup),
            ("scorer_construction", scorer_setup),
        ):
            cases.append(BenchmarkCase(_parameters(operation, sweep, **common), setup))
    return cases


def _score_scalar_loop(
    scorer: PairwiseCompatibilityModel, times: np.ndarray, distances: np.ndarray
) -> np.ndarray:
    return np.fromiter(
        (float(scorer(float(t), float(g))) for t, g in zip(times, distances, strict=True)),
        dtype=float,
        count=len(times),
    )


def scoring_cases(config: BenchmarkConfig) -> list[BenchmarkCase]:
    profile = _profile(config.genome_length, config.rng_seed)
    model = _model(profile, config.mc_samples, config.maximum_depth)
    scorer = model.pairwise_model()
    rng = np.random.default_rng(config.rng_seed + 1)
    times = rng.uniform(-30.0, 30.0, max(config.batch_sizes))
    distances = rng.poisson(5.0, max(config.batch_sizes))
    # A matched comparison: same observations, cached draws, targets and output array.
    np.testing.assert_allclose(
        _score_scalar_loop(scorer, times, distances), scorer(times, distances)
    )
    cases = []
    common = {
        "mc_samples": config.mc_samples,
        "maximum_depth": config.maximum_depth,
        "scenario_count": len(model.scenarios),
        "genome_length": config.genome_length,
    }
    for count in config.batch_sizes:
        t, g = times[:count], distances[:count]

        def scalar_setup(t=t, g=g):
            return lambda: _score_scalar_loop(scorer, t, g)

        def batch_setup(t=t, g=g):
            return lambda: scorer(t, g)

        for operation, setup in (
            ("target_scalar_loop", scalar_setup),
            ("target_batch", batch_setup),
        ):
            cases.append(
                BenchmarkCase(
                    _parameters(
                        operation,
                        "observations",
                        **common,
                        observations=count,
                        work_items=count,
                        work_unit="observations",
                    ),
                    setup,
                    repeat_calls=True,
                )
            )
    for samples in config.mc_sizes:
        detailed_model = _model(
            _profile(config.genome_length, config.rng_seed), samples, config.maximum_depth
        )
        detailed_model.pairwise_model()  # First-use sorting is excluded from warm detailed scoring.

        def detailed_setup(model=detailed_model):
            return lambda: model.score_pair(sample_time_difference=3.0, genetic_distance=2.0)

        cases.append(
            BenchmarkCase(
                _parameters(
                    "detailed_score_pair",
                    "mc_samples",
                    mc_samples=samples,
                    maximum_depth=config.maximum_depth,
                    scenario_count=len(detailed_model.scenarios),
                    genome_length=config.genome_length,
                    observations=1,
                    work_items=1,
                    work_unit="observations",
                ),
                detailed_setup,
                repeat_calls=True,
            )
        )
    return cases


def simulation_cases(config: BenchmarkConfig) -> list[BenchmarkCase]:
    cases = []
    for count in config.tree_sizes:
        tree = _build_tree(count)
        date_profile = _profile(config.genome_length, config.rng_seed)
        date_profile.cdf(0.0)

        def dates_setup(profile=date_profile, tree=tree):
            profile = _seed_profile(profile, config.rng_seed)
            return lambda: simulate_epidemic_dates(profile, tree, fraction_sampled=1.0)

        cases.append(
            BenchmarkCase(
                _parameters(
                    "epidemic_dates",
                    "tree_nodes",
                    tree_nodes=count,
                    genome_length=config.genome_length,
                    work_items=count,
                    work_unit="cases",
                ),
                dates_setup,
            )
        )
        for length in config.genome_sizes:
            profile = _profile(length, config.rng_seed)
            profile.cdf(0.0)
            epidemic_tree = simulate_epidemic_dates(profile, tree, fraction_sampled=1.0)
            packed = simulate_genomic_sequences(
                _seed_profile(profile, config.rng_seed + 2),
                epidemic_tree,
                genome_length=length,
                return_raw=False,
            )

            def sequences_setup(profile=profile, tree=epidemic_tree, length=length):
                profile = _seed_profile(profile, config.rng_seed + 2)
                return lambda: simulate_genomic_sequences(
                    profile, tree, genome_length=length, return_raw=False
                )

            def table_setup(packed=packed, tree=epidemic_tree):
                return lambda: build_pairwise_case_table(packed, tree)

            cases.append(
                BenchmarkCase(
                    _parameters(
                        "genomic_sequences",
                        "tree_nodes",
                        tree_nodes=count,
                        genome_length=length,
                        work_items=count,
                        work_unit="cases",
                    ),
                    sequences_setup,
                )
            )
            cases.append(
                BenchmarkCase(
                    _parameters(
                        "pairwise_table",
                        "tree_nodes",
                        tree_nodes=count,
                        genome_length=length,
                        work_items=count * (count - 1) // 2,
                        work_unit="pairs",
                    ),
                    table_setup,
                )
            )
    return cases


def _elapsed(fn: Callable[[], object], calls: int) -> tuple[float, object]:
    result = None
    start = time.perf_counter()
    for _ in range(calls):
        result = fn()
    elapsed = time.perf_counter() - start
    # Keep the last result alive until after the stop timestamp (not constructor + teardown).
    return elapsed, result


def measure_cases(
    cases: Sequence[BenchmarkCase],
    config: BenchmarkConfig,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Set up outside timing; interleave fresh trials in a seeded, shuffled order."""
    calls_by_case = []
    for case in cases:
        for _ in range(config.warmups):
            case.setup()()
        calls = 1
        if case.repeat_calls:
            fn = case.setup()
            while True:
                elapsed, result = _elapsed(fn, calls)
                del result
                if elapsed >= config.min_time:
                    break
                calls *= 2
            del fn
        calls_by_case.append(calls)
    records = []
    rng = np.random.default_rng(config.rng_seed + 3)
    for repeat in range(config.repeats):
        for order, index in enumerate(rng.permutation(len(cases))):
            case = cases[index]
            fn = case.setup()
            elapsed, result = _elapsed(fn, calls_by_case[index])
            del result, fn  # Outside the measured interval.
            records.append(
                {
                    **case.parameters,
                    "case_id": f"case-{index:04d}",
                    "repeat": repeat,
                    "order": order,
                    "calls": calls_by_case[index],
                    "elapsed_seconds": elapsed,
                    "seconds_per_call": elapsed / calls_by_case[index],
                }
            )
        if progress is not None:
            progress(f"Completed trial {repeat + 1}/{config.repeats} ({len(cases)} workloads)")
    return pd.DataFrame(records)


def summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in trials.groupby(CASE_COLUMNS, sort=False, dropna=False):
        values = group["seconds_per_call"].to_numpy(dtype=float)
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        row = dict(zip(CASE_COLUMNS, keys, strict=True))
        row.update(
            {
                "trials": len(values),
                "median_seconds": median,
                "q25_seconds": q25,
                "q75_seconds": q75,
                "mean_seconds": float(np.mean(values)),
                "min_seconds": float(np.min(values)),
                "work_items_per_second": (
                    float(row["work_items"]) / median if row["work_items"] else np.nan
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("case_id", ignore_index=True)


def run_benchmarks(config: BenchmarkConfig, progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    builders = {
        "initialization": initialization_cases,
        "scoring": scoring_cases,
        "simulation": simulation_cases,
    }
    cases = [case for section in config.sections for case in builders[section](config)]
    trials = measure_cases(cases, config, progress)
    return trials, summarize_trials(trials)


def environment_metadata(config: BenchmarkConfig) -> dict[str, object]:
    profile = _profile(config.genome_length, config.rng_seed)
    packages = {}
    for name in ("epilink", "numpy", "scipy", "pandas", "networkx", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unknown"
    git = {"commit": None, "dirty": None}
    try:
        cwd = Path(__file__).resolve().parents[1]
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=False
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=False
        )
        if revision.returncode == status.returncode == 0:
            git = {"commit": revision.stdout.strip(), "dirty": bool(status.stdout.strip())}
    except OSError:
        pass
    return {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "targets": list(TARGETS),
        "profile_parameters": profile.parameters.to_dict(),
        "profile_grid": {
            "sampling_points": profile.grid_points,
            "integration_points": profile.integration_grid_points,
            "minimum_days": profile.grid_min_days,
            "maximum_days": profile.grid_max_days,
        },
        "observations": {
            "time_distribution": "uniform",
            "minimum_days": -30.0,
            "maximum_days": 30.0,
            "genetic_distribution": "Poisson",
            "genetic_mean": 5.0,
            "seed": config.rng_seed + 1,
            "smaller_batches": "prefixes of the largest batch",
        },
        "versions": packages,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "git": git,
        "gc_enabled": gc.isenabled(),
        "timer_resolution_seconds": time.get_clock_info("perf_counter").resolution,
        "methodology": {
            "cold": "Fresh profile + numerical CDF + EpiLink construction; imports are excluded.",
            "warm_model": "Seed-reset profile with a primed numerical CDF; model construction only.",
            "scorer_construction": "Fresh model prepared outside timing; first pairwise_model call only.",
            "scoring": "Cached scorer; matched targets/observations/output; detailed scoring is separate.",
            "simulation": "Seed-reset inputs per trial; profile and sequence genome lengths match.",
            "statistics": "Median/IQR across trials; warm scoring trials average calibrated repeated calls.",
            "ordering": "Seeded shuffled workload order each trial; setup/warm-ups/reporting excluded.",
        },
    }


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def _positive_seconds(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def parse_args(argv: Sequence[str] | None = None) -> tuple[BenchmarkConfig, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="quick")
    parser.add_argument("--output-dir", type=Path, default=Path("build/benchmarks"))
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=("initialization", "scoring", "simulation"),
        default=list(BenchmarkConfig.sections),
    )
    parser.add_argument(
        "--mc-samples", type=_positive_int, default=20000, help="Baseline draws per scenario"
    )
    parser.add_argument(
        "--maximum-depth", type=_nonnegative_int, default=2, help="Baseline scenario depth"
    )
    parser.add_argument(
        "--genome-length", type=_positive_int, default=29903, help="Baseline profile genome length"
    )
    parser.add_argument("--mc-sizes", nargs="+", type=_positive_int, help="MC-sample sweep")
    parser.add_argument("--depths", nargs="+", type=_nonnegative_int, help="Scenario-depth sweep")
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=_positive_int,
        help="Number of paired observations per scoring call",
    )
    parser.add_argument("--tree-sizes", nargs="+", type=_positive_int, help="Case-count sweep")
    parser.add_argument(
        "--genome-sizes", nargs="+", type=_positive_int, help="Simulation genome-length sweep"
    )
    parser.add_argument(
        "--tree-nodes", type=_positive_int, help="Legacy shorthand for a single --tree-sizes value"
    )
    parser.add_argument(
        "--grid-size",
        type=_positive_int,
        help="Legacy shorthand for --batch-sizes SIZE**2; uses matched random observations",
    )
    parser.add_argument("--repeats", type=_positive_int)
    parser.add_argument("--warmups", type=_nonnegative_int, default=2)
    parser.add_argument(
        "--min-time",
        type=_positive_seconds,
        help="Minimum seconds in each calibrated warm-scoring pilot block",
    )
    parser.add_argument("--rng-seed", type=_nonnegative_int, default=2026)
    args = parser.parse_args(argv)
    if args.tree_nodes is not None and args.tree_sizes is not None:
        parser.error("use --tree-nodes or --tree-sizes, not both")
    if args.grid_size is not None and args.batch_sizes is not None:
        parser.error("use --grid-size or --batch-sizes, not both")
    preset = PRESETS[args.preset]
    sweeps = {
        name: tuple(sorted(set(getattr(args, name) or preset[name])))
        for name in (
            "mc_sizes",
            "depths",
            "batch_sizes",
            "tree_sizes",
            "genome_sizes",
        )
    }
    if args.tree_nodes is not None:
        sweeps["tree_sizes"] = (args.tree_nodes,)
    if args.grid_size is not None:
        sweeps["batch_sizes"] = (args.grid_size**2,)
    return (
        BenchmarkConfig(
            preset=args.preset,
            mc_samples=args.mc_samples,
            maximum_depth=args.maximum_depth,
            genome_length=args.genome_length,
            **sweeps,
            repeats=args.repeats or preset["repeats"],
            warmups=args.warmups,
            min_time=args.min_time or preset["min_time"],
            rng_seed=args.rng_seed,
            sections=tuple(dict.fromkeys(args.sections)),
        ),
        args.output_dir,
    )


def main(argv: Sequence[str] | None = None) -> None:
    # Support both python -m docs.benchmark_api and python docs/benchmark_api.py.
    if __package__:
        from .plot_benchmarks import plot_run, require_matplotlib
    else:
        from plot_benchmarks import plot_run, require_matplotlib
    config, output = parse_args(argv)
    require_matplotlib()
    metadata = environment_metadata(config)
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(tempfile.mkdtemp(prefix=f"run-{stamp}-", dir=output))
    print(f"Benchmark output: {run_dir}", flush=True)
    started = time.perf_counter()
    trials, summary = run_benchmarks(config, progress=lambda text: print(text, flush=True))
    metadata["benchmark_wall_seconds"] = time.perf_counter() - started
    trials.to_csv(run_dir / "raw_trials.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    paths = plot_run(run_dir)
    print(
        summary[
            [
                "operation",
                "sweep",
                "mc_samples",
                "maximum_depth",
                "observations",
                "tree_nodes",
                "genome_length",
                "median_seconds",
            ]
        ].to_string(index=False)
    )
    print(
        f"Saved raw_trials.csv, summary.csv, metadata.json and {len(paths)} figure files in {run_dir}"
    )


if __name__ == "__main__":
    main()
