"""Replot saved benchmark measurements: python -m docs.plot_benchmarks RUN_DIR."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Benchmark figures require Matplotlib: pip install -e '.[benchmark]'."
        ) from error
    return plt


def _curve(axis, data: pd.DataFrame, x: str, label: str, color: str, *, scale=1.0):
    data = data.sort_values(x)
    axis.plot(
        data[x],
        data.median_seconds * scale,
        "o-",
        label=label,
        color=color,
        linewidth=1.6,
        markersize=4,
    )
    axis.fill_between(
        data[x], data.q25_seconds * scale, data.q75_seconds * scale, color=color, alpha=0.16
    )


def paired_speedups(trials: pd.DataFrame) -> pd.DataFrame:
    """Median/IQR of per-trial ratios, not a ratio of unmatched summary statistics."""
    matched = trials[trials.operation.isin(["target_scalar_loop", "target_batch"])]
    rows = []
    for count, group in matched.groupby("observations"):
        paired = group.pivot(index="repeat", columns="operation", values="seconds_per_call")
        ratios = paired.target_scalar_loop / paired.target_batch
        if ratios.isna().any():
            raise ValueError("Each scoring trial needs both scalar and batch timings.")
        q25, median, q75 = np.quantile(ratios, [0.25, 0.5, 0.75])
        rows.append(
            {
                "observations": count,
                "median_seconds": median,
                "q25_seconds": q25,
                "q75_seconds": q75,
            }
        )
    return pd.DataFrame(rows)


def _initialization(plt, summary: pd.DataFrame, metadata: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    labels = {
        "cold_profile_model": "Cold profile + model",
        "warm_model": "Model, primed profile",
        "scorer_construction": "First target-scorer construction",
    }
    for axis, sweep, xlabel in zip(
        axes,
        ("mc_samples", "maximum_depth"),
        ("Monte Carlo draws / scenario", "Maximum scenario depth"),
        strict=True,
    ):
        for (operation, label), color in zip(labels.items(), COLORS, strict=False):
            data = summary[(summary.operation == operation) & (summary.sweep == sweep)]
            if not data.empty:
                _curve(axis, data, sweep, label, color)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Time per construction (seconds)")
        axis.set_yscale("log")
        if sweep == "mc_samples":
            axis.set_xscale("log")
        else:
            axis.set_xticks(sorted(summary.loc[summary.sweep == sweep, sweep].unique()))
        axis.legend(fontsize=8)
        axis.grid(True, which="major", alpha=0.2)
    fig.suptitle("EpiLink initialization scaling — median and interquartile range")
    return fig


def _scoring(plt, summary: pd.DataFrame, trials: pd.DataFrame, metadata: dict):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    matched = summary[summary.operation.isin(["target_scalar_loop", "target_batch"])]
    labels = {
        "target_scalar_loop": "Scalar loop (same targets)",
        "target_batch": "Vectorized batch (same targets)",
    }
    for (operation, label), color in zip(labels.items(), COLORS, strict=False):
        data = matched[matched.operation == operation].sort_values("observations")
        _curve(axes[0, 0], data, "observations", label, color)
        x = data.observations
        axes[0, 1].plot(x, x / data.median_seconds, "o-", label=label, color=color, markersize=4)
        axes[0, 1].fill_between(
            x, x / data.q75_seconds, x / data.q25_seconds, color=color, alpha=0.16
        )
    for axis, ylabel in zip(
        axes[0],
        ("Time for all observations (seconds)", "Throughput (observations / second)"),
        strict=True,
    ):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Number of paired observations")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
    speedups = paired_speedups(trials)
    _curve(axes[1, 0], speedups, "observations", "Paired scalar / batch ratio", COLORS[2])
    axes[1, 0].axhline(1.0, linestyle="--", color="gray", linewidth=1)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Number of paired observations")
    axes[1, 0].set_ylabel("Matched-workload speedup (×)")
    detailed = summary[summary.operation == "detailed_score_pair"]
    _curve(axes[1, 1], detailed, "mc_samples", "Detailed score_pair", COLORS[3], scale=1000)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Monte Carlo draws / scenario")
    axes[1, 1].set_ylabel("Detailed single-pair latency (milliseconds)")
    scenarios = int(detailed.scenario_count.iloc[0])
    axes[1, 1].set_title(f"All {scenarios} scenarios; independent workload", fontsize=10)
    for axis in axes.flat:
        axis.grid(True, which="major", alpha=0.2)
    targets = ", ".join(metadata["targets"])
    fig.suptitle(f"Warm scoring — matched target subset: {targets}\nMedian and interquartile range")
    return fig


def _simulation(plt, summary: pd.DataFrame, metadata: dict):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    for axis, operation, title in zip(
        axes,
        ("epidemic_dates", "genomic_sequences", "pairwise_table"),
        (
            "Epidemic-date simulation",
            "Genomic simulation (both outputs)",
            "Pairwise distance table (both outputs)",
        ),
        strict=True,
    ):
        data = summary[summary.operation == operation]
        for index, (length, group) in enumerate(data.groupby("genome_length")):
            label = (
                "Primed transmission profile"
                if operation == "epidemic_dates"
                else f"{int(length):,} sites"
            )
            _curve(axis, group, "tree_nodes", label, COLORS[index % len(COLORS)])
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Number of cases")
        axis.set_ylabel("Time per call (seconds)")
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.legend(fontsize=8)
        axis.grid(True, which="major", alpha=0.2)
    fig.suptitle("Simulation scaling — median and interquartile range")
    return fig


def plot_run(
    run_dir: str | Path, output_dir: str | Path | None = None, *, dpi: int = 200
) -> list[Path]:
    """Read saved records and regenerate only figures; measurement files are untouched."""
    run_dir = Path(run_dir).expanduser().resolve()
    trials = pd.read_csv(run_dir / "raw_trials.csv", keep_default_na=False)
    summary = pd.read_csv(run_dir / "summary.csv", keep_default_na=False)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark schema version.")
    required = {"operation", "sweep", "median_seconds", "q25_seconds", "q75_seconds"}
    if summary.empty or not required <= set(summary.columns):
        raise ValueError("Benchmark summary is empty or missing timing columns.")
    if (summary.median_seconds <= 0).any() or not np.isfinite(summary.median_seconds).all():
        raise ValueError("Benchmark timings must be finite and positive.")
    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    output = (
        Path(output_dir).expanduser().resolve() if output_dir is not None else run_dir / "figures"
    )
    output.mkdir(parents=True, exist_ok=True)
    plt = require_matplotlib()
    paths = []
    with plt.rc_context(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    ):
        builders = (
            (
                "initialization_scaling",
                "cold_profile_model",
                lambda: _initialization(plt, summary, metadata),
            ),
            (
                "scoring_performance",
                "target_batch",
                lambda: _scoring(plt, summary, trials, metadata),
            ),
            ("simulation_scaling", "epidemic_dates", lambda: _simulation(plt, summary, metadata)),
        )
        for filename, operation, build in builders:
            if operation not in set(summary.operation):
                continue
            fig = build()
            try:
                for extension in ("png", "pdf"):
                    path = output / f"{filename}.{extension}"
                    fig.savefig(path, dpi=dpi)
                    paths.append(path)
            finally:
                plt.close(fig)
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)
    for path in plot_run(args.run_dir, args.output_dir, dpi=args.dpi):
        print(path)


if __name__ == "__main__":
    main()
