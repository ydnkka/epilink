"""
Command-line interface for epilink.
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from .infectiousness_profile import TOIT, MolecularClock
from .transmission_linkage_model import linkage_probability, linkage_probability_matrix


def _parse_intermediate_generations(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("intermediate_generations cannot be empty.")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "intermediate_generations must be comma-separated integers, e.g. '0,1,2'."
        ) from exc


def _normalize_intermediate_generations(value: tuple[int, ...] | str) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    return _parse_intermediate_generations(value)


def _build_models(args: argparse.Namespace) -> tuple[TOIT, MolecularClock]:
    toit = TOIT(rng_seed=args.seed)
    clock = MolecularClock(
        subs_rate=args.subs_rate,
        relax_rate=args.relax_rate,
        subs_rate_sigma=args.subs_rate_sigma,
        gen_len=args.gen_len,
        rng_seed=args.seed,
    )
    return toit, clock


def _write_point_results(
    out_stream,
    genetic_distances: np.ndarray,
    temporal_distances: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    writer = csv.writer(out_stream)
    writer.writerow(["GeneticDistance", "TemporalDistance", "Probability"])
    for g, t, p in zip(genetic_distances, temporal_distances, probabilities):
        writer.writerow([g, t, p])


def _write_grid_results(
    out_stream,
    genetic_distances: np.ndarray,
    temporal_distances: np.ndarray,
    matrix: np.ndarray,
) -> None:
    writer = csv.writer(out_stream)
    header = ["GeneticDistance"] + [str(t) for t in temporal_distances]
    writer.writerow(header)
    for i, g in enumerate(genetic_distances):
        row = [g] + [matrix[i, j] for j in range(matrix.shape[1])]
        writer.writerow(row)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--nsims", type=int, default=10000, help="Number of Monte Carlo draws.")
    parser.add_argument(
        "--no-intermediates",
        type=int,
        default=10,
        help="Maximum number of intermediate hosts (M).",
    )
    parser.add_argument(
        "-m",
        "--intermediate-generations",
        type=_parse_intermediate_generations,
        default="0,1",
        help="Comma-separated intermediate generations to include, e.g. '0,1,2'.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for simulations.")
    parser.add_argument(
        "--subs-rate", type=float, default=1e-3, help="Substitution rate per site/year."
    )
    parser.add_argument(
        "--subs-rate-sigma",
        type=float,
        default=0.33,
        help="Lognormal dispersion for relaxed clock.",
    )
    parser.add_argument("--gen-len", type=int, default=29903, help="Genome length in sites.")
    parser.add_argument(
        "--strict-rate",
        dest="relax_rate",
        action="store_false",
        help="Use a strict (constant) molecular clock.",
    )
    parser.add_argument(
        "--relax-rate",
        dest="relax_rate",
        action="store_true",
        help="Use a relaxed (lognormal) molecular clock (default).",
    )
    parser.set_defaults(relax_rate=True)


def _handle_point(args: argparse.Namespace) -> int:
    toit, clock = _build_models(args)
    intermediate_generations = _normalize_intermediate_generations(args.intermediate_generations)

    genetic = np.asarray(args.genetic_distance, dtype=float)
    temporal = np.asarray(args.temporal_distance, dtype=float)
    if genetic.size != temporal.size:
        raise SystemExit("genetic_distance and temporal_distance must have the same length.")

    probs = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=genetic,
        temporal_distance=temporal,
        intermediate_generations=intermediate_generations,
        no_intermediates=args.no_intermediates,
        num_simulations=args.nsims,
        cache_unique_distances=True,
    )
    probs = np.atleast_1d(np.asarray(probs, dtype=float))

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            _write_point_results(handle, genetic, temporal, probs)
    else:
        _write_point_results(sys.stdout, genetic, temporal, probs)
    return 0


def _handle_grid(args: argparse.Namespace) -> int:
    if args.g_step <= 0 or args.t_step <= 0:
        raise SystemExit("g-step and t-step must be positive.")

    genetic = np.arange(args.g_start, args.g_stop, args.g_step, dtype=float)
    temporal = np.arange(args.t_start, args.t_stop, args.t_step, dtype=float)
    if genetic.size == 0 or temporal.size == 0:
        raise SystemExit("g-start/g-stop or t-start/t-stop produce an empty grid.")

    toit, clock = _build_models(args)
    intermediate_generations = _normalize_intermediate_generations(args.intermediate_generations)

    matrix = linkage_probability_matrix(
        toit=toit,
        clock=clock,
        genetic_distances=genetic,
        temporal_distances=temporal,
        intermediate_generations=intermediate_generations,
        no_intermediates=args.no_intermediates,
        num_simulations=args.nsims,
    )

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            _write_grid_results(handle, genetic, temporal, matrix)
    else:
        _write_grid_results(sys.stdout, genetic, temporal, matrix)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="epilink",
        description="Estimate linkage probabilities from genetic and temporal distances.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    point = subparsers.add_parser("point", help="Estimate linkage probability for pairs.")
    point.add_argument(
        "-g",
        "--genetic-distance",
        type=float,
        nargs="+",
        required=True,
        help="One or more genetic distances (SNPs).",
    )
    point.add_argument(
        "-t",
        "--temporal-distance",
        type=float,
        nargs="+",
        required=True,
        help="One or more temporal distances (days).",
    )
    point.add_argument("--out", help="Optional output CSV path (defaults to stdout).")
    _add_common_args(point)
    point.set_defaults(func=_handle_point)

    grid = subparsers.add_parser("grid", help="Estimate linkage probabilities on a grid.")
    grid.add_argument(
        "--g-start", type=float, required=True, help="Genetic distance start (inclusive)."
    )
    grid.add_argument(
        "--g-stop", type=float, required=True, help="Genetic distance stop (exclusive)."
    )
    grid.add_argument("--g-step", type=float, required=True, help="Genetic distance step.")
    grid.add_argument(
        "--t-start", type=float, required=True, help="Temporal distance start (inclusive)."
    )
    grid.add_argument(
        "--t-stop", type=float, required=True, help="Temporal distance stop (exclusive)."
    )
    grid.add_argument("--t-step", type=float, required=True, help="Temporal distance step.")
    grid.add_argument("--out", help="Optional output CSV path (defaults to stdout).")
    _add_common_args(grid)
    grid.set_defaults(func=_handle_grid)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
