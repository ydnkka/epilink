#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence

import numpy as np

from .transmission_linkage_model import (
    estimate_linkage_probabilities,
    pairwise_linkage_probability_matrix,
)


def _parse_intermediates(val: str) -> tuple[int, ...]:
    """Parse comma-separated list of non-negative ints, e.g. '0,1,2' -> (0, 1, 2)"""
    try:
        parts = [p.strip() for p in val.split(",") if p.strip() != ""]
        out = tuple(int(p) for p in parts)
        if any(x < 0 for x in out):
            raise ValueError
        return out
    except Exception as err:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated non-negative integers, e.g. '0,1,2'."
        ) from err


def _add_common_options(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--intermediate-generations",
        "-m",
        type=_parse_intermediates,
        default=(0,),
        help="Comma-separated list of intermediate generation counts to "
        "include in mixture, e.g. '0,1,2' (default: 0)",
    )
    p.add_argument(
        "--no-intermediates",
        "-M",
        type=int,
        default=10,
        help="Maximum number of intermediates used in simulations/kernel (default: 10)",
    )
    p.add_argument(
        "--num-simulations",
        "--nsims",
        type=int,
        default=10000,
        help="Monte Carlo draws (default: 10000)",
    )
    p.add_argument(
        "--subs-rate",
        type=float,
        default=1e-3,
        help="Substitution rate per site per year (default: 1e-3)",
    )
    p.add_argument(
        "--subs-rate-sigma",
        type=float,
        default=0.33,
        help="Lognormal sigma for relaxed clock (default: 0.33)",
    )
    p.add_argument(
        "--relax-rate",
        action="store_true",
        help="Use relaxed molecular clock (default: off)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed (default: 12345)",
    )


def cmd_point(args: argparse.Namespace) -> int:
    """Estimate probability for one or more (g, t) pairs."""
    # Convert inputs to floats
    try:
        g_vals: list[float] = [float(x) for x in args.genetic_distance]
        t_vals: list[float] = [float(x) for x in args.sampling_interval]
    except ValueError:
        sys.exit("Error: genetic_distance and sampling_interval must be numeric")

    if any(x < 0 for x in g_vals + t_vals):
        sys.exit("Error: genetic_distance and sampling_interval must be non-negative")

    if args.num_simulations <= 0:
        sys.exit("Error: num_simulations must be greater than 0")

    if len(g_vals) != len(t_vals):
        sys.exit("Error: genetic_distance and sampling_interval must have same count")

    p = estimate_linkage_probabilities(
        genetic_distance=np.array(g_vals, dtype=float),
        sampling_interval=np.array(t_vals, dtype=float),
        intermediate_generations=args.intermediate_generations,
        no_intermediates=args.no_intermediates,
        subs_rate=args.subs_rate,
        subs_rate_sigma=args.subs_rate_sigma,
        relax_rate=args.relax_rate,
        num_simulations=args.num_simulations,
        rng_seed=args.seed,
    )

    # Output
    p = np.atleast_1d(p)
    if len(g_vals) == 1:
        print(p.item())

    else:
        writer = csv.writer(sys.stdout)
        if not getattr(args, "no_header", False):
            writer.writerow(["g", "t", "p"])
        for gi, ti, pi in zip(g_vals, t_vals, p):
            writer.writerow([gi, ti, float(pi)])
    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    """Estimate P(link) over a grid of g and t and output CSV."""

    # Validate steps
    if args.g_step <= 0 or args.t_step <= 0:
        sys.exit("Error: step sizes must be positive")

    # Validate range
    if args.g_stop < args.g_start or args.t_stop < args.t_start:
        sys.exit("Error: stop values must be >= start values")

    g = np.arange(args.g_start, args.g_stop + 1e-12, args.g_step, dtype=float)
    t = np.arange(args.t_start, args.t_stop + 1e-12, args.t_step, dtype=float)

    mat = pairwise_linkage_probability_matrix(
        genetic_distances=g,
        temporal_distances=t,
        intermediate_generations=args.intermediate_generations,
        no_intermediates=args.no_intermediates,
        subs_rate=args.subs_rate,
        subs_rate_sigma=args.subs_rate_sigma,
        relax_rate=args.relax_rate,
        num_simulations=args.num_simulations,
        rng_seed=args.seed,
    )

    # Write CSV as rows (g, t, p)
    fh = sys.stdout if args.out == "-" else open(args.out, "w", newline="", encoding="utf-8")
    close = fh is not sys.stdout
    try:
        writer = csv.writer(fh)
        if not getattr(args, "no_header", False):
            writer.writerow(["g", "t", "p"])
        for i, gi in enumerate(g):
            for j, tj in enumerate(t):
                writer.writerow([float(gi), float(tj), float(mat[i, j])])
    finally:
        if close:
            fh.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="epilink",
        description="Estimate transmission linkage probabilities from genetic and temporal distances.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # point subcommand
    p_point = sub.add_parser("point", help="Estimate P(link) for one or more (g, t) pairs")
    p_point.add_argument(
        "--genetic-distance",
        "-g",
        nargs="+",
        required=True,
        help="Genetic distance(s) in SNPs (space-separated for multiple).",
    )
    p_point.add_argument(
        "--sampling-interval",
        "-t",
        nargs="+",
        required=True,
        help="Temporal distance(s) in days (space-separated, must match number of genetic distances).",
    )
    p_point.add_argument(
        "--no-header", action="store_true", help="Do not print CSV header for multiple pairs"
    )
    _add_common_options(p_point)
    p_point.set_defaults(func=cmd_point)

    # grid subcommand
    p_grid = sub.add_parser("grid", help="Estimate P(link) over a grid of g and t and output CSV")
    p_grid.add_argument(
        "--g-start", type=float, required=True, help="Start genetic distance (inclusive)"
    )
    p_grid.add_argument(
        "--g-stop", type=float, required=True, help="Stop genetic distance (inclusive)"
    )
    p_grid.add_argument(
        "--g-step", type=float, default=1.0, help="Genetic distance step (default: 1.0)"
    )
    p_grid.add_argument(
        "--t-start", type=float, required=True, help="Start temporal distance (inclusive)"
    )
    p_grid.add_argument(
        "--t-stop", type=float, required=True, help="Stop temporal distance (inclusive)"
    )
    p_grid.add_argument(
        "--t-step", type=float, default=1.0, help="Temporal distance step (default: 1.0)"
    )
    p_grid.add_argument(
        "--out", default="-", help="Output file path (CSV). Use '-' for stdout (default)."
    )
    p_grid.add_argument("--no-header", action="store_true", help="Do not print CSV header")
    _add_common_options(p_grid)
    p_grid.set_defaults(func=cmd_grid)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
