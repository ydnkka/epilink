"""Command-line interface for epilink."""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from ..inference import estimate_linkage_probability, estimate_linkage_probability_grid
from ..model import InfectiousnessToTransmissionTime, MolecularClock, NaturalHistoryParameters


def _parse_included_intermediate_counts(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("included_intermediate_counts cannot be empty.")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "included_intermediate_counts must be comma-separated integers, e.g. '0,1,2'."
        ) from exc


def _normalize_included_intermediate_counts(
    value: tuple[int, ...] | str,
) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    return _parse_included_intermediate_counts(value)


def _build_models(
    args: argparse.Namespace,
) -> tuple[InfectiousnessToTransmissionTime, MolecularClock]:
    natural_history_parameters = NaturalHistoryParameters(
        incubation_shape=args.incubation_shape,
        incubation_scale=args.incubation_scale,
        latent_shape=args.latent_shape,
        symptomatic_rate=args.symptomatic_rate,
        symptomatic_shape=args.symptomatic_shape,
        rel_presymptomatic_infectiousness=args.rel_presymptomatic_infectiousness,
    )
    transmission_profile = InfectiousnessToTransmissionTime(
        grid_min_days=args.grid_min_days,
        grid_max_days=args.grid_max_days,
        parameters=natural_history_parameters,
        rng_seed=args.seed,
    )
    molecular_clock = MolecularClock(
        substitution_rate=args.substitution_rate,
        use_relaxed_clock=args.use_relaxed_clock,
        relaxed_clock_sigma=args.relaxed_clock_sigma,
        genome_length=args.genome_length,
        rng_seed=args.seed,
    )
    return transmission_profile, molecular_clock


def _write_point_results(
    out_stream,
    genetic_distances: np.ndarray,
    temporal_distances: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    writer = csv.writer(out_stream)
    writer.writerow(["GeneticDistance", "TemporalDistance", "Probability"])
    for g, t, p in zip(genetic_distances, temporal_distances, probabilities, strict=False):
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
    default_params = NaturalHistoryParameters()
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=10000,
        help="Number of Monte Carlo draws.",
    )
    parser.add_argument(
        "--max-intermediate-hosts",
        type=int,
        default=10,
        help="Maximum number of intermediate hosts (M).",
    )
    parser.add_argument(
        "-m",
        "--included-intermediate-counts",
        type=_parse_included_intermediate_counts,
        default="0",
        help="Comma-separated intermediate-host counts to include; default '0'.",
    )
    parser.add_argument("--seed", type=int, help="Random seed for simulations.")
    parser.add_argument(
        "--grid-min-days",
        type=float,
        default=0.0,
        help="Lower bound of the InfectiousnessToTransmissionTime numerical grid (days).",
    )
    parser.add_argument(
        "--grid-max-days",
        type=float,
        default=60.0,
        help="Upper bound of the InfectiousnessToTransmissionTime numerical grid (days).",
    )
    parser.add_argument(
        "--incubation-shape",
        type=float,
        default=default_params.incubation_shape,
        help="Incubation Gamma shape parameter.",
    )
    parser.add_argument(
        "--incubation-scale",
        type=float,
        default=default_params.incubation_scale,
        help="Incubation Gamma scale parameter.",
    )
    parser.add_argument(
        "--latent-shape",
        type=float,
        default=default_params.latent_shape,
        help="Latent Gamma shape parameter.",
    )
    parser.add_argument(
        "--symptomatic-rate",
        type=float,
        default=default_params.symptomatic_rate,
        help="Symptomatic removal rate.",
    )
    parser.add_argument(
        "--symptomatic-shape",
        type=float,
        default=default_params.symptomatic_shape,
        help="Symptomatic Gamma shape parameter.",
    )
    parser.add_argument(
        "--rel-presymptomatic-infectiousness",
        type=float,
        default=default_params.rel_presymptomatic_infectiousness,
        help="Relative presymptomatic infectiousness.",
    )
    parser.add_argument(
        "--substitution-rate",
        type=float,
        default=1e-3,
        help="Median substitution rate per site per year.",
    )
    parser.add_argument(
        "--relaxed-clock-sigma",
        type=float,
        default=0.33,
        help="Lognormal dispersion for relaxed clock.",
    )
    parser.add_argument("--genome-length", type=int, default=29903, help="Genome length in sites.")
    parser.add_argument(
        "--strict-clock",
        dest="use_relaxed_clock",
        action="store_false",
        help="Use a strict (constant) molecular clock.",
    )
    parser.add_argument(
        "--relaxed-clock",
        dest="use_relaxed_clock",
        action="store_true",
        help="Use a relaxed (lognormal) molecular clock (default).",
    )
    parser.set_defaults(use_relaxed_clock=True)


def _handle_point(args: argparse.Namespace) -> int:
    transmission_profile, molecular_clock = _build_models(args)
    included_intermediate_counts = _normalize_included_intermediate_counts(
        args.included_intermediate_counts
    )

    genetic = np.asarray(args.genetic_distance, dtype=float)
    temporal = np.asarray(args.temporal_distance, dtype=float)
    if genetic.size != temporal.size:
        raise SystemExit("genetic_distance and temporal_distance must have the same length.")

    probs = estimate_linkage_probability(
        transmission_profile=transmission_profile,
        clock=molecular_clock,
        genetic_distance=genetic,
        temporal_distance=temporal,
        included_intermediate_counts=included_intermediate_counts,
        max_intermediate_hosts=args.max_intermediate_hosts,
        num_simulations=args.num_simulations,
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
    if args.genetic_step <= 0 or args.temporal_step <= 0:
        raise SystemExit("genetic-step and temporal-step must be positive.")

    genetic = np.arange(args.genetic_start, args.genetic_stop, args.genetic_step, dtype=float)
    temporal = np.arange(
        args.temporal_start,
        args.temporal_stop,
        args.temporal_step,
        dtype=float,
    )
    if genetic.size == 0 or temporal.size == 0:
        raise SystemExit(
            "genetic-start/genetic-stop or temporal-start/temporal-stop produce an empty grid."
        )

    transmission_profile, molecular_clock = _build_models(args)
    included_intermediate_counts = _normalize_included_intermediate_counts(
        args.included_intermediate_counts
    )

    matrix = estimate_linkage_probability_grid(
        transmission_profile=transmission_profile,
        clock=molecular_clock,
        genetic_distances=genetic,
        temporal_distances=temporal,
        included_intermediate_counts=included_intermediate_counts,
        max_intermediate_hosts=args.max_intermediate_hosts,
        num_simulations=args.num_simulations,
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
    point.add_argument("--output", dest="out", help="Optional output CSV path.")
    _add_common_args(point)
    point.set_defaults(func=_handle_point)

    grid = subparsers.add_parser("grid", help="Estimate linkage probabilities on a grid.")
    grid.add_argument(
        "--genetic-start",
        type=float,
        required=True,
        help="Genetic distance start (inclusive).",
    )
    grid.add_argument(
        "--genetic-stop",
        type=float,
        required=True,
        help="Genetic distance stop (exclusive).",
    )
    grid.add_argument(
        "--genetic-step",
        type=float,
        required=True,
        help="Genetic distance step.",
    )
    grid.add_argument(
        "--temporal-start",
        type=float,
        required=True,
        help="Temporal distance start (inclusive).",
    )
    grid.add_argument(
        "--temporal-stop",
        type=float,
        required=True,
        help="Temporal distance stop (exclusive).",
    )
    grid.add_argument(
        "--temporal-step",
        type=float,
        required=True,
        help="Temporal distance step.",
    )
    grid.add_argument("--output", dest="out", help="Optional output CSV path.")
    _add_common_args(grid)
    grid.set_defaults(func=_handle_grid)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


__all__ = ["build_parser", "main"]
