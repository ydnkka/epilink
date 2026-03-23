from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from .model import (
    EpiLink,
    InfectiousnessToTransmission,
    NaturalHistoryParameters,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EpiLink: Epidemiological linkage inference from temporal and genetic data."
    )
    parser.add_argument(
        "input",
        help="Input CSV file with 'sample_time_difference' and 'genetic_distance' columns.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV file path. If not provided, results are printed to stdout.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=10000,
        help="Number of Monte Carlo samples (default: 10000).",
    )
    parser.add_argument(
        "--maximum-depth",
        type=int,
        default=2,
        help="Maximum hidden depth for scenarios (default: 2).",
    )
    parser.add_argument(
        "--target",
        nargs="+",
        default=["ad(0)"],
        help="Target scenario(s) to score (default: ad(0)).",
    )
    parser.add_argument(
        "--mutation-process",
        choices=["deterministic", "stochastic"],
        default="stochastic",
        help="Mutation process model (default: stochastic).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    nh_group = parser.add_argument_group("Natural History Parameters")
    nh_group.add_argument(
        "--incubation-shape",
        type=float,
        default=5.807,
        help="Shape parameter for incubation period (default: 5.807).",
    )
    nh_group.add_argument(
        "--incubation-scale",
        type=float,
        default=0.948,
        help="Scale parameter for incubation period in days (default: 0.948).",
    )
    nh_group.add_argument(
        "--latent-shape",
        type=float,
        default=3.38,
        help="Shape parameter for latent (E) stage (default: 3.38).",
    )
    nh_group.add_argument(
        "--symptomatic-rate",
        type=float,
        default=0.37,
        help="Symptomatic removal rate in 1/day (default: 0.37).",
    )
    nh_group.add_argument(
        "--symptomatic-shape",
        type=float,
        default=1.0,
        help="Shape parameter for symptomatic infectious (I) stage (default: 1.0).",
    )
    nh_group.add_argument(
        "--transmission-rate-ratio",
        type=float,
        default=2.29,
        help="Ratio of transmission rates in P and I stages (default: 2.29).",
    )
    nh_group.add_argument(
        "--testing-delay-shape",
        type=float,
        default=1.0,
        help="Shape parameter for testing delay (default: 2.0).",
    )
    nh_group.add_argument(
        "--testing-delay-scale",
        type=float,
        default=1.0,
        help="Scale parameter for testing delay in days (default: 1.0).",
    )
    nh_group.add_argument(
        "--substitution-rate",
        type=float,
        default=1e-3,
        help="Median substitution rate per site per year (default: 1e-3).",
    )
    nh_group.add_argument(
        "--relaxation",
        type=float,
        default=0.33,
        help="Lognormal std dev for relaxed clock (default: 0.33, 0 for strict).",
    )
    nh_group.add_argument(
        "--genome-length",
        type=int,
        default=29903,
        help="Genome length in sites (default: 29903).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        df = pd.read_csv(args.input)
        required_cols = {"sample_time_difference", "genetic_distance"}
        if not required_cols.issubset(df.columns):
            logger.error("Input CSV must contain columns: %s", required_cols)
            sys.exit(1)

        params = NaturalHistoryParameters(
            incubation_shape=args.incubation_shape,
            incubation_scale=args.incubation_scale,
            latent_shape=args.latent_shape,
            symptomatic_rate=args.symptomatic_rate,
            symptomatic_shape=args.symptomatic_shape,
            transmission_rate_ratio=args.transmission_rate_ratio,
            testing_delay_shape=args.testing_delay_shape,
            testing_delay_scale=args.testing_delay_scale,
            substitution_rate=args.substitution_rate,
            relaxation=args.relaxation,
            genome_length=args.genome_length,
        )

        profile = InfectiousnessToTransmission(parameters=params)
        model = EpiLink(
            transmission_profile=profile,
            maximum_depth=args.maximum_depth,
            mc_samples=args.mc_samples,
            target=args.target,
            mutation_process=args.mutation_process,
        )

        logger.info("Scoring %d pairs...", len(df))
        scores = model.score_target(
            sample_time_difference=df["sample_time_difference"].values,
            genetic_distance=df["genetic_distance"].values,
        )

        df["epilink_score"] = scores

        if args.output:
            df.to_csv(args.output, index=False)
            logger.info("Results saved to %s", args.output)
        else:
            print(df.to_csv(index=False))

    except Exception as e:
        logger.exception("An error occurred during execution: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
