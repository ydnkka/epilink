import argparse

import numpy as np

from epilink import (
    InfectiousnessToTransmissionTime,
    MolecularClock,
    estimate_linkage_probability_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a linkage-probability grid CSV.")
    parser.add_argument("--output", default="grid.csv", help="Output CSV file path.")
    args = parser.parse_args()

    transmission_profile = InfectiousnessToTransmissionTime(rng_seed=123)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=123)

    genetic_distances = np.arange(0, 6, 1)
    temporal_distances = np.arange(0, 15, 3)

    matrix = estimate_linkage_probability_grid(
        transmission_profile=transmission_profile,
        clock=clock,
        genetic_distances=genetic_distances,
        temporal_distances=temporal_distances,
        num_simulations=10_000,
    )

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        header = ["GeneticDistance"] + [str(t) for t in temporal_distances]
        handle.write(",".join(header) + "\n")
        for i, g in enumerate(genetic_distances):
            row = [str(g)] + [str(matrix[i, j]) for j in range(matrix.shape[1])]
            handle.write(",".join(row) + "\n")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
