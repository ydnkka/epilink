import argparse

import numpy as np

from epilink import TOIT, MolecularClock, linkage_probability_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a linkage-probability grid CSV.")
    parser.add_argument("--out", default="grid.csv", help="Output CSV file path.")
    args = parser.parse_args()

    toit = TOIT(rng_seed=123)
    clock = MolecularClock(relax_rate=False, rng_seed=123)

    genetic_distances = np.arange(0, 6, 1)
    temporal_distances = np.arange(0, 15, 3)

    matrix = linkage_probability_matrix(
        toit=toit,
        clock=clock,
        genetic_distances=genetic_distances,
        temporal_distances=temporal_distances,
        num_simulations=10_000,
    )

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        header = ["GeneticDistance"] + [str(t) for t in temporal_distances]
        handle.write(",".join(header) + "\n")
        for i, g in enumerate(genetic_distances):
            row = [str(g)] + [str(matrix[i, j]) for j in range(matrix.shape[1])]
            handle.write(",".join(row) + "\n")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
