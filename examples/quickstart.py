from epilink import TOIT, MolecularClock, linkage_probability


def main() -> None:
    toit = TOIT(rng_seed=123)
    clock = MolecularClock(relax_rate=False, rng_seed=123)

    p = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=2,
        temporal_distance=4,
        intermediate_generations=(0,),
        num_simulations=10_000,
    )
    print(f"P(link | g=2, t=4): {p:.4f}")


if __name__ == "__main__":
    main()
