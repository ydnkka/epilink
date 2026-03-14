from epilink import (
    InfectiousnessToTransmissionTime,
    MolecularClock,
    estimate_linkage_probability,
)


def main() -> None:
    transmission_profile = InfectiousnessToTransmissionTime(rng_seed=123)
    clock = MolecularClock(use_relaxed_clock=False, rng_seed=123)

    probability = estimate_linkage_probability(
        transmission_profile=transmission_profile,
        clock=clock,
        genetic_distance=2,
        temporal_distance=4,
        included_intermediate_counts=(0,),
        num_simulations=10_000,
    )
    print(f"P(link | g=2, t=4): {probability:.4f}")


if __name__ == "__main__":
    main()
