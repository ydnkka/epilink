# Usage

```python
from epilink import InfectiousnessToTransmissionTime, MolecularClock, estimate_linkage_probability

toit = InfectiousnessToTransmissionTime(rng_seed=123)
clock = MolecularClock(use_relaxed_clock=False, rng_seed=123)

p = estimate_linkage_probability(
    transmission_profile=toit,
    clock=clock,
    genetic_distance=2,
    temporal_distance=4,
    max_intermediate_hosts=10,
    num_simulations=10_000,
)
print(p)
```

Notes:
- Default behavior uses `included_intermediate_counts=(0,)`, matching the appendix definition of recent transmission.
- Reproducibility: set rng_seed in InfectiousnessToTransmissionTime/MolecularClock if needed.
- Performance: increase num_simulations for smoother estimates.
- InfectiousnessToTransmissionTime parameters can be customized via NaturalHistoryParameters.

CLI:

```bash
epilink point -g 2 -t 4 --num-simulations 10000
epilink point -g 2 -t 4 --num-simulations 200 \
  --grid-min-days 0 --grid-max-days 40 \
  --incubation-shape 5.0 --incubation-scale 1.1 \
  --latent-shape 2.0 --symptomatic-rate 0.4 \
  --symptomatic-shape 1.2 --rel-presymptomatic-infectiousness 2.0
epilink grid --genetic-start 0 --genetic-stop 6 --genetic-step 1 \
  --temporal-start 0 --temporal-stop 15 --temporal-step 3
```

Notes:
- Infectiousness profile parameters are configurable via CLI flags (see `epilink point --help`).
- InfectiousnessToTransmissionTime numerical grid bounds are set with `--grid-min-days` and `--grid-max-days`.
