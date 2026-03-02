# Usage

```python
from epilink import TOIT, MolecularClock, linkage_probability

toit = TOIT(rng_seed=123)
clock = MolecularClock(relax_rate=False, rng_seed=123)

p = linkage_probability(
    toit=toit,
    clock=clock,
    genetic_distance=2,
    temporal_distance=4,
    intermediate_generations=(0, 1, 2),
    intermediate_hosts=10,
    num_simulations=10_000,
)
print(p)
```

Notes:
- Reproducibility: set rng_seed in TOIT/MolecularClock if needed.
- Performance: increase num_simulations for smoother estimates.
- TOIT parameters can be customized via InfectiousnessParams.

CLI:

```bash
epilink point -g 2 -t 4 --nsims 10000
epilink point -g 2 -t 4 --nsims 200 \
  --a 0 --b 40 \
  --incubation-shape 5.0 --incubation-scale 1.1 \
  --latent-shape 2.0 --symptomatic-rate 0.4 \
  --symptomatic-shape 1.2 --rel-presymptomatic-infectiousness 2.0
epilink grid --g-start 0 --g-stop 6 --g-step 1 --t-start 0 --t-stop 15 --t-step 3
```

Notes:
- Infectiousness profile parameters are configurable via CLI flags (see `epilink point --help`).
- TOIT support bounds are set with `--a` and `--b`.
