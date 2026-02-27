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
    no_intermediates=10,
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
epilink grid --g-start 0 --g-stop 6 --g-step 1 --t-start 0 --t-stop 15 --t-step 3
```
