# Usage

```python
from epilink import estimate_linkage_probability, InfectiousnessParams

p = estimate_linkage_probability(
    genetic_distance=2,
    sampling_interval=4,
    intermediate_generations=(0, 1, 2),
    no_intermediates=10,
    num_simulations=10_000,
    relax_rate=False,
)
print(p)
```

Notes:
- Reproducibility: set rng_seed in estimate_linkage_probability if needed.
- Performance: increase num_simulations for smoother estimates.
- TOIT parameters can be customized via InfectiousnessParams and passed to the estimator.
