# epilink

This package estimates probabilities of transmission linkage between cases based on genetic distances (SNPs) and temporal distances (days), using:

- a mechanistic infectiousness model (E/P/I) from Hart et al. (2021)
- Monte Carlo simulations for epidemiological quantities
- Numba-accelerated kernels for probability calculations

See Usage for examples and API for reference. The package ships with a CLI:

```bash
epilink point -g 2 -t 4 --nsims 10000
```
