# Performance notes

## What costs the most

- `EpiLink(...)` front-loads work by precomputing Monte Carlo draws for every scenario up to `maximum_depth`.
- Larger `mc_samples` values usually improve score stability but increase model initialization time and memory use.
- `pairwise_model(...)` is the preferred API when you need to score many sample pairs for the same target subset.
- `simulate_genomic_sequences(...)` and `build_pairwise_case_table(...)` can become expensive when the number of sampled cases or genome length grows.

## Cache behavior

- `EpiLink` caches Monte Carlo draws in `draws_by_scenario` during construction.
- `pairwise_model(...)` caches vectorized scorers by canonicalized target-label tuple.
- Repeated calls to `score_pair(...)` and `score_target(...)` on the same model reuse the cached draws.
- Replacing `draws_by_scenario` invalidates cached pairwise scorers so they stay consistent with the new draws.

## Practical guidance

- Reuse a single model instance when scoring many observations with the same transmission profile and target subset.
- Use `score_pair(...)` for one-off inspection and `pairwise_model(...)` or `score_target(...)` for array workloads.
- Start with moderate `mc_samples` values while iterating, then increase them for final analyses if needed.
- Prefer fixed RNG seeds when benchmarking to reduce noise between runs.

## Quick benchmark

Run the bundled helper from the repository root:

```bash
python -m docs.benchmark_api --mc-samples 20000 --grid-size 100 --tree-nodes 63 --genome-length 500
```

It reports timings for:

- model initialization
- single-pair scoring
- vectorized grid scoring
- pairwise distance table construction
