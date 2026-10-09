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

## Run benchmarks and generate figures

From the repository root, install the plotting extra and run the quick preset:

```bash
python -m pip install -e '.[benchmark]'
python -m docs.benchmark_api --preset quick
```

The runner prints its output directory and automatically saves three multi-panel
figures as **PNG and vector PDF** in `build/benchmarks/run-*/figures/`:

- `initialization_scaling`: cold/warm initialization and first-time scorer
  construction versus Monte Carlo sample count and scenario depth.
- `scoring_performance`: matched scalar/batch runtime, throughput, paired speedup,
  and detailed single-pair scoring latency.
- `simulation_scaling`: epidemic-date simulation, genomic simulation, and
  pairwise table construction versus case count and genome length.

Each run has its own directory and retains:

- `raw_trials.csv`: individual trial durations, repeated-call counts, normalized
  time per API call, seeded workload order, and workload dimensions.
- `summary.csv`: median, mean, minimum, 25th/75th percentiles, trial count, and
  work items per second. Work units are explicitly named (observations, cases,
  or unique pair-table rows).
- `metadata.json`: complete configuration, fixed target subset, package/Python
  versions, platform/processor information, CPU count, thread environment,
  timer resolution, GC state, Git revision/dirty flag, and measurement method.

Recreate the figures from saved measurements without rerunning the benchmark:

```bash
python -m docs.plot_benchmarks build/benchmarks/run-YOUR-RUN
# Optional alternate figure directory / raster resolution:
python -m docs.plot_benchmarks build/benchmarks/run-YOUR-RUN --output-dir figures --dpi 300
```

Both scripts also support direct execution with `python docs/benchmark_api.py`
and `python docs/plot_benchmarks.py` from the checkout. Plotting uses a headless
backend, so a graphical display is not required.

### Presets and custom sweeps

| Setting | Quick (default) | Full |
|---|---|---|
| MC draws per scenario | 2,000; 10,000; 20,000 | 2,000; 10,000; 50,000; 100,000 |
| Scenario depths | 0; 1; 2; 3 | 0; 1; 2; 4; 6 |
| Paired observations | 1; 10; 100; 1,000 | 1; 10; 100; 1,000; 10,000; 100,000 |
| Cases | 15; 31; 63; 127 | 15; 63; 255; 511 |
| Simulation genome sites | 500; 2,000 | 500; 5,000; 29,903 |
| Trials | 5 | 10 |
| Warm-scoring pilot duration | 0.05 s | 0.2 s |

The baseline for the batch-size and depth sweeps is 20,000 MC draws, depth 2,
and a 29,903-site profile. The fixed target subset is `ad(0)` plus `ca(0,0)`;
model initialization precomputes **all** scenarios at the selected depth.
All scenarios are also evaluated by the separate detailed `score_pair()` timing.

```bash
python -m docs.benchmark_api --preset full --repeats 10

python -m docs.benchmark_api --mc-samples 10000 --maximum-depth 2 \
  --mc-sizes 1000 10000 50000 --depths 0 1 2 4 \
  --batch-sizes 1 100 10000 --tree-sizes 31 127 255 \
  --genome-sizes 1000 5000 29903 --repeats 7 --output-dir build/custom-benchmarks

# Run just the scoring experiments:
python -m docs.benchmark_api --sections scoring
```

`--genome-length` changes the baseline profile length; `--genome-sizes` changes
the genomic/table simulation sweep. Every genomic simulation uses a profile
with the **same** genome length as its sequences. `--warmups`, `--min-time`, and
`--rng-seed` are configurable. The legacy `--tree-nodes N` selects one case count;
`--grid-size N` now selects `N*N` randomly distributed paired observations,
not a Cartesian grid. Use `--help` for all options.

### Timing methodology and interpretation

- **Cold profile + model:** creates a new profile, builds its numerical CDF,
  and constructs `EpiLink`, including Monte Carlo precomputation. Python/package
  imports and process startup are outside the timer; this is a fresh *profile*
  measurement, not a new-process startup benchmark.
- **Warm model:** constructs a new model from a primed profile. The RNG is reset
  outside timing, so cold and warm models use the same seeded draws.
- **Scorer construction:** a fresh model is prepared outside timing; only its
  first `pairwise_model()` call, including sorting, is measured. A new model is
  prepared for every trial, preventing first-use and cached retrieval costs from
  being mixed.
- **Warm scoring:** the scorer is already cached. Scalar and vectorized methods
  receive the same seeded uniform time observations (-30 to 30 days) and integer
  Poisson genetic distances (mean 5),
  use the same draws/targets, and return the same score array. Their
  equality is verified before timing. The scalar workload includes its Python
  loop and output-array construction, reflecting an API caller's batch workload.
- **Detailed scoring:** measured separately because `score_pair()` returns all
  scenario details and performs more work than target-only scoring.
- **Simulation:** tree/input preparation and profile construction are outside
  timing. Epidemic profiles have a primed CDF; stateful simulations reset their
  seed before each trial. Genomic simulation includes both deterministic and
  stochastic outputs with `return_raw=False`; the pairwise table likewise
  computes both distance matrices.

Untimed warm-ups precede measurements. Fast scoring workloads use calibrated
repeated-call blocks to reduce timer overhead; each trial records the mean
duration per call within its block. Constructors and stateful simulations are
timed once per trial. Setup and final-result cleanup occur outside the timer;
earlier output cleanup within a repeated block is part of steady-state call cost.
Workloads are interleaved in a seeded shuffled order on each trial.

Figures show the median and **interquartile range**, not confidence intervals.
Speedup is the median/IQR of the scalar-to-batch ratios paired by trial. The
full preset gives broader scaling data; hardware timing remains machine-dependent.
Compare runs on the same machine under similar load and inspect their metadata.
There are no machine-dependent performance pass/fail thresholds in the tests.
