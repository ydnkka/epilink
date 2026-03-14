# Package Structure Notes

## Goal

Keep `epilink` as a standalone installable research tool with a small,
coherent public API, while leaving manuscript workflows and evaluation logic
to the separate `evaluating-epilink` repository.

## Final Layout

```text
src/epilink/
  __init__.py
  __main__.py
  cli/
    __init__.py
    main.py
  model/
    __init__.py
    parameters.py
    profiles.py
    clock.py
  inference/
    __init__.py
    draws.py
    kernels.py
    estimator.py
  simulation/
    __init__.py
    epidemic.py
    genomics.py
    pairwise.py
  sequence/
    __init__.py
    packing.py
```

## Module Responsibilities

### Model

- `model/parameters.py`
  - `NaturalHistoryParameters`
  - `estimate_presymptomatic_transmission_fraction`
- `model/profiles.py`
  - `BaseTransmissionProfile`
  - `InfectiousnessToTransmissionTime`
  - `SymptomOnsetToTransmissionTime`
- `model/clock.py`
  - `MolecularClock`

### Inference

- `inference/kernels.py`
  - `temporal_kernel`
  - `genetic_kernel`
- `inference/draws.py`
  - `LinkageMonteCarloSamples`
- `inference/estimator.py`
  - `estimate_linkage_probability`
  - `estimate_linkage_probability_grid`
  - `estimate_temporal_linkage_probability`
  - `estimate_genetic_linkage_probability`

### Simulation and Sequence Utilities

- `simulation/epidemic.py`
  - `simulate_epidemic_dates`
- `simulation/genomics.py`
  - `simulate_genomic_sequences`
- `simulation/pairwise.py`
  - `build_pairwise_case_table`
- `sequence/packing.py`
  - `SequencePacker64`
  - `PackedGenomicData`

### CLI

- `cli/main.py`
  - parser construction
  - argument normalization
  - point and grid execution

## Design Notes

- The package-level API in `epilink.__init__` is the primary user-facing entry
  point.
- Internal modules are organized by scientific responsibility rather than by
  paper sections or legacy script boundaries.
- Old one-file compatibility shims were removed once the standalone API
  stabilized.
