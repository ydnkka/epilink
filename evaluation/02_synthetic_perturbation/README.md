# EpiLink clustering perturbation analysis

## Purpose

Test how biological-parameter perturbations affect **full-graph, score-weighted EpiLink Leiden clustering**, and whether updating inference parameters, resolution, or both improves recovery.

[Baseline](../01_synthetic_baseline/README.md) supplies: backbone, truth, inference parameters, selected resolutions, search bounds/budget, operating criterion. Fresh observations: one generation parameter at a time.

## Four modes

| Mode | Inference | Resolution |
|------|-----------|------------|
| `baseline_inference_baseline_clustering` | Baseline | Baseline-selected |
| `baseline_inference_updated_clustering` | Baseline | Scenario-specific development |
| `matched_inference_baseline_clustering` | Scenario values | Baseline-selected |
| `matched_inference_updated_clustering` | Scenario values | Scenario-specific development |

Every observed pair = graph edge (original EpiLink score as weight, **including zero**). No graph cutoff/threshold search. Updated clustering: **resolution only** (per scenario/inference/scorer). Baseline objective/restarts/seed fixed.

Config: **EDD, EDS, ESD, ESS**, `balanced_M0` criterion. One `leiden/<scorer>` pipeline each. 2nd letter: inference genetics; 3rd: observed genetics. Baseline/matched = biological parameters (independent of genetic assumptions).

| Scorer | Inference | Observed |
|--------|-----------|----------|
| EDD | Deterministic | Deterministic |
| EDS | Deterministic | Stochastic |
| ESD | Stochastic | Deterministic |
| ESS | Stochastic | Stochastic |

Output: EpiLink clustering metrics/partitions. Feature ambiguity: [diagnostics](../00_synthetic_diagnostics/README.md).

## Development and evaluation

Updated resolutions: `development_seeds`, **same bounds/budget/rule as baseline**. Coarse-to-fine trials: per-arm fresh development evidence, shared across scorers. Selection: complete development evidence, feasible settings per pipeline. No evaluation observations before settings saved.

All four modes evaluate same separate `seeds`. Both seed sets distinct/fresh relative to **every baseline split**. Within scenario: inference modes share observations; baseline/updated clustering share scores. EpiLink: training-free.

Every arm: fresh **unperturbed control**.

```text
delta = perturbed metric - same-seed, same-mode control metric
```

Pair by mode/seed/criterion/pipeline. **Not by setting ID** (updated resolutions may differ). `results_deltas.csv`: `setting_id` + `baseline_setting_id`. Updated arm: scenario-specific adaptation vs development-selected control.

Negative ΔF1: reduced recovery. Positive ΔM≥3 contamination: more distant within-cluster pairs (≠ all M=0 false positives). Means/SDs/ranges: observation variation (conditional on backbone). Undefined metrics: missing.

## Run

From repository root, after baseline evaluation:

```bash
python evaluation/02_synthetic_perturbation/run.py --smoke
python evaluation/02_synthetic_perturbation/run.py
python evaluation/02_synthetic_perturbation/run.py --stage report
```

CLI: `epilink-evaluate perturbation --smoke`. Default reference: `../01_synthetic_baseline/outputs/baseline/current.json`. Pin: `--baseline-run "<path>"`. `--config`: schema-1 YAML. `--output`: override root. CLI paths: working directory-relative; YAML paths: YAML-relative.

`all` (default): development selection (where required), held-out clustering, collection, reporting. `report`: render saved tables. Repeat command/options to resume. Smoke: `_smoke` root, 64 cases, dev seed 90001, eval seed 91001, two incubation-mean perturbations + control. Pipeline validation.

Reference: completed full-graph Leiden development/selection/evaluation. Smoke baseline = smoke study only.

## Configuration

| Field | Meaning |
|-------|---------|
| `schema_version` | `1` (clustering-only, four-arm) |
| `baseline_run` | Completed baseline run or pointer |
| `output_directory` | Study root (default `outputs/perturbation`) |
| `development_seeds` | Resolution-selection seeds (default 80001–80003) |
| `seeds` | Paired evaluation seeds (default 81001–81003) |
| `scorers` | EpiLink only (default `[EDD, EDS, ESD, ESS]`) |
| `criterion` | Baseline rule (default `balanced_M0`) |
| `modes` | Four mode names |
| `perturbations` | One-at-a-time `parameter` (`multipliers` or `values`) |
| `smoke` | `cases`, `development_seed`, `seed`, `parameters` subset |

Full study: incubation mean/CV, testing-delay mean/CV, substitution rate (0.75×/1.25×); relaxation: 0/0.66 (absolute). 12 scenarios + control, four modes, four pipelines, three seeds: **624 rows**. Smoke: three scenarios × four modes × four scorers × one seed: **48 rows**. Updated-clustering modes sweep resolutions. Levels: must satisfy EpiLink requirements; duplicate/baseline-equal rejected.

## Outputs

Locate run: `outputs/perturbation[_smoke]/current.json`. Inspect:

1. `coverage.csv`: scenario/mode completion, expected rows, errors.
2. `reference.json`, `selection.json`, `settings.json`, `scenarios.json`: reference identity, baseline decisions, perturbations.
3. `report.html`/`report.md`: coverage, resolutions, performance, paired changes.
4. `results.csv`, `results_summary.csv`: seed-specific/summarized metrics (with controls).
5. `results_deltas.csv`, `results_delta_summary.csv`: seed-paired changes (mean/SD/min/max/count), control availability.
6. `scenarios/<scenario>/<mode>/selection.json`: arm decisions, development seeds, evidence checksum; `settings.json`: replayed definitions.
7. Arm's `development/` (updated modes) and `evaluation/seed_<seed>/clusters/<setting-id>/` (partitions, metrics, algorithm).

Missing controls: `control_available: false`, missing deltas. Failed/incomplete arms: partial study, nonzero exit. Reference: read-only (validates implementation, evidence, truth, partitions). See [OUTPUTS.md](../../OUTPUTS.md#12-perturbation-study-outputs).

## Manuscript displays from saved results

After complete full study:

```bash
python -m evaluation.results.tab03  # four-arm fresh-control clustering
python -m evaluation.results.fig13  # F1/contamination heatmaps
python -m evaluation.results.fig14  # paired means, seed-level ranges
python -m evaluation.results.fig15  # matched-minus-baseline contrasts
python -m evaluation.results.fig16  # four-arm overview
```

Options: `--run-dir`, `--output-dir`; figures: `--format pdf|png|both`. Output: `evaluation/results/outputs/02_synthetic_perturbation/<run-id>/`. Per scorer: labelled panel, grouped by observed genetics (EDD/ESD, EDS/ESS). All four modes visible. `fig15`: control-paired deltas (baseline/updated resolution). `tab03`: one row per scorer/mode (fresh control). `fig13`/`fig16` CSVs: scorer, pipeline, mode identifiers. Figures 13–16.
