# EpiLink clustering perturbation analysis

## Purpose

Test how biological-parameter perturbations affect **full-graph, score-weighted EpiLink Leiden clustering**, and whether updating inference parameters, clustering resolution, or both improves recovery.

The completed [synthetic baseline](../01_synthetic_baseline/README.md) supplies the transmission backbone, known truth, inference parameters, selected resolutions, resolution search bounds/budget and operating criterion. Fresh observations change one generation parameter at a time.

## Four modes

| Mode                                     | EpiLink inference parameters | Leiden resolution                       |
| ---------------------------------------- | ---------------------------- | --------------------------------------- |
| `baseline_inference_baseline_clustering` | Baseline values              | Baseline-selected                       |
| `baseline_inference_updated_clustering`  | Baseline values              | Scenario-specific development selection |
| `matched_inference_baseline_clustering`  | Scenario's generation values | Baseline-selected                       |
| `matched_inference_updated_clustering`   | Scenario's generation values | Scenario-specific development selection |

Every observed pair is a graph edge, with its original EpiLink score as weight, **including zero-weight edges**. There is no graph cutoff or threshold search. Updated clustering selects **only resolution**, independently for each scenario, inference mode and EpiLink scorer. The baseline's objective, restarts and algorithm seed stay fixed.

The supplied config uses **EDD, EDS, ESD and ESS**, with the baseline's `balanced_M0` criterion. Each has one `leiden/<scorer>` pipeline. The second letter denotes deterministic/stochastic genetic **inference**, and the third denotes deterministic/stochastic **observations**. Baseline/matched in the four modes refers to biological parameter inputs, independently of these genetic assumptions.

| Scorer | Inference genetics | Observed genetics |
| ------ | ------------------ | ----------------- |
| EDD    | Deterministic      | Deterministic     |
| EDS    | Deterministic      | Stochastic        |
| ESD    | Stochastic         | Deterministic     |
| ESS    | Stochastic         | Stochastic        |

The study produces EpiLink clustering metrics and partitions. General feature ambiguity is available separately in [synthetic diagnostics](../00_synthetic_diagnostics/README.md); baseline and Boston retain their method comparisons.

## Development and evaluation

Updated resolutions are chosen on `development_seeds`, using the **same resolution bounds, search budget and selection rule as the baseline**. Coarse-to-fine trials are adapted to each arm's fresh development evidence and shared across its scorers. Selection requires complete development evidence and feasible settings for every requested pipeline. No evaluation observations are accessed before the arm's settings are saved.

All four modes then evaluate the same separate `seeds`. Both seed sets must be distinct and fresh relative to **every baseline split**. Within a scenario, inference modes share observations; baseline/updated clustering additionally share score artifacts. EpiLink is training-free.

Every arm includes a fresh **unperturbed control**. Compute:

```text
delta = perturbed metric - same-seed, same-mode unperturbed-control metric
```

Pair controls by mode, seed, criterion and pipeline. **Do not pair by setting ID:** updated resolutions may differ between a scenario and its control. `results_deltas.csv` retains both `setting_id` and `baseline_setting_id`. The updated arm measures scenario-specific adaptation relative to a development-selected unperturbed control.

Negative F1 changes indicate reduced recovery. Positive M≥3 contamination changes indicate more distant within-cluster pairs; M≥3 contamination is not every M=0 false positive. Means, sample SDs and ranges describe observation variation conditional on one backbone. Undefined metrics remain missing.

## Run

From the repository root, after completing the baseline's evaluation:

```bash
python evaluation/02_synthetic_perturbation/run.py --smoke
python evaluation/02_synthetic_perturbation/run.py
python evaluation/02_synthetic_perturbation/run.py --stage report
```

Equivalent CLI: `epilink-evaluate perturbation --smoke`. The default reference is `../01_synthetic_baseline/outputs/baseline/current.json`, resolved once at startup. Pin a run with `--baseline-run "evaluation/01_synthetic_baseline/outputs/baseline/runs/<id>"`. Use `--config` for another schema-1 YAML and `--output` to override the root. CLI path overrides are relative to the working directory; YAML paths are relative to the YAML.

`all` (default) runs development selection where required, held-out clustering, collection and reporting. `report` renders saved tables. Repeat the same command/options to resume validated observations, scores and partitions. Smoke appends `_smoke` to the output root, uses 64 cases, development seed 90001, evaluation seed 91001, and the two incubation-mean perturbations plus the control. It is pipeline validation.

The reference must contain completed full-graph Leiden development, selection and evaluation. A smoke baseline is valid only for a smoke study.

## Configuration

| Field               | Meaning                                                                   |
| ------------------- | ------------------------------------------------------------------------- |
| `schema_version`    | `1` for clustering-only, four-arm analysis.                               |
| `baseline_run`      | Completed baseline run or current-run pointer.                            |
| `output_directory`  | Separate study root, default `outputs/perturbation`.                      |
| `development_seeds` | Resolution-selection seeds, default 80001–80003.                          |
| `seeds`             | Paired evaluation seeds, default 81001–81003.                             |
| `scorers`           | EpiLink scorers only, supplied `[EDD, EDS, ESD, ESS]`.                    |
| `criterion`         | Baseline selection rule, default `balanced_M0`.                           |
| `modes`             | All four unique mode names from the table above.                          |
| `perturbations`     | One-at-a-time `parameter` with either `multipliers` or absolute `values`. |
| `smoke`             | `cases`, `development_seed`, `seed`, and subset of `parameters`.          |

The supplied full study perturbs incubation mean/CV, testing-delay mean/CV and substitution rate to 0.75×/1.25× baseline; relaxation uses absolute 0/0.66. This gives 12 scenarios plus the control, four modes, four EpiLink pipelines, and three evaluation seeds: **624 evaluation rows**. Smoke has three scenarios × four modes × four scorers × one evaluation seed: **48 rows**. Only the two updated-clustering modes sweep the development resolutions. Levels must satisfy EpiLink's natural-history requirements; duplicate or baseline-equal levels are rejected.

## Outputs

Locate the run through `outputs/perturbation[_smoke]/current.json`, then inspect:

1. `coverage.csv`: each scenario/mode's completion, expected rows and errors.
2. `reference.json`, `selection.json`, `settings.json`, `scenarios.json`: reference identity, baseline EpiLink decisions and resolved perturbations.
3. `report.html` / `report.md`: coverage, actual resolutions, absolute performance and paired changes.
4. `results.csv`, `results_summary.csv`: seed-specific and summarized clustering metrics, including fresh controls.
5. `results_deltas.csv`, `results_delta_summary.csv`: seed-paired changes and their mean/SD/min/max/count, with control availability.
6. `scenarios/<scenario>/<mode>/selection.json`: exact arm-specific decisions, development seeds and evidence checksum; `settings.json` stores the replayed definitions.
7. The arm's `development/` sweep for updated modes and `evaluation/seed_<seed>/clusters/<setting-id>/` partitions, metrics and algorithm metadata.

Missing controls remain visible with `control_available: false` and missing deltas. Failed/incomplete arms give a partial study and a nonzero exit code. The reference is read-only and validates scientific implementation, source evidence, truth and the requested completed held-out EpiLink partitions. See [OUTPUTS.md](../../OUTPUTS.md#12-perturbation-study-outputs).

## Manuscript displays from saved results

After a complete full study:

```bash
python -m evaluation.results.tab03  # four-arm fresh-control clustering
python -m evaluation.results.fig13  # four-arm F1/contamination heatmaps
python -m evaluation.results.fig14  # paired means and seed-level ranges
python -m evaluation.results.fig15  # matched-minus-baseline inference contrasts
python -m evaluation.results.fig16  # main-results four-arm overview
```

Each accepts `--run-dir` and `--output-dir`; figures accept `--format pdf|png|both`. Outputs default to `evaluation/results/outputs/02_synthetic_perturbation/<run-id>/`. Each scorer has its own labelled panel, grouped by observed genetics (EDD/ESD and EDS/ESS). All four modes remain visible. `fig15` contrasts control-paired deltas within seed, separately for baseline and updated resolution. `tab03` has one row per scorer/mode on the fresh control. `fig13`/`fig16` CSVs include scorer, pipeline and mode identifiers. Perturbation occupies figures 13–16 in the consecutive results sequence.
