# EpiLink evaluation: operational guide

Configure and run the project, locate results, and resume interrupted work. Commands assume the **repository root** is the working directory.

- [Overview](README.md)
- [Shared experiment](evaluation/shared_synthetic/README.md)
- [Diagnostics](evaluation/00_synthetic_diagnostics/README.md)
- [Baseline](evaluation/01_synthetic_baseline/README.md)
- [Perturbation](evaluation/02_synthetic_perturbation/README.md)
- [Boston](evaluation/03_boston_application/README.md)
- [Output reference](OUTPUTS.md)

## Contents

- [EpiLink evaluation: operational guide](#epilink-evaluation-operational-guide)
  - [Contents](#contents)
  - [1. How the pipeline works](#1-how-the-pipeline-works)
  - [2. Environment and source inputs](#2-environment-and-source-inputs)
    - [Existing checkout](#existing-checkout)
    - [Fresh environment](#fresh-environment)
    - [Obtain inputs and check the installation](#obtain-inputs-and-check-the-installation)
  - [3. Configure an experiment](#3-configure-an-experiment)
    - [Inputs, simulation, and seeds](#inputs-simulation-and-seeds)
    - [Natural-history parameters](#natural-history-parameters)
    - [Scorers, searches, and comparison settings](#scorers-searches-and-comparison-settings)
  - [4. Prepare or regenerate the SCoVMod tree](#4-prepare-or-regenerate-the-scovmod-tree)
  - [5. Run smoke validation and the baseline](#5-run-smoke-validation-and-the-baseline)
  - [6. Stage reference](#6-stage-reference)
  - [7. Find and interpret results](#7-find-and-interpret-results)
    - [Read development evidence in this order](#read-development-evidence-in-this-order)
  - [8. Choose and freeze operating criteria](#8-choose-and-freeze-operating-criteria)
  - [9. Resume work and understand caching](#9-resume-work-and-understand-caching)
    - [Project-wide progress logs](#project-wide-progress-logs)
    - [Repeatable pipeline smoke validation](#repeatable-pipeline-smoke-validation)
    - [Checkpoint reuse](#checkpoint-reuse)
    - [Clean restart after updating phylogeny code](#clean-restart-after-updating-phylogeny-code)
  - [10. Troubleshooting](#10-troubleshooting)
  - [11. Clear outputs with reset-outputs](#11-clear-outputs-with-reset-outputs)
  - [12. Perturbation and Boston application](#12-perturbation-and-boston-application)

## 1. How the pipeline works

Studies are under `evaluation/`. Diagnostics characterizes observations and graph controls. Baseline compares methods and freezes settings. Perturbation tests EDD/EDS/ESD/ESS under different inference/resolution. Boston applies frozen settings to empirical data. Downstream studies use completed baseline. Sections 3–9 cover preparation, diagnostics, and baseline; section 12 covers downstream commands.

```text
SCoVMod CSVs -> transmission backbone -> relationship truth (AD, CA, M)
  -> simulate dates and genomes -> GD/TD -> diagnostics
  -> EpiLink, genetic, logistic scores -> pairwise/clustering metrics
  -> IQ-TREE trees -> TreeCluster partitions
  -> compare with truth -> select settings -> evaluate on held-out data
```

The backbone supplies truth. IQ-TREE/LSD2 reconstruct trees from observations. Diagnostics generates development observations; baseline reuses them and generates training realizations for logistic fitting. Development determines operating settings. Evaluation observations are generated after frozen settings are validated. Unsampled intermediates remain in truth.

**M=0** (primary target) includes direct transmission and shared-infector pairs. See [protocol](evaluation/01_synthetic_baseline/README.md) for targets, scorers, and metrics.

Entry points: [`cli.py`](src/epilink_evaluation/cli.py), [`workflows/diagnostics.py`](src/epilink_evaluation/workflows/diagnostics.py), [`workflows/baseline.py`](src/epilink_evaluation/workflows/baseline.py). Modules: `inputs`, `diagnostics`, `truth`, `scorers`, `graphs`, `phylogeny`, `clusterers`, `metrics`, `selection`, `reporting`.

## 2. Environment and source inputs

### Existing checkout

```bash
conda activate epilik_evaluation
python -m pip install -e '.[test]'
```

### Fresh environment

Python ≥3.10:

```bash
conda create -n epilik_evaluation -c conda-forge python=3.14 pip
conda activate epilik_evaluation
python -m pip install -e '.[test]'
```

The install supplies `epilink-evaluate`, EpiLink 0.1.5, TreeCluster, and pytest. **IQ-TREE ≥2.0.6** is external:

```bash
conda install -c conda-forge -c bioconda iqtree
```

`phylogeny.executable: iqtree` discovers `iqtree3`, `iqtree2`, or `iqtree` on PATH. An explicit path pins the executable. TreeCluster is a Python dependency. `check` verifies tools.

**Boston:** IQ-TREE builds trees from FASTA and reference. TN93 pair table supplies genetic distances.

### Obtain inputs and check the installation

Tracked formats use Git LFS:

```bash
git lfs install
git lfs pull
```

SCoVMod needs:

```text
data/raw/scovmod/InfectedIndividuals.1.csv
data/raw/scovmod/TransmissionEvents.1.csv
```

Prepared inputs and outputs are Git-ignored. Generate the tree in `evaluation/shared_synthetic/outputs/inputs/` or configure `inputs.tree_path`.

```bash
epilink-evaluate check --config evaluation/01_synthetic_baseline/config.yaml
```

`check` prints versions, paths, splits, output location, and tree existence. It does not validate CSV contents or run external tools; smoke workflow exercises integration.

## 3. Configure an experiment

[`evaluation/shared_synthetic/config.yaml`](evaluation/shared_synthetic/config.yaml) owns `inputs`, `generation`, `simulation`, and `splits`. Diagnostics and baseline configs load it via `experiment_config`. Keep diagnostic controls in the former; scorers, method grids, and selection rules in the latter. For another design, copy the shared config and update both study configs.

**Path rules:** Paths resolve relative to their YAML file. Shared inputs and `outputs/synthetic` resolve under `evaluation/shared_synthetic/`; study configs resolve relative to their YAML. `outputs/baseline` means `evaluation/01_synthetic_baseline/outputs/baseline`. Moving a config changes its relative paths. CLI `--config` and `--output` are shell-relative; use absolute paths for executables. `inputs.tree_source_path` is config-relative (default: tree's `.source.json`). Input paths are shared; changing `--output` doesn't relocate them.

### Inputs, simulation, and seeds

| Field | Meaning |
|-------|---------|
| `experiment_config` | Reference to shared data-design YAML |
| `name` | Experiment label |
| `output_directory` | Study: `artifacts/`, `runs/`, `current.json`; Shared: `artifacts/`, `experiments/`, `heldout_access/` |
| `inputs.tree_path` | Transmission backbone path |
| `inputs.infection_path`, `transmission_path` | Raw SCoVMod inputs |
| `inputs.target_component_size` | Requested component size |
| `inputs.tree_seed` | Random infector assignment seed |
| `inputs.smoke_cases` | `null` = full backbone; count = ancestor-preserving prefix (`--smoke` sets 64) |
| `simulation.fraction_sampled` | Fraction observed, `(0, 1]` |
| `simulation.sequence_length` | Simulated sequence sites |
| `simulation.alignment_length` | TreeCluster threshold scaling (default: `sequence_length`) |
| `splits.train` | Logistic model training seeds |
| `splits.development` | Parameter-grid comparison seeds |
| `splits.evaluation` | Held-out evaluation seeds |
| `scorer.seed`, `scorer.mc_samples` | EpiLink Monte Carlo seed and draws |
| `clustering.leiden.seed`, `restarts` | Leiden seed and restarts |
| `phylogeny.seed` | IQ-TREE seed |
| `phylogeny.model` | IQ-TREE model (e.g., JC, MFP) |
| `phylogeny.threads` | IQ-TREE threads |
| `phylogeny.clock_rate` | LSD2 clock rate (subs/site/day); null for estimation |

Shared config: `inputs`, `simulation`, `splits`. Baseline: scorer and clustering. Observation seeds must be nonnegative, unique across splits. Held-out seeds cannot become training/development. Algorithm seeds control inference independently.

### Natural-history parameters

Shared `generation` configures observation simulation. Baseline derives `inference` from it and requires matched values; edit in shared config and rerun diagnostics.

| Field | Meaning |
|-------|---------|
| `incubation.mean`, `testing_delay.mean` | Mean durations (days) |
| `incubation.cv`, `testing_delay.cv` | Coefficient of variation; Gamma shape = `1 / cv²` |
| `latent_shape` | Latent-stage Gamma shape (must be < incubation shape) |
| `symptomatic_rate`, `symptomatic_shape` | Removal rate (1/day) and Gamma shape |
| `transmission_rate_ratio` | Presymptomatic-to-symptomatic transmission ratio |
| `substitution_rate` | Substitutions/site/year |
| `relaxation` | Lognormal SD of branch rates; zero = strict clock |
| `genome_length` | Sites for EpiLink mutation expectations |

Configuration uses `genome_length: 29903` and `simulation.sequence_length: 5000`. Changing either changes the experiment. IQ-TREE infers genetic lengths; LSD2 dates using simulation days (synthetic) or calendar dates (Boston).

### Scorers, searches, and comparison settings

| Field | Meaning |
|-------|---------|
| `scorers` | Subset of EDD, EDS, ESD, ESS, GD_D, GD_S, LOGIT_D, LOGIT_S |
| `scorer.logistic_C` | Inverse regularization for logistic fitting |
| `clustering.algorithms` | `components`, `leiden`, or both |
| `clustering.leiden.objective` | `CPM` or `modularity` |
| `clustering.leiden.resolutions` | Adaptive search: `min`, `max`, `initial_points`, `budget`, `scale` |
| `treecluster.enabled` | Include raw/dated phylogenetic comparisons |
| `treecluster.methods` | `max_clade`, `avg_clade`, `single_linkage` |
| `treecluster.genetic_threshold_snps` | Integer-SNP search (converted to subs/site) |
| `treecluster.temporal_threshold_days` | Day search (fractional allowed) |
| `treecluster.executable` | Default: `TreeCluster.py` |
| `treecluster.timeout` | TreeCluster timeout (seconds) |
| `phylogeny.model` | IQ-TREE model (JC for synthetic, MFP for Boston) |
| `phylogeny.threads` | IQ-TREE threads |
| `phylogeny.seed` | IQ-TREE seed |
| `phylogeny.clock_rate` | LSD2 clock rate (subs/site/day); null for estimation |
| `phylogeny.timeout` | IQ-TREE/LSD2 timeout (seconds) |
| `phylogeny.executable` | IQ-TREE name/path |
| `selection.criteria` | Objectives/constraints for freezing settings (section 8) |

Pairwise/component searches share distinct development-score cutoffs plus empty-selection. Cutoffs retain whole ties; genetic distances use `<=`, EpiLink/logistic use `>=`. Each pipeline chooses its optimum. Components add edges incrementally. Leiden graphs retain all edges; `threshold` is null.

Leiden: `initial_points` trials spanning `min`/`max`, refines intervals, explores widest per batch, up to `budget` resolutions. Trials cover all scorers/seeds. Completed trials reuse on restart. Baseline bounds: 0.01–1, 8 initial points, 28 budget, log scale.

TreeCluster: separate searches for SNPs and days. Cutoff trials cover all processes/methods/seeds. SNP trials are integers; days can be fractional. Supplied: 0–10 SNPs (4 initial, budget 11) and 0–105 days (8 initial, budget 28), linear scale. Completed partitions/trees reuse. Explicit lists supported.

## 4. Prepare or regenerate the SCoVMod tree

```bash
epilink-evaluate scovmod --stage prepare --config evaluation/01_synthetic_baseline/config.yaml
```

- Default tree/provenance: `evaluation/shared_synthetic/outputs/inputs/`
- `scovmod --stage prepare` loads shared input settings; diagnostics uses same preparation
- `scovmod` supports only `prepare`. Diagnostics prepares experiment/truth/observations. Baseline requires completed diagnostics.
- **Matching artifact:** reuse after validating manifest signature/checksums
- **Missing/stale artifact:** reconstruct from CSVs, write tree, `.source.json`, `manifest.json`
- **Prebuilt tree without manifest:** validate and retain
- Full/smoke runs share inputs. `--output` changes run root, not input paths.

To generate a new target:

```yaml
inputs:
  tree_path: outputs/inputs_target1000_seed12345/transmission_tree.gml
  target_component_size: 1000
  tree_seed: 12345
```

Use a separate directory to retain the existing artifact. No CLI `--force`; settings are in YAML.

Reconstruction assigns infectors by seed, keeps one incoming edge per case (earliest time, then infector ID), chooses the weakly connected component closest to target size. Equally close sizes prefer larger component, then lowest member ID. Actual count can differ from target.

Inspect provenance:

```bash
python -m json.tool evaluation/shared_synthetic/outputs/inputs_target1000_seed12345/transmission_tree.source.json
```

`n_cases` is the actual count. Use recorded size, not target. Complete diagnostics for new designs before baseline.

## 5. Run smoke validation and the baseline

Smoke validation:

```bash
python evaluation/00_synthetic_diagnostics/run.py --config evaluation/00_synthetic_diagnostics/config.yaml --smoke --stage all
python evaluation/01_synthetic_baseline/run.py --config evaluation/01_synthetic_baseline/config.yaml --smoke --stage all
```

`--smoke`: up to 64 cases, seeds 71001/72001/73001 (train/development/evaluation), separate `_smoke` roots. Diagnostics: 0.1–0.5 resolution, 3 initial points, 5 budget, 2 restarts. Baseline: 1,024 MC draws, exact cutoffs, reduced TreeCluster, Leiden 0.05–0.5 (3 initial, 5 budget, 2 restarts), IQ-TREE seed 76001. Artifacts include FASTA and dates; see [phylogenetic artifacts](OUTPUTS.md#9-phylogenetic-artifacts).

Full development:

```bash
python evaluation/00_synthetic_diagnostics/run.py --config evaluation/00_synthetic_diagnostics/config.yaml --stage all
python evaluation/01_synthetic_baseline/run.py --config evaluation/01_synthetic_baseline/config.yaml --stage develop
```

Sections 7–8: inspect evidence, configure criteria, select, evaluate. Candidate counts depend on development scores. Work scales with methods, replicates, pairs: `n * (n - 1) / 2`. Example: 1,000 cases = 499,500 pairs. Smoke runtime ≠ full-scale estimate.

Commands log `Execution started` and `Execution finished` with `status` and `elapsed_seconds`. Logs go to stderr; capture with stdout. Cached/report commands receive same timing.

Baseline entry points are equivalent:

```bash
epilink-evaluate baseline --config evaluation/01_synthetic_baseline/config.yaml --stage develop
python evaluation/01_synthetic_baseline/run.py --config evaluation/01_synthetic_baseline/config.yaml --stage develop
```

Default stage: `develop`. Output overrides apply to the whole root; use consistently. With `--smoke`, `--output` becomes `*_smoke`.

## 6. Stage reference

**Diagnostics** defaults to `all`:

| Stage | Work |
|-------|------|
| `prepare` | Backbone, truth, development observations |
| `backbone` | Describe backbone cases (no observations) |
| `observations` | GD/TD cells, ambiguity, prevalence, relationships |
| `graphs` | Oracle graphs per M horizon |
| `all` | All stages above |
| `report` | Render from saved tables |

`backbone` describes cases without observations. Baseline requires completed diagnostics (backbone, observations, graphs). Oracle graphs reuse truth/sampled cases across seeds/processes. Unsampled intermediates remain in truth. See [diagnostics protocol](evaluation/00_synthetic_diagnostics/README.md).

**Baseline** stages (all require completed diagnostics):

| Stage | Work | Prerequisite |
|-------|------|--------------|
| `prepare` | Training observations | Completed diagnostics |
| `pairwise` | Logistic models, scores, rankings, cutoffs, calibration | Training/model/score deps |
| `clusters` | Scorers, clustering sweeps | Training/model/score deps |
| `develop` | `pairwise` + `clusters` | — |
| `select` | Freeze settings per criterion | Complete sweep |
| `evaluate` | Apply frozen settings to held-out data | Frozen settings + evidence |
| `report` | Rebuild from saved tables | — |
| `all` | `develop` + `select` + `evaluate` | — |

Stages render reports. Manifest `status: complete` describes the **last requested stage** only. Use coverage tables and manifests to assess the full experiment.

## 7. Find and interpret results

See [output reference](OUTPUTS.md) for table schemas and [worked joins](OUTPUTS.md#11-worked-joins-in-python).

Print run pointers:

```bash
python -m json.tool evaluation/00_synthetic_diagnostics/outputs/diagnostics/current.json
python -m json.tool evaluation/01_synthetic_baseline/outputs/baseline/current.json
```

Open `report.html` in each `run_directory`. Smoke roots: `diagnostics_smoke`, `baseline_smoke`. Pointers identify the most recent run (including partial). Inspect diagnostics coverage first, then backbone, ambiguity, and graph trade-offs.

Baseline layout:

```text
<output-root>/
  current.json
  artifacts/
    models/<id>/, scores/<id>/, trees/<id>/
  runs/<run-id>/
    manifest.json, settings.json, inputs.json, experiment.json, diagnostics.json
    development/
      metrics.csv, summary.csv, frontier.csv
      cutoff_candidates/, resolution_search/, treecluster_search/{raw,dated}/
      seed_<seed>/pairwise/, seed_<seed>/clusters/
    selection/operating_points.json
    evaluation/
      seed_<seed>/pairwise/, seed_<seed>/clusters/
      metrics.csv, summary.csv, operating_results.csv
      heldout_access.json, selection_used.json
    report.md, report.html, figures/
```

Shared artifacts: `evaluation/shared_synthetic/outputs/synthetic[_smoke]/`. Follow `experiment.json` to resolve. `heldout_access/seed_<seed>.json` survives cleanup. See [shared layout](evaluation/shared_synthetic/README.md#artifact-and-provenance-contract).

Files appear as stages complete. `status.json` lists configured/completed counts; `<setting-id>/` has memberships, clusters, metrics, algorithm, manifest.

### Read development evidence in this order

1. **Coverage:** Check `development/seed_<seed>/clusters/status.json` for missing comparisons.
2. **Pairwise discrimination:** Compare scorers within same genetic process using `figures/pairwise_precision_recall_<endpoint>.png` and `pairwise/rankings.csv`. Endpoints: `M0`, `Mle1`, `Mle2`. AP summarizes rankings; metrics in `pairwise/metrics.csv`.
3. **Workload/calibration:** `selected_pairs` and `selected_fraction` in `pairwise/metrics.csv`; `pairwise/calibration.csv` for logistic reliability. Brier/log loss in `rankings.csv`.
4. **Clustering trade-offs:** `components_thresholds_<endpoint>.png`, `leiden_resolution_<endpoint>.png`, `treecluster_thresholds_<endpoint>.png`, `cluster_tradeoffs_<endpoint>.png`. Inspect singleton/largest-cluster behavior with recovery/contamination.
5. **Realization variation:** `development/metrics.csv` (seed-specific), `summary.csv` (means/SDs/ranges), `frontier.csv` (non-dominated settings). Use `settings.json` to translate setting IDs.

M=0: every M>0 pair is false positive. M>=3 contamination counts only distant relationships. Cluster precision includes all within-cluster pairs (including transitive). Empty selection or all-singleton has undefined precision. Logistic calibration applies to probabilities; EpiLink values are raw scores. Smoke SD undefined (one realization).

After evaluation, `operating_results.csv` has per-seed results with criterion names. `operating_summary.csv` includes frozen objective/setting, endpoint metrics, retention, workload, cluster sizes. Reports show actual optimized endpoint from `selection_used.json`. Logistic probabilities are M0-trained. Variability is backbone-conditional.

Refresh report:

```bash
python evaluation/01_synthetic_baseline/run.py --stage report
```

Use `--smoke`, `--config`, `--output` consistently. Report stage reads saved config; editing YAML alone doesn't recompute results.

## 8. Choose and freeze operating criteria

Supplied criteria (`balanced_M0`, `balanced_Mle1`, `balanced_Mle2`) maximize mean development F1. M0 is primary. After reviewing results, edit `selection`:

```yaml
selection:
  criteria:
    - name: balanced_M0
      objective: M0_f1
      constraints: {}
    - name: recall_with_precision_floor
      objective: M0_recall
      constraints:
        M0_precision: { min: 0.5 }
        Mge3_contamination: { max: 0.1 }
```

Bounds are illustrative; choose from intended use and evidence. Objectives are **maximized** (finite numeric columns). Constraints use `min`/`max` and must hold on **every** realization. Use metrics shared by all pipelines.

Selection is per-pipeline/criterion. Feasible settings prefer: highest mean objective, lowest SD, stable setting ID. TreeCluster methods compete within raw/dated pipelines. Infeasible pipelines stay `infeasible`.

```bash
python evaluation/01_synthetic_baseline/run.py --config evaluation/01_synthetic_baseline/config.yaml --stage select
```

Inspect `selection/operating_points.json` and report. Selection records training identity, method definitions, evidence hash. Then evaluate:

```bash
python evaluation/01_synthetic_baseline/run.py --config evaluation/01_synthetic_baseline/config.yaml --stage evaluate
```

Evaluation checks configuration, training, criteria, and evidence match frozen decisions. Before held-out access, changing criteria reuses development evidence. After access, revised analyses need fresh evaluation seeds, new diagnostics, and new selection. Shared ledger enforces this across runs. Reports render from saved outputs. Resets preserve access history.

## 9. Resume work and understand caching

### Project-wide progress logs

Workflows log stage starts, status, and elapsed seconds. Logs cover inputs, truth, observations, diagnostics, pairwise, clustering, phylogeny, TreeCluster, selection, evaluation, perturbation, Boston, and reports. Cache reuse is identified separately.

IQ-TREE milestones relay to terminal; full output in `inference.log`. One-minute heartbeat during quiet phases. Task counts describe completed work, not ETA.

Use `--log-file PATH` to persist logs:

```bash
python evaluation/01_synthetic_baseline/run.py --stage develop --log-file evaluation/01_synthetic_baseline/outputs/development.log
```

### Repeatable pipeline smoke validation

Smoke enables TreeCluster with 64-case namespace:

```bash
python evaluation/00_synthetic_diagnostics/run.py --smoke --stage all
python evaluation/01_synthetic_baseline/run.py --smoke --stage all --phylogeny-timeout 7200 --phylogeny-threads 4
```

IQ-TREE and TreeCluster required. Confirm `treecluster/<process>/raw` and `dated` in metrics. `--stage develop` runs development only; `--stage all` includes selection/evaluation.

### Checkpoint reuse

Resume interrupted work with the same command, config, code, environment, and output root:

```bash
python evaluation/01_synthetic_baseline/run.py --config evaluation/01_synthetic_baseline/config.yaml --stage develop
```

Artifacts reuse when signatures/checksums match manifests. Pairwise checkpoints cover a seed's stage; clustering checkpoints cover settings. Unfinished work retries; "Prepare observations" can appear when reusing artifacts. Abrupt kills may leave status `running`; manifests determine resumable work.

IQ-TREE timeout: retry with longer limit:

```bash
python evaluation/01_synthetic_baseline/run.py --stage develop --phylogeny-timeout 7200
```

`--phylogeny-timeout`: seconds per IQ-TREE/LSD2 call. Applies to `develop`, `clusters`, `select`, `evaluate`, `all`. Preserves completed artifacts. Manifest records limit under `execution_overrides.phylogeny.timeout`. Repeat flag on subsequent attempts.

`--phylogeny-threads N`: thread count for IQ-TREE:

```bash
python evaluation/01_synthetic_baseline/run.py --stage develop --phylogeny-timeout 7200 --phylogeny-threads 4
```

Thread count participates in tree artifact identity. Completed TreeCluster partitions remain checkpointed. Repeat for stages needing new trees.

IQ-TREE creates fresh working directories; incomplete checkpoints don't resume. Two-hour limit is a starting point for ~5,000-case alignment.

Run IDs incorporate config, input/truth identity, implementation, packages, tools. Changes can create new run directories. Absolute input paths participate. Shared artifacts reuse by signature. Keep `manifest.json` for comparisons.

Excluded from run ID: `selection`, `output_directory`, config path. Selection has separate validation. Pointer updates on initialization; not an index of successful runs.

Changing `target_component_size` or `tree_seed` triggers reconstruction. Different targets can select same component. Prebuilt trees without manifest are retained; use new input directory to reconstruct.

### Clean restart after updating phylogeny code

Tree export fixes span this repository and the EpiLink package.

After clearing results, run diagnostics before baseline:

```bash
python evaluation/00_synthetic_diagnostics/run.py --stage all
python evaluation/01_synthetic_baseline/run.py --stage develop --phylogeny-timeout 7200 --phylogeny-threads 4
```

New runs exclude `GD_D`/`GD_S` from Leiden settings (retained in pairwise/components). Implementation fingerprints change; fresh diagnostics required.

## 10. Troubleshooting

| Symptom | Solution |
|---------|----------|
| `epilink-evaluate: command not found` | Activate environment; `pip install -e '.[test]'` |
| `Executable not found` | Run `check`; install tool or set explicit path |
| IQ-TREE failure | Check `backend/run-*/inference.log`; retry with `--phylogeny-timeout 7200` |
| `RecursionError` in tree copy | Update EpiLink and evaluation adapter; rerun diagnostics + baseline |
| CSV parsing fails | Confirm raw paths; `git lfs pull` |
| Tree size unchanged | Check `n_cases` in provenance; different targets can select same component |
| Diagnostics mismatch | Run diagnostics `--stage all` with matching config |
| Report `partial` / incomplete sweep | Check `status.json` and manifests; rerun `clusters` or `develop` |
| IQ-TREE/TreeCluster fails | Check `artifacts/trees/<id>/backend/run-*/` or `treecluster.stderr.log` |
| Timeout | Check logs and timeouts; use `--phylogeny-timeout` to preserve identity |
| No frozen settings | Run `select` before `evaluate` |
| No feasible criterion | Inspect decisions/metrics; revise objectives/bounds |
| Held-out seeds accessed | Before access: rerun `select`. After: fresh seeds, new diagnostics/selection |
| `current.json` missing | Check config/output/smoke; run computational stage first |
| Different run ID on rerun | Compare manifests for config, paths, implementation, packages |
| Incubation error | Gamma shape = `1 / cv²`; `latent_shape` must be smaller |

## 11. Clear outputs with reset-outputs

Clear study results with `reset-outputs`. Removes runs, artifacts, and `current.json`; preserves shared experiment and held-out access.

```bash
epilink-evaluate reset-outputs --dry-run                    # Preview
epilink-evaluate reset-outputs                               # All studies
epilink-evaluate reset-outputs --evaluations diagnostics     # Diagnostics only
epilink-evaluate reset-outputs --evaluations baseline        # Baseline only
epilink-evaluate reset-outputs --evaluations perturbation boston
```

Removes: `runs/`, `artifacts/`, `current.json`.

`--dry-run` previews paths. Shared experiment (`evaluation/shared_synthetic/outputs/`) and Boston `outputs/inputs/` retained. After clearing diagnostics, rerun before baseline. Reset doesn't restore accessed evaluation seeds.

## 12. Perturbation and Boston application

**Perturbation** runs after baseline evaluation. Crosses baseline/matched inference with baseline/updated Leiden resolution:

```bash
python evaluation/02_synthetic_perturbation/run.py --smoke  # or omit --smoke for full study
```

CLI: `epilink-evaluate perturbation --smoke`. Config: `evaluation/02_synthetic_perturbation/config.yaml`. Use `--baseline-run` to pin a completed run.

Four clustering modes (EDD/EDS/ESD/ESS). Graphs include all pairs with scores as weights. Updated resolutions use fresh development seeds, baseline bounds/budget, frozen criterion. Full: 13 scenarios × 4 modes × 4 scorers × 3 seeds = 624 rows. Smoke: 64 cases, 48 rows. Outputs: `evaluation/02_synthetic_perturbation/outputs/perturbation[_smoke]/`.

See [perturbation guide](evaluation/02_synthetic_perturbation/README.md) and [output schema](OUTPUTS.md#12-perturbation-study-outputs). Stages: `all` (default), `report`.

**Boston** applies baseline to real observations. Stages use frozen selection; preparation doesn't need completed baseline.

Prepare inputs:

```bash
python evaluation/03_boston_application/run.py --stage prepare
# or:
epilink-evaluate boston --stage prepare --config evaluation/03_boston_application/config.yaml
```

Run analysis:

```bash
python evaluation/03_boston_application/run.py --stage all
```

| Stage | Work |
|-------|------|
| `prepare` | Write/reuse `outputs/inputs/` tables |
| `trees` | Build/reuse trees, apply frozen TreeCluster |
| `all` | Score pairs, apply graph settings, TreeCluster, assess |
| `report` | Render from `current.json` |

Boston reads metadata, Nextclade, TN93, FASTA from `data/raw/boston/`. Prepared tables: `evaluation/03_boston_application/outputs/inputs/`; results: `outputs/boston/`. TN93 censored at 0.0005/site; missing pairs unobserved.

Boston supports `--config`, `--output`, `--baseline-run`; no `--smoke`. Baseline pointer: `../01_synthetic_baseline/outputs/baseline/current.json`.

```bash
python evaluation/03_boston_application/run.py --stage report
```

See [Boston guide](evaluation/03_boston_application/README.md) and [output schema](OUTPUTS.md#10-boston-inputs-and-results).
