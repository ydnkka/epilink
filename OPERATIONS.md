# EpiLink evaluation: operational guide

Use this guide to configure and run the project, locate results, and resume interrupted work. All shell commands below assume the **repository root** is the working directory.

- [Project overview and implementation status](README.md)
- [Shared synthetic experiment](evaluation/shared_synthetic/README.md)
- [Synthetic diagnostics](evaluation/00_synthetic_diagnostics/README.md)
- [Baseline protocol and metric definitions](evaluation/01_synthetic_baseline/README.md)
- [Perturbation study](evaluation/02_synthetic_perturbation/README.md)
- [Boston empirical application](evaluation/03_boston_application/README.md)
- [Column-level output reference](OUTPUTS.md)

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
    - [Scorers, grids, and comparison settings](#scorers-grids-and-comparison-settings)
  - [4. Prepare or regenerate the SCoVMod tree](#4-prepare-or-regenerate-the-scovmod-tree)
  - [5. Run smoke validation and the baseline](#5-run-smoke-validation-and-the-baseline)
    - [Observed full-run wall times](#observed-full-run-wall-times)
  - [6. Stage reference](#6-stage-reference)
  - [7. Find and interpret results](#7-find-and-interpret-results)
    - [Read development evidence in this order](#read-development-evidence-in-this-order)
  - [8. Choose and freeze operating criteria](#8-choose-and-freeze-operating-criteria)
  - [9. Resume work and understand caching](#9-resume-work-and-understand-caching)
  - [10. Troubleshooting](#10-troubleshooting)
  - [11. Clear outputs with reset-outputs](#11-clear-outputs-with-reset-outputs)
  - [12. Perturbation and Boston application](#12-perturbation-and-boston-application)

## 1. How the pipeline works

The studies live under `evaluation/`. Diagnostics characterizes development observations and known-truth graph/tree controls on a shared synthetic experiment. Baseline compares methods on those observations and freezes settings. Perturbation crosses baseline/matched inference with baseline/updated full-graph EpiLink Leiden resolution for EDD/EDS/ESD/ESS; Boston applies frozen settings to empirical observations. Both downstream studies use the completed baseline directly. Sections 3–9 describe shared preparation, diagnostics and baseline; section 12 gives the downstream commands.

```text
SCoVMod infection and transmission CSVs
  -> reconstruct/select one fixed transmission backbone
     -> full-tree relationship truth (AD, CA, M)
     -> simulate sampling dates and deterministic/stochastic genomes by seed
         -> shared development cases/pairs: genetic distance (GD), temporal distance (TD)
            -> exact feature-cell diagnostics, endpoint-oracle graphs, known hop tree
            -> complete diagnostics gate
            -> EpiLink, genetic-distance, and training-fitted logistic scores
              -> pairwise rankings and threshold metrics
               -> thresholded graphs -> connected components
               -> full observed-pair graphs -> Leiden resolution sweep
            -> aligned sequences + reference -> IQ-TREE trees
              -> raw trees / dated trees (LSD2) -> TreeCluster partitions
     -> compare scores and every within-cluster pair with full-tree truth
        -> development curves, sweeps, and reports
        -> select method-specific settings under common operating criteria
        -> replay frozen settings on held-out observation realizations
```

The transmission backbone supplies truth. IQ-TREE and LSD2 reconstruct comparison trees from simulated observations. These have different roles.

One experiment keeps its transmission backbone fixed. Diagnostics generates development observations; baseline reuses them and generates separate training realizations for logistic fitting. Development realizations determine operating settings. Evaluation observations are generated only after frozen settings are validated by `evaluate`. Unsampled intermediates remain in relationship truth.

The primary target, **M=0**, includes direct transmission and shared-infector pairs. The [protocol](evaluation/01_synthetic_baseline/README.md) defines secondary targets, the eight scorers, and the interpretation of the metrics.

Implementation entry points are [`cli.py`](src/epilink_evaluation/cli.py), [`workflows/diagnostics.py`](src/epilink_evaluation/workflows/diagnostics.py), and [`workflows/baseline.py`](src/epilink_evaluation/workflows/baseline.py). Shared modules live under `src/epilink_evaluation/`, including `inputs`, `diagnostics`, `truth`, `scorers`, `graphs`, `phylogeny`, `clusterers`, `metrics`, `selection`, and `reporting`.

## 2. Environment and source inputs

### Existing checkout

```bash
conda activate epilik_evaluation
python -m pip install -e '.[test]'
```

### Fresh environment

The package declares Python \>=3.10; For a new Conda environment:

```bash
conda create -n epilik_evaluation -c conda-forge python=3.14 pip
conda activate epilik_evaluation
python -m pip install -e '.[test]'
```

The editable install supplies the `epilink-evaluate` command and Python dependencies, including EpiLink 0.1.5, TreeCluster, and pytest. **IQ-TREE ≥2.0.6** is an external executable. Install a build for your platform; where Bioconda supplies it:

```bash
conda install -c conda-forge -c bioconda iqtree
```

The supplied `phylogeny.executable: iqtree` discovers `iqtree3`, `iqtree2`, or `iqtree` on PATH or beside the interpreter. An explicit path pins the executable; the legacy `phylogeny.iqtree_executable` field is also accepted. The same resolved path is recorded and invoked. TreeCluster is a Python dependency. `check` verifies tool availability.

**Boston:** IQ-TREE builds trees directly from the alignment; TN93 is no longer required for tree construction.

### Obtain inputs and check the installation

Tracked CSV, TSV, Parquet, sequence, and several tree formats use Git LFS. On a fresh checkout with Git LFS installed:

```bash
git lfs install
git lfs pull
```

SCoVMod reconstruction needs these files, or equivalent paths configured under `inputs`:

```text
data/raw/scovmod/InfectedIndividuals.1.csv
data/raw/scovmod/TransmissionEvents.1.csv
```

Prepared inputs and run outputs under each study's `outputs/` directory are
ignored by Git. Generate the shared tree in
`evaluation/shared_synthetic/outputs/inputs/` as described below, or configure
`inputs.tree_path` in the shared config to use an existing supplied tree.

```bash
epilink-evaluate check --config evaluation/01_synthetic_baseline/config.yaml
```

`check` prints package versions, executable paths/hashes, configured splits, resolved output location, number of operating definitions, and whether the tree file exists. It performs configuration/tool discovery without simulation. A successful check does not validate raw CSV contents or run the external tools; the smoke workflow exercises their integration.

## 3. Configure an experiment

The data design is [`evaluation/shared_synthetic/config.yaml`](evaluation/shared_synthetic/config.yaml): it owns `inputs`, `generation`, `simulation`, and `splits`. The [diagnostics config](evaluation/00_synthetic_diagnostics/config.yaml) and [baseline config](evaluation/01_synthetic_baseline/config.yaml) both load it through `experiment_config`. Keep diagnostic controls in the former and scorers, method grids, and selection rules in the latter. For another design, copy the shared config and point both study configs at it; select each study config with `--config` on that study's command.

**Path rules:** paths resolve relative to the YAML file that defines them. Shared input paths and `outputs/synthetic` resolve under `evaluation/shared_synthetic/`; each study's `experiment_config` and `output_directory` resolve relative to its own YAML. Thus `outputs/baseline` means `evaluation/01_synthetic_baseline/outputs/baseline`. Moving a config changes its relative paths. CLI `--config` and `--output` paths are relative to the shell's working directory; executable overrides are best made absolute.
The optional `inputs.tree_source_path` is also config-relative; by default it is
the tree's companion `.source.json` file. Input paths are shared across run roots,
so changing a study's `--output` relocates neither prepared inputs nor the shared
experiment root.

### Inputs, simulation, and seeds

| Configuration field                          | Meaning                                                                                                                                  |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `experiment_config`                          | Study-config reference to the shared data-design YAML.                                                                                   |
| `name`                                       | Experiment label, recorded in the run signature.                                                                                         |
| `output_directory`                           | In a study:`artifacts/`, `runs/`, `current.json`; in the shared config: `artifacts/`, `experiments/`, `heldout_access/`, `current.json`. |
| `inputs.tree_path`                           | Transmission backbone to load, or destination when generating a missing tree.                                                            |
| `inputs.infection_path`, `transmission_path` | Raw SCoVMod inputs used for reconstruction.                                                                                              |
| `inputs.target_component_size`               | Requested component size; reconstruction chooses the closest available component.                                                        |
| `inputs.tree_seed`                           | Random infector assignment during reconstruction; affects a newly generated tree.                                                        |
| `inputs.smoke_cases`                         | `null` uses the full backbone; a count uses an ancestor-preserving topological prefix. `--smoke` sets this to 64.                        |
| `simulation.fraction_sampled`                | Fraction of tree cases observed, in`(0, 1]`; full-tree truth is retained.                                                                |
| `simulation.sequence_length`                 | Number of simulated sequence sites; used for Hamming distance denominator and alignment_length default.                                  |
| `simulation.alignment_length`                | Alignment length for TreeCluster threshold scaling (preserves absolute SNP counts). Defaults to `sequence_length`.                       |
| `splits.train`                               | Seeds for observation realizations used to fit logistic models.                                                                          |
| `splits.development`                         | Seeds for parameter-grid comparison and operating-point selection.                                                                       |
| `splits.evaluation`                          | Held-out observation seeds used by`evaluate`.                                                                                            |
| `scorer.seed`, `scorer.mc_samples`           | EpiLink Monte Carlo seed and number of draws.                                                                                            |
| `clustering.leiden.seed`, `restarts`         | Leiden random seed and restarts; restart quality is judged by its declared objective.                                                    |
| `phylogeny.seed`                             | IQ-TREE random seed.                                                                                                                     |
| `phylogeny.model`                            | IQ-TREE substitution model (e.g., JC, MFP).                                                                                              |
| `phylogeny.threads`                          | IQ-TREE parallel threads.                                                                                                                |
| `phylogeny.clock_rate`                       | Fixed LSD2 clock rate (substitutions/site/day); null for estimation.                                                                     |

The `inputs`, `simulation`, and `splits` rows belong to the shared config; scorer and clustering rows belong to baseline. Observation seeds must be nonnegative integers, unique across all three splits. Previously accessed held-out seeds cannot become training/development seeds. Algorithm seeds control inference randomness independently of observation seeds.

### Natural-history parameters

Shared `generation` configures observation simulation. Baseline derives `inference` from that block and requires matched values; edit natural history in the shared config and rerun diagnostics for the changed design.

| Field within`generation` / `inference`  | Meaning and units                                                                         |
| --------------------------------------- | ----------------------------------------------------------------------------------------- |
| `incubation.mean`, `testing_delay.mean` | Mean durations in days.                                                                   |
| `incubation.cv`, `testing_delay.cv`     | Dimensionless coefficient of variation; Gamma shape is`1 / cv²`, scale is `mean / shape`. |
| `latent_shape`                          | Dimensionless latent-stage Gamma shape; must be below the incubation shape.               |
| `symptomatic_rate`, `symptomatic_shape` | Symptomatic removal rate (1/day) and dimensionless Gamma shape.                           |
| `transmission_rate_ratio`               | Dimensionless presymptomatic-to-symptomatic transmission-rate ratio.                      |
| `substitution_rate`                     | Median substitution rate in substitutions/site/year.                                      |
| `relaxation`                            | Dimensionless lognormal SD of branch-specific rates; zero gives a strict clock.           |
| `genome_length`                         | Site count used by EpiLink to calculate mutation-count expectations.                      |

The preserved convention uses `genome_length: 29903` and `simulation.sequence_length: 5000`. Their roles differ: changing either changes the experiment. IQ-TREE infers genetic branch lengths from aligned sequences. LSD2 dates them using numeric simulation days for synthetic data and calendar collection dates for Boston; exported dated branch lengths are days in both cases.

### Scorers, grids, and comparison settings

| Field                                 | Meaning                                                                                                                                                                     |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scorers`                             | Any configured subset of EDD, EDS, ESD, ESS, GD_D, GD_S, LOGIT_D, LOGIT_S. Comparisons should cover both observed genetic processes.                                        |
| `scorer.logistic_C`                   | Fixed inverse regularization strength for logistic fitting.                                                                                                                 |
| `thresholds.epilink`                  | Graph compatibility cutoffs (also pairwise in`configured` mode); retain scores **\>=** the cutoff. Values can exceed one.                                                   |
| `thresholds.genetic`                  | Hamming-distance cutoffs in substitutions; retain distances**\<=** the cutoff.                                                                                              |
| `thresholds.logistic`                 | Probability cutoffs in`[0, 1]`; retain scores **\>=** the cutoff.                                                                                                           |
| `pairwise.selected_fractions`         | Candidate-budget fractions of all observed pairs; whole ties are retained and achieved sizes reported.                                                                      |
| `pairwise.threshold_mode`             | `all_development_scores` (supplied): union of unique development cutoffs plus empty selection; `configured`: use `thresholds`. One shared cutoff is evaluated across seeds. |
| `clustering.algorithms`               | `components`, `leiden`, or both.                                                                                                                                            |
| `clustering.leiden.objective`         | `CPM` or `modularity`; resolution scales depend on objective and scorer.                                                                                                    |
| `clustering.leiden.resolutions`       | Full-graph resolution grid. EpiLink/logistic use original scores including zeros; genetic-distance graphs use unit weights. One `leiden/<scorer>` pipeline per scorer.      |
| `treecluster.enabled`                 | Whether raw and dated phylogenetic comparisons are included.                                                                                                                |
| `treecluster.methods`                 | Methods to sweep:`max_clade`, `avg_clade`, `single_linkage`.                                                                                                                |
| `treecluster.genetic_threshold_snps`  | Raw-tree SNP counts; converted to substitutions/site using `alignment_length` (preserves absolute SNP counts).                                                              |
| `treecluster.temporal_threshold_days` | Dated-tree cutoffs in days, passed directly to TreeCluster.                                                                                                                 |
| `treecluster.executable`              | TreeCluster executable, default `TreeCluster.py`.                                                                                                                           |
| `treecluster.timeout`                 | Timeout for TreeCluster invocation in seconds.                                                                                                                              |
| `phylogeny.model`                     | IQ-TREE substitution model (e.g., JC for synthetic, MFP for Boston).                                                                                                        |
| `phylogeny.threads`                   | IQ-TREE parallel threads.                                                                                                                                                   |
| `phylogeny.seed`                      | IQ-TREE random seed.                                                                                                                                                        |
| `phylogeny.clock_rate`                | Fixed LSD2 clock rate (substitutions/site/day); null for estimation.                                                                                                        |
| `phylogeny.timeout`                   | IQ-TREE/LSD2 timeout in seconds.                                                                                                                                            |
| `phylogeny.executable`                | IQ-TREE name/path; `iqtree` discovers versioned executable names.                                                                                                           |
| `selection.criteria`                  | Objectives and constraints for freezing settings; see section 8.                                                                                                            |
| `grid_audit.objective_tolerance`      | Absolute mean-objective refinement tolerance; supplied value 0.005. A numerical diagnostic, not a confidence interval.                                                      |
| `grid_audit.reference`                | Coarse-grid overrides: `thresholds`, `leiden_resolution_grid`, and/or `treecluster` threshold lists. Compare on the same development observations.                          |

Pairwise and component grids include an explicit empty-selection setting. Inclusive component cutoffs retain zero-valued observations when the cutoff allows them. Full Leiden graphs retain every observed edge, including zero score weights; `threshold` is null and `empty` is false.
Exact pairwise candidates are generated only after all development curves exist;
they never expand the clustering grids. `check` reports the configured clustering
count and marks the total operating count as unknown until these candidates exist.

## 4. Prepare or regenerate the SCoVMod tree

```bash
epilink-evaluate scovmod --stage prepare --config evaluation/01_synthetic_baseline/config.yaml
```

- The default tree and provenance live in `evaluation/shared_synthetic/outputs/inputs/`.
- `scovmod --stage prepare` loads the baseline config's shared input settings; diagnostics uses the same backbone preparation.
- `scovmod` supports only `prepare` (also its default stage), writing the backbone and provenance. Diagnostics prepares the shared experiment, truth, and development observations. Baseline requires completed diagnostics even for `prepare`, then prepares training observations and reuses development data.
- **Matching managed artifact:** reuse the tree after validating the manifest's input/settings signature and output checksums.
- **Missing or stale managed artifact:** reconstruct from the raw CSVs, write the tree, companion `<tree-stem>.source.json`, and `manifest.json`.
- **Explicit prebuilt tree without a manifest:** validate the graph and retain it without requiring raw inputs.
- Full and smoke runs share these inputs. `--output` changes the run root, not the prepared input paths.

To generate a new target while keeping the previous tree, edit these entries in the shared config, keeping its other fields:

```yaml
inputs:
  tree_path: outputs/inputs_target1000_seed12345/transmission_tree.gml
  target_component_size: 1000
  tree_seed: 12345
```

Choose a separate directory to retain the old artifact, then run `scovmod --stage prepare`
again. By default, provenance is saved beside the tree; if using an explicit
`tree_source_path`, update it too. There is no CLI `--force` or target-size
override; these settings live in YAML.

Reconstruction assigns candidate infectors using the seed, keeps one incoming edge per case ordered by earliest time then infector ID, and chooses the weakly connected component closest to the requested size. Equally close sizes prefer the larger component, then the lowest member ID. A spanning transmission tree is constructed from that component. Actual case count can differ from the target; changing the target can also select the same component again.

Inspect the provenance for the example above:

```bash
python -m json.tool evaluation/shared_synthetic/outputs/inputs_target1000_seed12345/transmission_tree.source.json
```

`n_cases` is the actual count. `target_size`, `seed`, input hashes, and `tree_sha256` identify the reconstruction. Use the actual recorded size rather than assuming that it equals the requested target component size. The shared experiment root's `artifacts/truth/<id>/manifest.json` also records actual `n_cases` and `n_pairs`, including any smoke subset. Complete diagnostics for the new design before baseline.

## 5. Run smoke validation and the baseline

First exercise the complete pipeline on a small subset:

```bash
python evaluation/00_synthetic_diagnostics/run.py --config evaluation/00_synthetic_diagnostics/config.yaml --smoke --stage all
python evaluation/01_synthetic_baseline/run.py --config evaluation/01_synthetic_baseline/config.yaml --smoke --stage all
```

Both commands use `--smoke`: up to 64 cases and seeds 71001/72001/73001 for train/development/evaluation, with separate `_smoke` roots. Diagnostics uses resolutions 0.1/0.5, two restarts and hop cutoffs 0/1/2/4. Baseline uses 1,024 Monte Carlo draws, reduced component/tree cutoff grids, full-graph Leiden resolutions 0.05/0.5, two restarts and IQ-TREE seed 76001. Scorers, criteria and the configured clock-rate choice are retained. Observation artifacts include sampled FASTA, reference FASTA and original sampling dates; see [phylogenetic artifacts](OUTPUTS.md#9-phylogenetic-artifacts). Smoke validates pipeline functionality.

Run full development using the configured tree and grids:

```bash
python evaluation/00_synthetic_diagnostics/run.py --config evaluation/00_synthetic_diagnostics/config.yaml --stage all
python evaluation/01_synthetic_baseline/run.py --config evaluation/01_synthetic_baseline/config.yaml --stage develop
```

Then follow sections 7–8 to inspect development evidence, configure criteria, select operating points, and evaluate them. `check` reports the static clustering count; exact pairwise candidate counts depend on development scores. Work scales with the graph/tree grids, replicates, and number of pairs: `n * (n - 1) / 2`. Exact pairwise candidates use cumulative-curve lookups rather than repeated pair scans. For example, 1,000 sampled cases give 499,500 pairs. Smoke runtime is not a full-scale runtime estimate.

### Observed full-run wall times

One historical execution on 2026-10-02–03 produced these timings. It predates the current four-scorer/four-mode clustering-only perturbation design, full-graph Leiden and IQ-TREE/LSD2 changes; these values are not current runtime estimates:

| Workflow/stage                        | Observed elapsed |
| ------------------------------------- | ---------------: |
| Diagnostics`all`                      |   about 5m 10s\* |
| Baseline`develop`                     |       2h 02m 10s |
| Baseline`select`                      |           1m 49s |
| Baseline`evaluate`                    |       1h 16m 33s |
| Baseline`develop`–`evaluate` sequence |      3h 20m 38s† |
| Full perturbation`all`                |     16h 44m 12s‡ |

The baseline and perturbation runs used the 5,051-case backbone, 5,000-nt sequences, eight scorers, and 10,000 EpiLink Monte Carlo draws. Perturbation covered the baseline control plus 12 parameter variants, two inference modes, and three seeds. The measured command windows sum to about 20h 10m, excluding the idle interval between baseline evaluation and the later perturbation run.

These are single-run observations, not guarantees; they were collected on a MacBook Pro (MacBookPro18,3) with an Apple M1 Pro (8 CPU cores: 6 performance and 2 efficiency), 16 GB memory, and macOS 27.0.1. Runtime also depends on tool versions and cache state. Stage times use the first timestamped workflow log through the final report log. The diagnostics estimate (\*) spans the first simulation log to the report file timestamp because command start/end timestamps were not captured. The baseline sequence duration (†) spans the first `develop` log through the `evaluate` report log. Perturbation duration (‡) spans its first scenario log through its final report log.

The two baseline entry points are equivalent:

```bash
epilink-evaluate baseline --config evaluation/01_synthetic_baseline/config.yaml --stage develop
python evaluation/01_synthetic_baseline/run.py --config evaluation/01_synthetic_baseline/config.yaml --stage develop
```

Omitting `--stage` defaults to `develop`. A CLI output override applies to the whole output root; use the same override on subsequent commands. For example:

```bash
python evaluation/01_synthetic_baseline/run.py --stage develop --output evaluation/01_synthetic_baseline/outputs/custom
```

With `--smoke`, that override becomes `evaluation/01_synthetic_baseline/outputs/custom_smoke`.

## 6. Stage reference

**Synthetic diagnostics** defaults to `all`. Its stages are:

| `--stage`      | Work performed                                                                                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prepare`      | Prepare/reuse the shared backbone, truth, and development observations only.                                                                                        |
| `backbone`     | Describe all pinned backbone cases: offspring heterogeneity, inclusive Poisson-percentile superspreading, concentration and generations; no observation generation. |
| `observations` | Exact GD and GD/TD cells, ambiguity, prevalence, and relationship summaries.                                                                                        |
| `graphs`       | One unit-weight oracle graph per M horizon; components and configured Leiden controls.                                                                              |
| `trees`        | Known transmission-hop tree; TreeCluster method/hop-threshold controls evaluated at every endpoint.                                                                 |
| `all`          | Backbone characterisation and all three diagnostic controls, with preparation as a dependency.                                                                      |
| `report`       | Render saved tables and coverage through the diagnostics root's`current.json`.                                                                                      |

The standalone `backbone` stage describes all backbone cases once without generating observations. Other computational stages prepare development data as needed. Baseline is released only when all required diagnostics (backbone, observations, graphs, and trees when enabled) have complete, checksummed coverage of the exact development datasets. A successful diagnostics `prepare` or `backbone` alone is insufficient. Oracle graph/tree controls reuse identical truth and sampled-case sets across seeds and genetic processes; full sampling gives one graph per horizon and one hop tree, rather than independent control replicates. The tree preserves sampled ancestors as zero-length tips and retains unsampled intermediates. Forests are unsupported for this tree control and produce visible failure/incomplete coverage. See the [diagnostics protocol](evaluation/00_synthetic_diagnostics/README.md).

The following table applies to the **synthetic baseline**. All computational stages require a matching shared experiment and completed diagnostics. They reuse shared truth/development observations and prepare their training/model/score dependencies. Baseline `prepare` is optional before `develop`.

| `--stage`  | Work performed                                                                                                            | Prerequisite / main saved output                                                                                                                         |
| ---------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prepare`  | Prepare shared training observations and validate/reuse development observations.                                         | Requires completed diagnostics; writes training datasets into the shared experiment.                                                                     |
| `pairwise` | Fit/reuse training logistic models, score development pairs, and evaluate rankings, thresholds, budgets, and calibration. | Prepares training/model/score dependencies. Writes per-seed`pairwise/` tables.                                                                           |
| `clusters` | Fit/reuse scorers and run development graph/tree clustering sweeps.                                                       | Prepares training/model/score dependencies; writes per-setting memberships/metrics and per-seed status.                                                  |
| `develop`  | Run`pairwise`, then `clusters`.                                                                                           | Writes the development comparison and report.                                                                                                            |
| `select`   | Complete/reuse the declared development comparison and freeze settings under each criterion.                              | Requires a complete configured sweep. Writes`selection/operating_points.json`.                                                                           |
| `evaluate` | Release/generate shared evaluation observations and apply frozen settings using training-fitted models.                   | Requires matching frozen settings and development evidence. Records shared held-out access before observation generation, then writes operating results. |
| `report`   | Rebuild figures and reports from saved tables.                                                                            | Uses`current.json` in the resolved output root; no simulation or setting selection.                                                                      |
| `all`      | Run development, selection, and evaluation sequentially if development succeeds.                                          | Uses the configured criteria immediately, without pausing for development review. Useful for smoke validation.                                           |

Computational stages render a report at the end, including caught failures. The run manifest's `status: complete` describes its **last requested stage**. For example, a complete `prepare` command does not mean clustering or evaluation has run. Use the report's comparison-coverage table and per-stage manifests to assess the whole experiment. Failures during initial input preparation may occur before a run manifest or report can be written.

## 7. Find and interpret results

The [output reference](OUTPUTS.md) defines each saved table's row unit, column names, formulas, units, missing values, and JSON metadata. It also includes [worked joins](OUTPUTS.md#11-worked-joins-in-python) for settings, observations, truth, scores, and cluster memberships.

For diagnostics and baseline, print the latest run pointers:

```bash
python -m json.tool evaluation/00_synthetic_diagnostics/outputs/diagnostics/current.json
python -m json.tool evaluation/01_synthetic_baseline/outputs/baseline/current.json
```

Open `report.html` inside each `run_directory`. Smoke roots are `diagnostics_smoke` and `baseline_smoke`. These pointers identify the most recently initialized run in each root, including partial runs. Inspect diagnostics coverage first, then exact-feature ambiguity, oracle graph trade-offs, and hop-tree threshold curves. Empirical feature-cell ambiguity is not a universal performance ceiling.

Baseline layout:

```text
<output-root>/
  current.json
  artifacts/
    models/<id>/          models.json, manifest.json
    scores/<id>/          scores.parquet, manifest.json
    trees/<id>/           tree files, tool logs, provenance
  runs/<run-id>/
    manifest.json         resolved config, implementation/tool identity, last status
    settings.json         setting_id -> complete method definition
    inputs.json           transmission tree path, checksum, and provenance
    experiment.json       pinned shared experiment_directory and fingerprint
    diagnostics.json      matching diagnostics completion reference
    development/
      metrics.csv, summary.csv, frontier.csv
      grid_adequacy.csv, grid_neighbors.csv
      pairwise_candidates/definitions.json, manifest.json
      seed_<seed>/pairwise/
        evidence/            cached curves and empty-selection metrics
      seed_<seed>/clusters/
    selection/operating_points.json
    evaluation/
      seed_<seed>/pairwise/, seed_<seed>/clusters/
      metrics.csv, summary.csv, frontier.csv
      heldout_access.json, selection_used.json
      operating_results.csv, operating_summary.csv
    report.md, report.html, figures/
```

Shared `artifacts/backbones/`, `artifacts/truth/`, and `artifacts/observations/`
live under `evaluation/shared_synthetic/outputs/synthetic[_smoke]/`. Follow the
baseline run's `experiment.json` to resolve them. That shared root's `current.json`
uses `experiment_directory`, not `run_directory`. Its `heldout_access/seed_<seed>.json`
ledger survives study-output cleanup. See the [shared layout](evaluation/shared_synthetic/README.md#artifact-and-provenance-contract)
and [diagnostic output contract](OUTPUTS.md#13-shared-experiments-and-diagnostics).

Files appear as their stages complete. Reports, aggregate tables, and status files are refreshed by subsequent commands. In each seed's clustering directory, `status.json` lists configured/completed counts and errors; `<setting-id>/` contains `memberships.parquet`, `clusters.parquet`, `metrics.json`, `algorithm.json`, and an artifact manifest.

### Read development evidence in this order

1. **Coverage:** confirm expected seeds and methods completed. Inspect `development/seed_<seed>/clusters/status.json` for missing comparisons.
2. **Pairwise discrimination:** compare scorers within the same observed genetic process using `figures/pairwise_precision_recall_<endpoint>.png` and each seed's `pairwise/rankings.csv`. `<endpoint>` is `M0`, `Mle1`, or `Mle2`. AP summarizes rankings; threshold-specific precision, recall, F1, and selected counts are in `pairwise/metrics.csv`.
3. **Workload and calibration:** inspect `pairwise/budgets.csv` for achieved tie-aware candidate budgets and `pairwise/calibration.csv` for logistic reliability. Brier score and log loss are in `rankings.csv`.
4. **Clustering trade-offs:** use `components_thresholds_<endpoint>.png`, `leiden_resolution_<endpoint>.png`, `treecluster_thresholds_<endpoint>.png`, and `cluster_tradeoffs_<endpoint>.png` under `figures/`. Leiden plots sweep resolution on full graphs. Inspect singleton/largest-cluster behavior alongside recovery and contamination.
5. **Grid adequacy:** inspect `development/grid_adequacy.csv` and `grid_neighbors.csv`. Extend arbitrary search boundaries and refine useful intervals. Keep declared reference clustering settings in the expanded sweep for a complete comparison. A gain within tolerance is only a numerical diagnostic; unchanged grids are labelled `not_refined`, and zero GD is a natural boundary.
6. **Realization variation:** `development/metrics.csv` retains seed-specific results. `summary.csv` provides equal-realization means, SDs, and ranges; `frontier.csv` lists non-dominated mean precision/recall settings by endpoint and pipeline, keeping `_mean` column suffixes. Use `settings.json` to translate a setting ID into thresholds and algorithms.

For M=0, every M\>0 pair is a false positive. M\>=3 contamination measures only distant relationships. Cluster precision includes every within-cluster pair, including pairs connected only transitively by graph edges. An empty selection or all-singleton partition has undefined pair precision; `undefined`/blank values in the reports can therefore be expected. Logistic calibration applies to probabilities; EpiLink values are raw compatibility scores. Baseline SD is undefined when a split has only one realization, as in the smoke workflow.

After evaluation, `evaluation/operating_results.csv` gives per-seed results joined to criterion names. `operating_summary.csv` includes the frozen objective/setting, all three endpoint metrics, AD0/CA00 retention, workload and cluster sizes, with realization and defined-value counts. Reports display each criterion's actual optimized endpoint, resolved from `selection_used.json`, not its name. Logistic probabilities remain M0-trained. Variability is conditional on the fixed backbone; pairs are dependent.

To refresh the latest report from saved results:

```bash
python evaluation/01_synthetic_baseline/run.py --stage report
```

Use `--smoke`, `--config`, and/or `--output` consistently to locate the intended root. The report stage follows that root's pointer and reads the run's saved config; editing YAML alone does not recompute results or change a saved decision.

## 8. Choose and freeze operating criteria

The supplied `balanced_M0`, `balanced_Mle1`, and `balanced_Mle2` criteria maximize their respective mean development F1 values. M0 remains primary. After reviewing development results, edit the `selection` block. This complete example compares balanced M0 F1 with a recall objective subject to precision and distant contamination bounds:

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

The numerical bounds are illustrative; choose them from the intended use and development evidence. Objectives are **maximized** and must be finite numeric metric columns. Constraints accept `min` and/or `max` and must hold on **every** development realization. Use metrics shared by all configured pipelines when comparing pairwise and clustering settings together.

Selection is independent for each pipeline and criterion. Among feasible settings it prefers highest mean objective, then lowest between-realization SD, then the stable setting ID. TreeCluster methods compete jointly within each raw/dated observed-process pipeline. Infeasible pipelines stay labeled `infeasible`; bounds are never automatically relaxed.

```bash
python evaluation/01_synthetic_baseline/run.py --config evaluation/01_synthetic_baseline/config.yaml --stage select
```

Inspect `selection/operating_points.json` and the updated report. Selection also records the training identity, exact method definitions, and development evidence hash. Then replay the selected settings:

```bash
python evaluation/01_synthetic_baseline/run.py --config evaluation/01_synthetic_baseline/config.yaml --stage evaluate
```

Evaluation checks that configuration, training, criteria, and development evidence match the frozen decisions. Before evaluation access, changing only selection criteria allows reuse of development evidence. After held-out access, revised analyses require fresh evaluation seeds in the shared config, diagnostics for the updated experiment, and new baseline selection. The shared ledger enforces this across replacement study runs, not just within one run directory. The supplied revised full design uses 63101–63103, reserving 63001–63003 for the previous completed comparison. Report-only endpoint corrections can be rendered from saved outputs without new observations. Resetting outputs preserves access history; smoke is repeatable validation in its separate namespace.

## 9. Resume work and understand caching

To resume interrupted development, repeat the same command with the same config, code, environment, and output root:

```bash
python evaluation/01_synthetic_baseline/run.py --config evaluation/01_synthetic_baseline/config.yaml --stage develop
```

Artifacts are reused when their signatures and file checksums match completed manifests. Pairwise checkpoints cover a seed's pairwise stage; clustering checkpoints cover individual settings. Unfinished work is retried, so progress messages such as “Prepare observations” can appear even when a valid artifact is being reused. A process killed abruptly may leave the last run status `running`; artifact manifests determine what can be resumed.

Run IDs incorporate scientific configuration, input/truth identity, implementation hashes, recorded package versions, and external-tool identities. Changes to these can create a new run directory. Absolute resolved input paths also participate, so moving a checkout can change its run ID. Shared artifacts are reused according to their own signatures. Keep the per-run `manifest.json` when comparing runs.

`selection`, `output_directory`, and the config-file path itself are excluded from the scientific configuration portion of the run ID. Selection decisions have their own validation; changing the output root changes where artifacts are found. The root-level pointer is updated when a run is initialized. Earlier run directories remain available; the pointer is not an index of successful runs.

Changing `target_component_size` or `tree_seed` invalidates a managed backbone's preparation signature and triggers reconstruction. Different targets can still select the same component. Explicit prebuilt trees without a manifest are retained; use a separate input directory to reconstruct a different backbone while keeping the previous one.

Retained outputs from earlier versions are historical results. Produce current evidence using diagnostics followed by baseline; old baseline-local truth and observation directories do not establish the shared-experiment completion contract.

## 10. Troubleshooting

| Symptom                                                              | What to check / next action                                                                                                                                                    |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `epilink-evaluate: command not found` or module import failure       | Activate the environment used for installation and run`python -m pip install -e '.[test]'`. `python -m epilink_evaluation --help` uses that interpreter directly.              |
| `Executable not found`                                               | Run `check`, install the missing tool, or set `phylogeny.executable` / `treecluster.executable` to an explicit path.                                                           |
| IQ-TREE inference failure                                            | Inspect `backend/run-*/inference.log`; check alignment/reference compatibility and complete sampling dates. Adjust threads or increase timeout if needed.                      |
| CSV parsing fails on a fresh checkout                                | Confirm raw paths and Git LFS downloads. An LFS pointer contains metadata rather than the input table; run`git lfs pull` after installing Git LFS.                             |
| Tree size did not change                                             | Inspect`n_cases` in the provenance; different targets can select the same component. Prebuilt trees without a manifest are retained; use a new input directory to reconstruct. |
| Baseline requests diagnostics or reports mismatched development data | Run diagnostics`--stage all` with the matching shared config and smoke mode; inspect its coverage and failures.                                                                |
| Diagnostics tree control rejects a forest                            | Use a single rooted transmission tree for this control; no between-component hop distance is defined. The failure remains visible in coverage.                                 |
| Report says`partial` / selection reports an incomplete sweep         | Inspect`seed_<seed>/clusters/status.json` and failed setting manifests. Fix the cause and rerun `clusters` or `develop`.                                                       |
| IQ-TREE or TreeCluster fails                                         | Inspect`<output-root>/artifacts/trees/<id>/backend/run-*/` for IQ-TREE logs, or `<setting-id>/treecluster.stderr.log`. Commands and paths are saved in manifests.              |
| TreeCluster fails                                                    | Inspect`<run>/development/seed_<seed>/clusters/<setting-id>/treecluster.stderr.log` and its manifest; evaluation uses the analogous evaluation path.                           |
| External command times out                                           | Inspect its stderr log and`treecluster.command_timeout_seconds`. Increasing the configured timeout changes the experiment signature and can create a new run.                  |
| No frozen operating settings                                         | Run`select` for the exact experiment/output root before `evaluate`.                                                                                                            |
| `No operating criterion is feasible`                                 | Inspect the frozen decisions and development metrics; revise objectives, bounds, or grids using development evidence.                                                          |
| Criteria changed or held-out seeds already accessed                  | Before access, rerun`select`. After access, assign fresh evaluation seeds in the shared config and rerun diagnostics and baseline selection; reset preserves the ledger.       |
| `current.json` missing when running `report`                         | Check config/output/smoke arguments. A computational stage must initialize that root first.                                                                                    |
| A rerun writes a different run ID                                    | Compare manifests for config, input paths/hashes, implementation, package, and executable changes.                                                                             |
| Incubation parameter error                                           | The Gamma incubation shape is`1 / cv²`; EpiLink requires `latent_shape` to be smaller. Check the complete matched natural-history block.                                       |

## 11. Clear outputs with reset-outputs

To clear study results, use `reset-outputs`. It removes runs, study artifacts, and the `current.json` pointer from the selected standard study roots, while preserving the shared experiment and held-out access history.

```bash
# Preview what would be deleted (recommended first)
epilink-evaluate reset-outputs --dry-run

# Clear all study outputs (diagnostics, baseline, perturbation, boston)
epilink-evaluate reset-outputs

# Clear only diagnostics outputs (includes diagnostics_smoke)
epilink-evaluate reset-outputs --evaluations diagnostics

# Clear only baseline outputs (includes baseline_smoke)
epilink-evaluate reset-outputs --evaluations baseline

# Clear perturbation and boston outputs
epilink-evaluate reset-outputs --evaluations perturbation boston
```

The command removes:

- All run directories under `runs/`
- All artifact directories under `artifacts/`
- The `current.json` pointer file

Use `--dry-run` to inspect the selected paths. The command targets the standard run roots listed above; `--output` does not select a custom cleanup location. The entire `evaluation/shared_synthetic/outputs/` area is retained, including prepared inputs, shared backbones/truth/observations, experiment manifests, and both full/smoke held-out access ledgers. Boston's `outputs/inputs/` is also retained. After clearing diagnostics, rerun it before baseline; the surviving shared marker alone cannot validate deleted diagnostic evidence. Reset does not make accessed evaluation seeds available for revised selection.

## 12. Perturbation and Boston application

**EpiLink clustering sensitivity is available after baseline evaluation completes.** The perturbation entry point crosses baseline/matched inference with baseline/updated full-graph Leiden resolution:

```bash
python evaluation/02_synthetic_perturbation/run.py --smoke
python evaluation/02_synthetic_perturbation/run.py
```

The equivalent CLI is `epilink-evaluate perturbation --smoke`. Its default configuration is `evaluation/02_synthetic_perturbation/config.yaml`; the baseline `check` command expects a baseline configuration. Perturbation validates its reference and levels at startup. Use `--baseline-run` to pin a completed run directory instead of the default current-baseline pointer.

The workflow evaluates EDD/EDS/ESD/ESS clustering in four modes. Graphs include every observed pair with its score as weight and have no cutoff. Updated resolutions use separate fresh development seeds and the baseline criterion/grid; all arms use paired evaluation seeds and fresh controls. Full coverage is 13 scenarios × four modes × four scorers × three evaluation seeds = 624 rows. Smoke uses up to 64 cases and the incubation-mean levels, yielding 48 rows. Outputs are under `evaluation/02_synthetic_perturbation/outputs/perturbation[_smoke]/`.

See the [perturbation guide](evaluation/02_synthetic_perturbation/README.md) for schema-2 configuration, resumption and migration from thresholded references, and the [output schema](OUTPUTS.md#12-perturbation-study-outputs) for paired results. Perturbation accepts `all` (the default, including development selection) and `report` stages.

**Boston applies the baseline reference to real observations.** Its computational stages use frozen selection and held-out evaluation provenance; preparation alone does not need a completed baseline. Prepare only the derived input tables with:

```bash
python evaluation/03_boston_application/run.py --stage prepare
# Equivalent installed preparation command (uses the Boston config):
epilink-evaluate boston --stage prepare --config evaluation/03_boston_application/config.yaml
```

Run the frozen transfer analysis with:

```bash
python evaluation/03_boston_application/run.py --stage all
```

| Boston stage    | Work performed                                                                                                    |
| --------------- | ----------------------------------------------------------------------------------------------------------------- |
| `prepare`       | Write or reuse`outputs/inputs/` tables and provenance.                                                            |
| `trees`         | Build/reuse raw and dated trees and apply frozen TreeCluster settings. Graph scoring/clustering is not requested. |
| `all` (default) | Score pairs, apply frozen graph settings, run enabled TreeCluster, and assess partitions.                         |
| `report`        | Render saved results for the run identified by the Boston root's`current.json`.                                   |

Boston reads metadata, Nextclade, TN93 distances, and the aligned FASTA under `data/raw/boston/`. Prepared tables are under `evaluation/03_boston_application/outputs/inputs/`; run results are under `evaluation/03_boston_application/outputs/boston/`. The TN93 table is censored at 0.0005/site; missing pairs remain unobserved, not zero. A complete tree-only stage does not establish graph coverage.

Boston supports `--config`, `--output`, and `--baseline-run`, but not `--smoke`. YAML paths are relative to the Boston config; CLI overrides are relative to the working directory. The sibling baseline pointer is `../01_synthetic_baseline/outputs/baseline/current.json` in the supplied YAML. Repeat the same configuration and overrides to resume a run.

```bash
python evaluation/03_boston_application/run.py --stage report
python -m json.tool evaluation/03_boston_application/outputs/boston/current.json
```

See the [Boston guide](evaluation/03_boston_application/README.md) and [Boston output schema](OUTPUTS.md#10-boston-inputs-and-results) for interpretation.
