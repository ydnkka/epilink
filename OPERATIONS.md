# EpiLink evaluation: operational guide

Use this guide to configure and run the project, locate results, and resume interrupted work. All shell commands below assume the **repository root** is the working directory.

- [Project overview and implementation status](README.md)
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
  - [6. Stage reference](#6-stage-reference)
  - [7. Find and interpret results](#7-find-and-interpret-results)
    - [Read development evidence in this order](#read-development-evidence-in-this-order)
  - [8. Choose and freeze operating criteria](#8-choose-and-freeze-operating-criteria)
  - [9. Resume work and understand caching](#9-resume-work-and-understand-caching)
  - [10. Troubleshooting](#10-troubleshooting)
  - [11. Clear outputs with reset-outputs](#11-clear-outputs-with-reset-outputs)
  - [12. Perturbation and Boston application](#12-perturbation-and-boston-application)

## 1. How the pipeline works

The three studies live under `evaluation/`. The baseline compares methods against known transmission relationships and freezes operating points. Perturbation tests their sensitivity to biological parameter changes; Boston examines empirical transfer and clustering-parameter sensitivity. Both downstream studies use the completed baseline directly. Sections 3–9 describe the baseline; section 12 gives the downstream commands and their distinct stage behavior.

```text
SCoVMod infection and transmission CSVs
  -> reconstruct/select one fixed transmission backbone
     -> full-tree relationship truth (AD, CA, M)
     -> simulate sampling dates and deterministic/stochastic genomes by seed
        -> sampled cases and all unordered pairs: genetic distance (GD), time (TD)
           -> EpiLink, genetic-distance, and logistic scores
              -> pairwise rankings and threshold metrics
              -> thresholded graphs -> components / Leiden partitions
           -> genetic-distance trees (FastME)
              -> raw trees / dated trees (TreeTime) -> TreeCluster partitions
     -> compare scores and every within-cluster pair with full-tree truth
        -> development curves, sweeps, and reports
        -> select method-specific settings under common operating criteria
        -> replay frozen settings on held-out observation realizations
```

The transmission backbone supplies truth. FastME and TreeTime reconstruct comparison trees from simulated observations. These have different roles.

One experiment keeps its transmission backbone fixed. Training, development, and evaluation seeds generate different observation realizations on that backbone. Logistic regression is fitted on training realizations; development realizations determine operating settings; evaluation realizations measure fixed-setting performance. Unsampled intermediates remain in relationship truth.

The primary target, **M=0**, includes direct transmission and shared-infector pairs. The [protocol](evaluation/01_synthetic_baseline/README.md) defines secondary targets, the eight scorers, and the interpretation of the metrics.

Implementation entry points are [`cli.py`](src/epilink_evaluation/cli.py) and [`workflows/baseline.py`](src/epilink_evaluation/workflows/baseline.py). Shared modules live under `src/epilink_evaluation/`: `inputs`, `truth`, `scorers`, `graphs`, `phylogeny`, `clusterers`, `metrics`, `selection`, and `reporting`.

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

The editable install supplies the `epilink-evaluate` command and Python dependencies, including EpiLink 0.1.5, TreeCluster, TreeTime, pytest, and BCubed. FastME is a separate executable. Install a build for your platform; where Bioconda supplies it:

```bash
conda install -c conda-forge -c bioconda fastme
```

The runner searches PATH first, then the active interpreter's directory. To specify executables explicitly, edit `treecluster.executables` in the config, using executable names or absolute paths. Each value identifies one executable, without extra command arguments.

Boston tree construction also requires the standalone `tn93` executable. Install it separately (for example, `conda install -c bioconda tn93` where available), or set `trees.tn93_executable` in the Boston config. The baseline `check` command checks baseline tools; it does not check Boston's TN93 dependency.

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
ignored by Git. Generate the baseline tree in
`evaluation/01_synthetic_baseline/outputs/inputs/` as described below, or configure
`inputs.tree_path` to use an existing supplied tree.

```bash
epilink-evaluate check --config evaluation/01_synthetic_baseline/config.yaml
```

`check` prints package versions, executable paths/hashes, configured splits, resolved output location, number of operating definitions, and whether the tree file exists. It performs configuration/tool discovery without simulation. A successful check does not validate raw CSV contents or run the external tools; the smoke workflow exercises their integration.

## 3. Configure an experiment

The default configuration is [`evaluation/01_synthetic_baseline/config.yaml`](evaluation/01_synthetic_baseline/config.yaml). For another experiment, save a complete copy alongside it, edit that copy, and pass it consistently with `--config evaluation/01_synthetic_baseline/my_experiment.yaml`.

**Path rules:** `output_directory`, `inputs.tree_path`, `inputs.infection_path`, and `inputs.transmission_path` are resolved relative to the YAML file. Thus `outputs/baseline` in the default config means `evaluation/01_synthetic_baseline/outputs/baseline`. Moving a config to a different directory changes those relative paths. CLI `--config` and `--output` paths are relative to the shell's working directory; executable overrides are best made absolute.
The optional `inputs.tree_source_path` is also config-relative; by default it is
the tree's companion `.source.json` file. Input paths are shared across run roots,
so changing `--output` does not relocate prepared inputs.

### Inputs, simulation, and seeds

| Configuration field                              | Meaning                                                                                                               |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `name`                                         | Experiment label, recorded in the run signature.                                                                      |
| `output_directory`                             | Root containing`artifacts/`, `runs/`, and `current.json`.                                                       |
| `inputs.tree_path`                             | Transmission backbone to load, or destination when generating a missing tree.                                         |
| `inputs.infection_path`, `transmission_path` | Raw SCoVMod inputs used for reconstruction.                                                                           |
| `inputs.target_component_size`                 | Requested component size; reconstruction chooses the closest available component.                                     |
| `inputs.tree_seed`                             | Random infector assignment during reconstruction; affects a newly generated tree.                                     |
| `inputs.smoke_cases`                           | `null` uses the full backbone; a count uses an ancestor-preserving topological prefix. `--smoke` sets this to 64. |
| `simulation.fraction_sampled`                  | Fraction of tree cases observed, in`(0, 1]`; full-tree truth is retained.                                           |
| `simulation.sequence_length`                   | Number of simulated sequence sites, also the denominator for FastME distances.                                        |
| `splits.train`                                 | Seeds for observation realizations used to fit logistic models.                                                       |
| `splits.development`                           | Seeds for parameter-grid comparison and operating-point selection.                                                    |
| `splits.evaluation`                            | Held-out observation seeds used by`evaluate`.                                                                       |
| `scorer.seed`, `scorer.mc_samples`           | EpiLink Monte Carlo seed and number of draws.                                                                         |
| `clustering.leiden.seed`, `restarts`         | Leiden random seed and restarts; restart quality is judged by its declared objective.                                 |
| `treecluster.rng_seed`                         | TreeTime random seed.                                                                                                 |

Observation seeds must be nonnegative integers, unique across all three splits. Algorithm seeds control inference randomness independently of observation seeds.

### Natural-history parameters

`generation` configures observation simulation; `inference` configures EpiLink. The active baseline requires equal values in these two sections. The supplied YAML anchor, `generation: &natural_history` and `inference: *natural_history`, keeps them matched when the generation block is edited.

| Field within`generation` / `inference`  | Meaning and units                                                                              |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `incubation.mean`, `testing_delay.mean` | Mean durations in days.                                                                        |
| `incubation.cv`, `testing_delay.cv`     | Dimensionless coefficient of variation; Gamma shape is`1 / cv²`, scale is `mean / shape`. |
| `latent_shape`                            | Dimensionless latent-stage Gamma shape; must be below the incubation shape.                    |
| `symptomatic_rate`, `symptomatic_shape` | Symptomatic removal rate (1/day) and dimensionless Gamma shape.                                |
| `transmission_rate_ratio`                 | Dimensionless presymptomatic-to-symptomatic transmission-rate ratio.                           |
| `substitution_rate`                       | Median substitution rate in substitutions/site/year.                                           |
| `relaxation`                              | Dimensionless lognormal SD of branch-specific rates; zero gives a strict clock.                |
| `genome_length`                           | Site count used by EpiLink to calculate mutation-count expectations.                           |

The preserved convention uses `genome_length: 29903` and `simulation.sequence_length: 5000`. Their roles differ: changing either changes the experiment. TreeTime estimates its clock from the generated observations.

### Scorers, grids, and comparison settings

| Field                                              | Meaning                                                                                                                                 |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `scorers`                                        | Any configured subset of EDD, EDS, ESD, ESS, GD_D, GD_S, LOGIT_D, LOGIT_S. Comparisons should cover both observed genetic processes.    |
| `scorer.logistic_C`                              | Fixed inverse regularization strength for logistic fitting.                                                                             |
| `thresholds.epilink`                             | Raw compatibility cutoffs; retain scores**\\>=** the cutoff. Values can exceed one.                                               |
| `thresholds.genetic`                             | Hamming-distance cutoffs in substitutions; retain distances**\\<=** the cutoff.                                                   |
| `thresholds.logistic`                            | Probability cutoffs in`[0, 1]`; retain scores **\\>=** the cutoff.                                                              |
| `pairwise.selected_fractions`                    | Candidate-budget fractions of all observed pairs; whole ties are retained and achieved sizes reported.                                  |
| `clustering.algorithms`                          | `components`, `leiden`, or both.                                                                                                    |
| `clustering.leiden.objective`                    | `CPM` or `modularity`; resolution scales depend on the objective and weight policy.                                                 |
| `clustering.leiden.weight_policies`              | `binary` and/or `native`; native EpiLink/logistic weights retain positive scores. Genetic graphs use binary weights.                |
| `clustering.leiden.resolutions`                  | Resolution grid crossed with graph thresholds for each scorer/weight policy.                                                            |
| `treecluster.enabled`                            | Whether raw and dated phylogenetic comparisons are included.                                                                            |
| `treecluster.methods`                            | Methods to sweep:`max_clade`, `avg_clade`, `single_linkage`.                                                                      |
| `treecluster.genetic_thresholds`                 | Raw-tree branch-distance cutoffs as integer SNP counts; converted internally to substitutions/site using`simulation.sequence_length`. |
| `treecluster.threshold_days`, `days_per_year`  | Dated-tree cutoffs in days, divided by days/year before TreeCluster.                                                                    |
| `treecluster.fastme_method`                      | FastME method code passed through`-m`; the supplied value is `N`.                                                                   |
| `treecluster.raw_rooting`, `negative_branches` | Supported policies:`midpoint` and `clip_zero`. Clipped-branch counts are recorded.                                                  |
| `treecluster.clock_filter`                       | Clock-filter value passed to TreeTime.                                                                                                  |
| `treecluster.command_timeout_seconds`            | Timeout for each external tool invocation.                                                                                              |
| `selection.criteria`                             | Objectives and constraints for freezing settings; see section 8.                                                                        |

The runner also adds an explicit empty-selection setting to each scorer's grid. Binary graphs include zero-valued edges at a zero compatibility/probability cutoff; native weighted graphs omit zero-weight edges.

## 4. Prepare or regenerate the SCoVMod tree

```bash
epilink-evaluate scovmod --stage prepare --config evaluation/01_synthetic_baseline/config.yaml
```

- The default tree and provenance live in `evaluation/01_synthetic_baseline/outputs/inputs/`.
- `scovmod --stage prepare` and baseline initialization use the same preparation and configured input paths, seed, and target size.
- `scovmod` supports only `prepare` (also its default stage). It writes the backbone and provenance; `baseline --stage prepare` additionally creates truth and training/development observations.
- **Matching managed artifact:** reuse the tree after validating the manifest's input/settings signature and output checksums.
- **Missing or stale managed artifact:** reconstruct from the raw CSVs, write the tree, companion `<tree-stem>.source.json`, and `manifest.json`.
- **Explicit prebuilt tree without a manifest:** validate the graph and retain it without requiring raw inputs.
- Full and smoke runs share these inputs. `--output` changes the run root, not the prepared input paths.

To generate a new target while keeping the previous tree, edit these entries in the existing config, keeping its other fields:

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
python -m json.tool evaluation/01_synthetic_baseline/outputs/inputs_target1000_seed12345/transmission_tree.source.json
```

`n_cases` is the actual count. `target_size`, `seed`, input hashes, and `tree_sha256` identify the reconstruction. Use the actual recorded size rather than assuming that it equals the requested target component size. The `artifacts/truth/<id>/manifest.json` in a baseline run also records its actual `n_cases` and `n_pairs`, including any smoke subset.

## 5. Run smoke validation and the baseline

First exercise the complete pipeline on a small subset:

```bash
python evaluation/01_synthetic_baseline/run.py --config evaluation/01_synthetic_baseline/config.yaml --smoke --stage all
python -m pytest -q
```

Smoke mode uses up to 64 backbone cases, one observation seed per split (71001/72001/73001), 1,024 Monte Carlo draws, smaller threshold/resolution grids, two Leiden restarts, and TreeTime seed 76001. It appends `_smoke` to the output root, preserving a separate namespace. It retains configured scorers, algorithms, and operating criteria. Smoke results establish pipeline functionality.

Run full development using the configured tree and grids:

```bash
python evaluation/01_synthetic_baseline/run.py --config evaluation/01_synthetic_baseline/config.yaml --stage develop
```

Then follow sections 7–8 to inspect development evidence, configure criteria, select operating points, and evaluate them. The supplied grids define 1,876 operating settings per development realization; `check` reports the count for your configuration. Work scales with the grids, replicates, and number of pairs: `n * (n - 1) / 2`. For example, 1,000 sampled cases give 499,500 pairs. Smoke runtime is not a full-scale runtime estimate.

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

The following table applies to the **synthetic baseline**. Stages load or create their required truth, observations, fitted models, and scores automatically. `prepare` is optional before `develop`.

| `--stage`  | Work performed                                                                                                            | Prerequisite / main saved output                                                                                                |
| ------------ | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `prepare`  | Prepare full-tree truth and training/development observations.                                                            | Configured tree, or raw inputs to generate it. Writes truth and observation artifacts.                                          |
| `pairwise` | Fit/reuse training logistic models, score development pairs, and evaluate rankings, thresholds, budgets, and calibration. | Creates missing prerequisites. Writes per-seed`pairwise/` tables.                                                             |
| `clusters` | Fit/reuse scorers and run development graph/tree clustering sweeps.                                                       | Creates missing prerequisites; does not run pairwise metric tables. Writes per-setting memberships/metrics and per-seed status. |
| `develop`  | Run`pairwise`, then `clusters`.                                                                                       | Writes the development comparison and report.                                                                                   |
| `select`   | Complete/reuse the declared development comparison and freeze settings under each criterion.                              | Requires a complete configured sweep. Writes`selection/operating_points.json`.                                                |
| `evaluate` | Apply frozen settings to evaluation observations using training-fitted models.                                            | Requires matching frozen settings and development evidence. Writes held-out operating results.                                  |
| `report`   | Rebuild figures and reports from saved tables.                                                                            | Uses`current.json` in the resolved output root; no simulation or setting selection.                                           |
| `all`      | Run development, selection, and evaluation sequentially if development succeeds.                                          | Uses the configured criteria immediately, without pausing for development review. Useful for smoke validation.                  |

Computational stages render a report at the end, including caught failures. The run manifest's `status: complete` describes its **last requested stage**. For example, a complete `prepare` command does not mean clustering or evaluation has run. Use the report's comparison-coverage table and per-stage manifests to assess the whole experiment. Failures during initial input preparation may occur before a run manifest or report can be written.

## 7. Find and interpret results

The [output reference](OUTPUTS.md) defines each saved table's row unit, column names, formulas, units, missing values, and JSON metadata. It also includes [worked joins](OUTPUTS.md#11-worked-joins-in-python) for settings, observations, truth, scores, and cluster memberships.

For the default output root, print the latest run pointer:

```bash
python -m json.tool evaluation/01_synthetic_baseline/outputs/baseline/current.json
```

Open `report.html` inside its `run_directory`. For smoke output, use `evaluation/01_synthetic_baseline/outputs/baseline_smoke/current.json`. These pointers identify the most recently initialized run in each root, including partial runs.

```text
<output-root>/
  current.json
  artifacts/
    truth/<id>/           relationships.parquet, nodes.parquet, manifest.json
    observations/<id>/    pairs.parquet, cases.parquet, manifest.json
    models/<id>/          models.json, manifest.json
    scores/<id>/          scores.parquet, manifest.json
    trees/<id>/           tree files, tool logs, provenance
  runs/<run-id>/
    manifest.json         resolved config, implementation/tool identity, last status
    settings.json         setting_id -> complete method definition
    inputs.json           transmission tree path, checksum, and provenance
    development/
      metrics.csv, summary.csv, frontier.csv
      seed_<seed>/pairwise/
      seed_<seed>/clusters/
    selection/operating_points.json
    evaluation/
      seed_<seed>/pairwise/, seed_<seed>/clusters/
      metrics.csv, summary.csv, frontier.csv
      heldout_access.json, selection_used.json
      operating_results.csv, operating_summary.csv
    report.md, report.html, figures/
```

Files appear as their stages complete. Reports, aggregate tables, and status files are refreshed by subsequent commands. In each seed's clustering directory, `status.json` lists configured/completed counts and errors; `<setting-id>/` contains `memberships.parquet`, `clusters.parquet`, `metrics.json`, `algorithm.json`, and an artifact manifest.

### Read development evidence in this order

1. **Coverage:** confirm expected seeds and methods completed. Inspect `development/seed_<seed>/clusters/status.json` for missing comparisons.
2. **Pairwise discrimination:** compare scorers within the same observed genetic process using `figures/pairwise_precision_recall.png` and each seed's `pairwise/rankings.csv`. AP summarizes rankings; threshold-specific precision, recall, F1, and selected counts are in `pairwise/metrics.csv`.
3. **Workload and calibration:** inspect `pairwise/budgets.csv` for achieved tie-aware candidate budgets and `pairwise/calibration.csv` for logistic reliability. Brier score and log loss are in `rankings.csv`.
4. **Clustering trade-offs:** use `components_thresholds.png`, `leiden_threshold_resolution.png`, `treecluster_thresholds.png`, and `cluster_tradeoffs.png` under `figures/`. Inspect singleton/largest-cluster behavior alongside recovery and contamination. Broaden useful regions that touch grid boundaries during development.
5. **Realization variation:** `development/metrics.csv` retains seed-specific results. `summary.csv` provides equal-realization means, SDs, and ranges; `frontier.csv` lists non-dominated mean precision/recall settings by pipeline. Use `settings.json` to translate a setting ID into thresholds and algorithms.

For M=0, every M\>0 pair is a false positive. M\>=3 contamination measures only distant relationships. Cluster precision includes every within-cluster pair, including pair

| col1 | col2 | col3 |
| ---- | ---- | ---- |
|      |      |      |
|      |      |      |

s connected only transitively by graph edges. An empty selection or all-singleton partition has undefined pair precision; `undefined`/blank values in the reports can therefore be expected. Logistic calibration applies to probabilities; EpiLink values are raw compatibility scores. SD is undefined when a split has only one realization, as in the smoke workflow.

After evaluation, `evaluation/operating_results.csv` gives per-seed results joined to criterion names, and `operating_summary.csv` summarizes them. `selection_used.json` records the frozen decisions actually replayed. Variability is conditional on the fixed backbone; pairs are dependent.

To refresh the latest report from saved results:

```bash
python evaluation/01_synthetic_baseline/run.py --stage report
```

Use `--smoke`, `--config`, and/or `--output` consistently to locate the intended root. The report stage follows that root's pointer and reads the run's saved config; editing YAML alone does not recompute results or change a saved decision.

## 8. Choose and freeze operating criteria

The supplied `balanced_M0` criterion maximizes mean development M=0 F1. After reviewing development results, edit the `selection` block. This complete example compares balanced F1 with a recall objective subject to precision and distant contamination bounds:

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

Evaluation checks that configuration, training, criteria, and development evidence match the frozen decisions. Before evaluation access, changing only selection criteria allows reuse of development evidence. After held-out access, revised criteria require fresh evaluation seeds and a newly selected experiment; the runner rejects replacement decisions within an already evaluated run.

## 9. Resume work and understand caching

To resume interrupted development, repeat the same command with the same config, code, environment, and output root:

```bash
python evaluation/01_synthetic_baseline/run.py --config evaluation/01_synthetic_baseline/config.yaml --stage develop
```

Artifacts are reused when their signatures and file checksums match completed manifests. Pairwise checkpoints cover a seed's pairwise stage; clustering checkpoints cover individual settings. Unfinished work is retried, so progress messages such as “Prepare observations” can appear even when a valid artifact is being reused. A process killed abruptly may leave the last run status `running`; artifact manifests determine what can be resumed.

Run IDs incorporate scientific configuration, input/truth identity, implementation hashes, recorded package versions, and external-tool identities. Changes to these can create a new run directory. Absolute resolved input paths also participate, so moving a checkout can change its run ID. Shared artifacts are reused according to their own signatures. Keep the per-run `manifest.json` when comparing runs.

`selection`, `output_directory`, and the config-file path itself are excluded from the scientific configuration portion of the run ID. Selection decisions have their own validation; changing the output root changes where artifacts are found. The root-level pointer is updated when a run is initialized. Earlier run directories remain available; the pointer is not an index of successful runs.

Changing `target_component_size` or `tree_seed` invalidates a managed backbone's
preparation signature and triggers reconstruction. Different targets can still
select the same component. Explicit prebuilt trees without a manifest are
retained; use a separate input directory to reconstruct a different backbone
while keeping the previous one.

## 10. Troubleshooting

| Symptom                                                          | What to check / next action                                                                                                                                                      |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `epilink-evaluate: command not found` or module import failure | Activate the environment used for installation and run`python -m pip install -e '.[test]'`. `python -m epilink_evaluation --help` uses that interpreter directly.            |
| `Executable not found`                                         | Run`check`, inspect reported paths, install the missing tool, or set its absolute path in `treecluster.executables`.                                                         |
| CSV parsing fails on a fresh checkout                            | Confirm raw paths and Git LFS downloads. An LFS pointer contains metadata rather than the input table; run`git lfs pull` after installing Git LFS.                             |
| Tree size did not change                                         | Inspect`n_cases` in the provenance; different targets can select the same component. Prebuilt trees without a manifest are retained; use a new input directory to reconstruct. |
| Report says`partial` / selection reports an incomplete sweep   | Inspect`seed_<seed>/clusters/status.json` and failed setting manifests. Fix the cause and rerun `clusters` or `develop`.                                                   |
| FastME or TreeTime fails                                         | Inspect`<output-root>/artifacts/trees/<id>/<tool>.stderr.log` and `.stdout.log`. Commands are saved in completed tree manifests; the exception names the failing log path.   |
| TreeCluster fails                                                | Inspect`<run>/development/seed_<seed>/clusters/<setting-id>/treecluster.stderr.log` and its manifest; evaluation uses the analogous evaluation path.                           |
| External command times out                                       | Inspect its stderr log and`treecluster.command_timeout_seconds`. Increasing the configured timeout changes the experiment signature and can create a new run.                  |
| No frozen operating settings                                     | Run`select` for the exact experiment/output root before `evaluate`.                                                                                                          |
| `No operating criterion is feasible`                           | Inspect the frozen decisions and development metrics; revise objectives, bounds, or grids using development evidence.                                                            |
| Criteria changed or held-out seeds already accessed              | Before access, rerun`select`. After access, assign fresh evaluation seeds before revised selection/evaluation.                                                                 |
| `current.json` missing when running `report`                 | Check config/output/smoke arguments. A computational stage must initialize that root first.                                                                                      |
| A rerun writes a different run ID                                | Compare manifests for config, input paths/hashes, implementation, package, and executable changes.                                                                               |
| Incubation parameter error                                       | The Gamma incubation shape is`1 / cv²`; EpiLink requires `latent_shape` to be smaller. Check the complete matched natural-history block.                                    |

## 11. Clear outputs with reset-outputs

To clear evaluation outputs and start fresh, use the `reset-outputs` command. This removes runs, artifacts, and the current.json pointer from the specified evaluation directories.

```bash
# Preview what would be deleted (recommended first)
epilink-evaluate reset-outputs --dry-run

# Clear all evaluation outputs (baseline, perturbation, boston)
epilink-evaluate reset-outputs

# Clear only baseline outputs (includes baseline_smoke)
epilink-evaluate reset-outputs --evaluations baseline

# Clear perturbation and boston outputs
epilink-evaluate reset-outputs --evaluations perturbation boston
```

The command removes:

- All run directories under `runs/`
- All artifact directories under `artifacts/`
- The `current.json` pointer file

Use `--dry-run` to inspect the selected paths. The command targets the standard
run roots listed above; `--output` does not select a custom cleanup location.
Shared `outputs/inputs/` directories, including the baseline backbone and Boston
tables, are retained.

## 12. Perturbation and Boston application

**Parameter sensitivity is available after baseline evaluation completes.** Use the separate perturbation entry point, which loads the baseline's frozen models/settings rather than fitting and selecting again:

```bash
python evaluation/02_synthetic_perturbation/run.py --smoke
python evaluation/02_synthetic_perturbation/run.py
```

The equivalent CLI is `epilink-evaluate perturbation --smoke`. Its default configuration is `evaluation/02_synthetic_perturbation/config.yaml`; the baseline `check` command expects a baseline configuration. Perturbation validates its reference and levels at startup. Use `--baseline-run` to pin a completed run directory instead of the default current-baseline pointer.

The workflow compares `matched` EpiLink inference with `baseline_fixed` inference on paired observations. Both logistic models and every operating setting stay fixed. Fresh unperturbed controls provide paired metric differences. Smoke uses 64 cases and the incubation-mean levels; full mode uses all six configured parameter families on the frozen backbone. Outputs are separate under `evaluation/02_synthetic_perturbation/outputs/perturbation[_smoke]/`.

See the [perturbation guide](evaluation/02_synthetic_perturbation/README.md) for all configuration fields, resumption, reference compatibility, and the validation checkpoint, and the [output schema](OUTPUTS.md#12-perturbation-study-outputs) for paired results. Retuning or retraining under changed parameters remains a separate adaptation analysis. Perturbation accepts `all` (the default) and `report` stages.

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

Run the separate descriptive threshold/resolution sensitivity sweep with:

```bash
python evaluation/03_boston_application/run.py --stage explore
```

| Boston stage      | Work performed                                                                                                     |
| ----------------- | ------------------------------------------------------------------------------------------------------------------ |
| `prepare`       | Write or reuse`outputs/inputs/` tables and provenance.                                                           |
| `trees`         | Build/reuse raw and dated trees and apply frozen TreeCluster settings. Graph scoring/clustering is not requested.  |
| `all` (default) | Score pairs, apply frozen graph settings, run enabled TreeCluster, and assess partitions.                          |
| `explore`       | Sweep the configured graph and TreeCluster grids under`exploration/`. This is requested separately from `all`. |
| `report`        | Render saved results for the run identified by the Boston root's`current.json`.                                  |

Boston reads metadata, Nextclade, TN93 distances, and the aligned FASTA under `data/raw/boston/`. Prepared tables are under `evaluation/03_boston_application/outputs/inputs/`; run results are under `evaluation/03_boston_application/outputs/boston/`. The TN93 table is censored at 0.0005/site; missing pairs remain unobserved, not zero. The exploration stage is descriptive and must not be treated as Boston truth-based operating-point selection.

The supplied exploration grid has 1,722 graph settings and 66 TreeCluster settings. Its scorer parameters and fitted models stay fixed while clustering settings vary. Inspect graph/tree `status.json` files within `exploration/` for completed counts and errors; a complete tree-only stage does not establish graph coverage.

Boston supports `--config`, `--output`, and `--baseline-run`, but not `--smoke`. YAML paths are relative to the Boston config; CLI overrides are relative to the working directory. The sibling baseline pointer is `../01_synthetic_baseline/outputs/baseline/current.json` in the supplied YAML. Repeat the same configuration and overrides to resume a run.

```bash
python evaluation/03_boston_application/run.py --stage report
python -m json.tool evaluation/03_boston_application/outputs/boston/current.json
```

See the [Boston guide](evaluation/03_boston_application/README.md) and [Boston output schema](OUTPUTS.md#10-boston-inputs-and-results) for interpretation.
