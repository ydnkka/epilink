# Parameter perturbation with frozen baseline settings

## Purpose and study sequence

**Main question:** How sensitive are baseline-selected methods to changes in
natural-history and mutation parameters, and how does performance differ when
EpiLink uses matching versus baseline-fixed parameter values?

This is the **biological-parameter sensitivity and model-mismatch study**. It
starts from the completed [synthetic baseline](../01_synthetic_baseline/README.md),
retains its transmission backbone and known relationship truth, and generates
fresh observations with one generation parameter changed at a time.

The study has three objectives:

1. **Measure sensitivity to changed observations:** compare each perturbed
   scenario with an unperturbed control generated using the same fresh seeds.
2. **Examine EpiLink parameter mismatch:** score the same perturbed observations
   using both `matched` inference (the scenario's parameter values) and
   `baseline_fixed` inference (the baseline values).
3. **Measure robustness of the selected operating points:** replay the baseline's
   pairwise thresholds, graph settings, and TreeCluster rules across scenarios.
   Logistic models remain baseline-fitted in both inference modes.

**Evidence produced:** absolute pairwise and clustering performance against known
truth, plus paired changes in AP, recovery, contamination, and cluster structure.
The matched/fixed comparison shows the effect of updating EpiLink's parameter
inputs while keeping operating points fixed. Threshold reselection and classifier
retraining would answer a separate adaptation question.

**Role in the study sequence:** the baseline establishes performance and selects
settings; this study assesses their sensitivity under controlled changes. The
[Boston application](../03_boston_application/README.md) independently applies the
baseline reference to empirical observations. Boston's clustering-parameter
exploration varies thresholds/resolutions on fixed real data; this study varies
the observation-generation parameters on synthetic data.

All commands run from the repository root in the installed environment.

## Run a smoke study

After the reference baseline's `--stage evaluate` has completed:

```bash
conda activate epilik_evaluation
python evaluation/02_synthetic_perturbation/run.py --smoke
```

The equivalent installed CLI command is:

```bash
epilink-evaluate perturbation --config evaluation/02_synthetic_perturbation/config.yaml --smoke
```

The default config resolves
`evaluation/01_synthetic_baseline/outputs/baseline/current.json` once at startup. To pin a
particular completed baseline, use its run directory:

```bash
python evaluation/02_synthetic_perturbation/run.py --smoke --baseline-run "evaluation/01_synthetic_baseline/outputs/baseline/runs/<completed-run-id>"
```

Replace `<completed-run-id>` with the ID of your completed baseline run.

`--baseline-run` also accepts a baseline `current.json`. The resolved reference,
its frozen decisions, and its model/truth identities are saved with the new study.
The input must be a completed baseline with valid selection, training, truth,
and held-out artifacts. The source output directory is read-only to this workflow.

## Full study and reporting

```bash
# Use the entire frozen backbone, all configured perturbations, and study seeds.
python evaluation/02_synthetic_perturbation/run.py

# Re-render saved smoke tables; no simulations or baseline reload.
python evaluation/02_synthetic_perturbation/run.py --smoke --stage report
```

Only `--stage all` (the default) and `--stage report` apply here. There is no
perturbation fitting or selection stage. Repeat a run command to resume validated
artifacts. Use `--config` for a different complete study YAML, or `--output` to
override its output root. Repeat the same options on subsequent commands;
`--smoke` appends `_smoke` to the resolved output root, including overrides.

## Scientific comparison

For every scenario, observation seed, and mode, replay the selected pairwise
thresholds and cluster definitions exactly as stored in the reference:

| Mode             | Generation parameters       | EpiLink inference parameters | Logistic models        |
| ---------------- | --------------------------- | ---------------------------- | ---------------------- |
| `matched`        | Scenario's perturbed values | Same perturbed values        | Baseline-fitted, fixed |
| `baseline_fixed` | Same scenario observations  | Baseline values              | Baseline-fitted, fixed |

Graph thresholds, Leiden objectives/resolutions/weights/restarts/seeds, and
TreeCluster methods/cutoffs stay fixed in both modes. EpiLink Monte Carlo settings
and external-tool settings also come from the baseline. The modes affect EpiLink
inference only: GD, LOGIT, and phylogenetic comparisons on the same observations
therefore provide identical comparison results across modes.

Every study includes an **unperturbed baseline control** on the same fresh seeds.
For each metric, compute:

```text
delta = perturbed metric - unperturbed-control metric
```

Controls are matched by inference mode, observation seed, pipeline, setting,
and criterion; ranking controls are matched by mode, seed, and scorer. These
deltas compare newly simulated paired realizations. Earlier baseline evaluation
results supplied the reference context; their seeds are not reused as controls.

Negative F1/AP deltas mean reduced recovery; positive contamination deltas mean
more distant pairs. Undefined precision for an empty selection stays undefined.
The same seed provides a paired simulation design but does not guarantee identical
latent random draws after changing distributions. All realizations share one
backbone; SD/range describe conditional observation variation.

Retraining logistic models or retuning thresholds under a changed parameter is a
separate adaptation study and is not performed here. Empirical transfer to real
Boston data is handled by the Boston application, which uses the same baseline
reference but has no synthetic transmission truth.

## Configuration

The default [`config.yaml`](config.yaml) contains:

| Field              | Meaning                                                                                                       |
| ------------------ | ------------------------------------------------------------------------------------------------------------- |
| `schema_version`   | `1`.                                                                                                          |
| `name`             | Study label, included in its signature.                                                                       |
| `baseline_run`     | Evaluated baseline run directory or current-run pointer, relative to this YAML.                               |
| `output_directory` | Study output root, relative to this YAML. Must be separate from baseline outputs.                             |
| `seeds`            | Distinct fresh observation seeds. Overlap with any baseline split is rejected. Defaults: 81001, 81002, 81003. |
| `modes`            | Nonempty subset of `matched`, `baseline_fixed`; both are supplied.                                            |
| `perturbations`    | One-at-a-time parameter definitions; use either `multipliers` or absolute `values` for each parameter.        |
| `smoke.cases`      | Ancestor-preserving topological prefix size, up to the reference size; default 64.                            |
| `smoke.seed`       | Separate smoke observation seed, default 91001.                                                               |
| `smoke.parameters` | Subset of configured parameters exercised in smoke mode; default `incubation.mean`.                           |

CLI path overrides are resolved relative to the working directory. For a second
study, copy this YAML alongside the default file and pass it with `--config`.
The baseline's own YAML is not edited or reloaded to construct the study.

Default full perturbations are:

| Parameter            | Levels                   | Units                                     |
| -------------------- | ------------------------ | ----------------------------------------- |
| `incubation.mean`    | 0.75× and 1.25× baseline | Days                                      |
| `incubation.cv`      | 0.75× and 1.25× baseline | Dimensionless CV                          |
| `testing_delay.mean` | 0.75× and 1.25× baseline | Days                                      |
| `testing_delay.cv`   | 0.75× and 1.25× baseline | Dimensionless CV                          |
| `substitution_rate`  | 0.75× and 1.25× baseline | Substitutions/site/year                   |
| `relaxation`         | Absolute 0.0 and 0.66    | Lognormal rate SD; zero is a strict clock |

These produce 12 perturbed scenarios plus one control, crossed with both modes
and three seeds. Exactly one generation parameter changes in each scenario.
Multipliers must be positive; resolved values must be positive except that
relaxation may be zero. The latent-stage shape must remain below `1 / incubation.cv²`.
Duplicate parameters/levels and levels equal to baseline are rejected. Inspect
absolute resolved levels in `scenarios.json`, especially for a custom baseline.

Smoke mode uses the control plus both incubation-mean levels, both modes, and one
seed. It preserves the reference's fitted models, operating settings, Monte Carlo
draw count, and algorithm settings while reducing the backbone and scenario set.
It is pipeline validation; full-backbone sensitivity conclusions require the full
study. A smoke baseline can be used as a reference for smoke validation only.

## Outputs and interpretation

Default roots:

```text
evaluation/02_synthetic_perturbation/outputs/perturbation/
evaluation/02_synthetic_perturbation/outputs/perturbation_smoke/
```

Each has `current.json`, shared `artifacts/`, and fingerprinted `runs/<id>/`.
Print the pointer to locate the report:

```bash
python -m json.tool evaluation/02_synthetic_perturbation/outputs/perturbation_smoke/current.json
```

Inside the run, inspect these files in order:

1. `coverage.csv`: completion of each scenario/mode across the configured seeds.
2. `reference.json`, `selection.json`, `scenarios.json`: source identity, unchanged
   operating decisions, and resolved parameter values.
3. `report.html` / `report.md`: paired AP and fixed-setting changes, coverage, and
   F1 change heatmaps.
4. `results.csv` and `rankings.csv`: seed-specific operating metrics and rankings,
   including unperturbed controls.
5. `results_deltas.csv` and `rankings_deltas.csv`: individual paired differences.
6. `results_delta_summary.csv` and `rankings_delta_summary.csv`: paired mean, sample
   SD, range, and nonmissing counts. The corresponding `*_summary.csv` files
   summarize absolute performance, including controls.

`scenarios/<scenario>/<mode>/evaluation/seed_<seed>/` retains the baseline-style
pairwise tables, cluster memberships, setting-level metrics, status, and tool
logs. The field reference is in [OUTPUTS.md](../../OUTPUTS.md#12-perturbation-study-outputs).

Missing controls remain explicit: `control_available` is false and paired deltas
are missing. A present control can itself have undefined metrics, so inspect
`delta_<metric>_count` as well as `n_controls`. A partial comparator or failed
replay gives a partial study, reports its coverage, and returns a nonzero exit
code. It never silently substitutes a different operating setting.

## Reference integrity and resumption

The runner verifies frozen selection/evidence, training identity, truth checksums,
and every requested baseline held-out artifact. Existing scientific producer
code, EpiLink code, recorded dependencies, and executable hashes must agree with
the reference. New workflow, CLI, and reporting code can consume a previous
baseline without forcing its development sweep to run again.

The frozen topology is reconstructed from the validated truth artifact and saved
under the perturbation root's `artifacts/backbones/<id>/transmission_tree.gml`.
Changing the live baseline `outputs/inputs/` tree cannot replace that reference
topology. Frozen model bytes are copied into the study's model
artifacts; training dataset identities still refer to the source baseline.

Changed perturbation configurations or implementations produce a new study run
ID. Unchanged runs reuse observations, scores, trees, and setting checkpoints.
Modes share the same observation artifacts within each scenario/seed. Fix failed
tool invocations using the saved logs, then repeat the command. If the source
baseline's scientific implementation or dependencies changed, restore that
environment or deliberately create a new baseline reference.

## Validation checkpoint — 2026-09-30

This is a dated record of the implementation at that checkpoint. Commands and
links use the current directory names; the recorded counts are not a new test run.

- **47 tests passed**, including frozen-model replay, independent paired-delta
  arithmetic, missing controls, source-integrity checks, and cache reuse.
- Reference baseline: `91a6a1370408b0849567`, with a 1,002-case backbone.
- Smoke run: `455c87d2dc1d53e6c7b8`, with 64 cases and seed 91001.
- All six scenario/mode combinations completed all 34 selected settings:
  204 operating result rows, including 156 clustering evaluations.
- All eight scorers and both raw/dated TreeCluster comparisons completed.

[Historical smoke report (requires retained local outputs)](outputs/perturbation_smoke/runs/455c87d2dc1d53e6c7b8/report.html).
Generated outputs are local ignored artifacts; the full perturbation study has
not been run as part of this checkpoint.
