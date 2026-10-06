# Shared synthetic experiment

[`config.yaml`](config.yaml) owns the data design used by [diagnostics](../00_synthetic_diagnostics/README.md) and [baseline](../01_synthetic_baseline/README.md): `inputs`, `generation`, `simulation`, and `splits`. Both study configs reference it with `experiment_config: ../shared_synthetic/config.yaml`. Baseline derives matched EpiLink inference from `generation`; method grids and operating criteria remain in the study configs. Paths are relative to the YAML file that defines them.

## Execution and split roles

From the repository root:

```bash
# Small end-to-end workflow; both commands must use --smoke.
python evaluation/00_synthetic_diagnostics/run.py --smoke --stage all
python evaluation/01_synthetic_baseline/run.py --smoke --stage all

# Full development, followed by review and frozen evaluation.
python evaluation/00_synthetic_diagnostics/run.py --stage all
python evaluation/01_synthetic_baseline/run.py --stage develop
python evaluation/01_synthetic_baseline/run.py --stage select
python evaluation/01_synthetic_baseline/run.py --stage evaluate
```

Diagnostics prepares the backbone, full-tree truth, and **development** observations. Its `prepare` stage stops there; `all` (the default) also characterises the backbone and completes observation, graph, and enabled tree controls. The standalone `backbone` stage describes the full pinned tree without generating observations, independently of the sampling fraction. Baseline validates that completed evidence matches the exact shared development datasets, then generates separate **training** realizations for logistic fitting. Baseline `prepare` requires completed diagnostics and prepares training data while reusing development data. **Evaluation** observations are generated only after `evaluate` validates and releases frozen selection.

Seeds must be distinct across splits. Changing the data design requires diagnostics for the new experiment. To keep multiple designs, copy the shared config and point both study configs' `experiment_config` at that copy, using distinct output roots where appropriate. Select configs with `--config`; study `--output` overrides only that study's result root.

## Artifact and provenance contract

```text
outputs/
  inputs/                          managed SCoVMod tree and preparation provenance
  synthetic/                       shared experiment root
    current.json                   experiment_directory, fingerprint
    artifacts/
      backbones/<id>/              transmission_tree.gml, source.json, manifest.json
      truth/<id>/                  nodes.parquet, relationships.parquet, manifest.json
      observations/<id>/           cases.parquet, pairs.parquet, manifest.json
    experiments/<id>/
      experiment.json              resolved data design and pinned backbone path
      manifest.json                specification, producer, backbone/truth identities, checksums
      observations/seed_<seed>.json role, dataset ID, fingerprint
      diagnostics.json             completed diagnostics and exact development dataset IDs
    heldout_access/seed_<seed>.json experiment identity and frozen-selection fingerprint
```

Preparation copies the selected backbone into a content-addressed artifact; its `source.json` records original tree path/hash and available reconstruction provenance. Truth retains unsampled intermediates. Observation links appear as each split is prepared or released. `current.json` identifies the latest prepared experiment, not completed diagnostics or evaluation. Baseline's own `<run>/experiment.json` pins `experiment_directory` and `fingerprint`; follow that reference when joining scores to shared observations and truth, rather than the latest shared pointer.

Smoke uses up to 64 backbone cases and seeds 71001/72001/73001 for training/development/evaluation. It writes `outputs/synthetic_smoke/`, while both modes share the managed input directory. The immutable backbone/truth artifacts record the actual subset used. Smoke evaluation is repeatable pipeline validation, rather than held-out scientific evidence. Accesses are recorded under `validation_access/<selection-fingerprint>/` and changed comparison implementations may reuse the smoke observations. Full runs retain the strict `heldout_access/seed_<seed>.json` ledger.

`reset-outputs` clears selected study results, including diagnostics, but preserves this entire shared output area and its held-out access ledger. Clearing baseline runs therefore does not release previously accessed seeds for revised selection or training/development. For revised analyses after held-out access, use fresh evaluation seeds and rerun diagnostics for the updated design.

The supplied full evaluation seeds are now **63101–63103** for the refined comparison. Seeds 63001–63003 were accessed by the earlier baseline. Training and development seed roles are unchanged. Reporting saved results needs no new observations; changing grids or selection after evaluation requires fresh held-out observations. Smoke continues to use its separate validation namespace.

Retained results from earlier versions remain historical evidence. Current results require the workflow above. See [operations](../../OPERATIONS.md) and the [output reference](../../OUTPUTS.md#13-shared-experiments-and-diagnostics) for completion checks, diagnostic table schemas, and analysis joins.
