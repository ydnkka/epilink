from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import logging
from typing import Any
from pathlib import Path
import json

import pandas as pd

try:
    from .config import build_run_specs, configure_logging, load_config, resolve_configured_output_path
    from .evaluate import ScenarioResult, evaluate_scenario
    from .specs import BASELINE_SCENARIO_NAME, parameter_columns
except ImportError:
    from config import build_run_specs, configure_logging, load_config, resolve_configured_output_path
    from evaluate import ScenarioResult, evaluate_scenario
    from specs import BASELINE_SCENARIO_NAME, parameter_columns



LOGGER = logging.getLogger(__name__)


def _evaluate_run(
        run: Any,
        evaluate_kwargs: dict[str, Any],
        baseline_performance: dict[str, dict[str, float]] | None,
        logistic_classifier: dict[str, Any] | None,
) -> tuple:
    """Worker: evaluate a single run. Must be module-level for ProcessPoolExecutor pickling."""
    scenario_result, classifiers = evaluate_scenario(
        tree_path=run.tree_path,
        scenario_name=run.scenario_name,
        generation_parameters=run.generation_parameters,
        inference_parameters=run.inference_parameters,
        logistic_classifier=logistic_classifier,
        baseline_performance=baseline_performance,
        **evaluate_kwargs,
    )
    return run, scenario_result


def _make_row(
        run: Any,
        scenario_result: ScenarioResult,
        model_key: str,
        model_result: Any,
) -> dict[str, Any]:
    return {
        "condition": run.condition,
        "scenario": run.scenario_name,
        "perturbed_parameter": run.perturbed_parameter,
        "perturbation_multiplier": run.perturbation_multiplier,
        "model": model_key,
        "n_pairs": scenario_result.n_pairs,
        "prevalence": scenario_result.prevalence,
        "ap": model_result.ap,
        "ap_loss": model_result.ap_loss,
        "best_f1": model_result.best_f1,
        "f1_loss": model_result.f1_loss,
        "mean_stability": model_result.mean_stability,
        "std_stability": model_result.std_stability,
        **parameter_columns(run.generation_parameters, prefix="generation"),
        **parameter_columns(run.inference_parameters, prefix="inference"),
    }


def run_experiment(config: dict[str, Any]) -> pd.DataFrame:
    """Run all configured scenarios and return a tidy results table.

    The loss-reference run executes first so that its classifier and performance metrics are
    available to all subsequent runs. All other runs are then dispatched in parallel via
    ProcessPoolExecutor.

    Set ``execution.max_workers`` in the config to cap the process-pool size (default: one
    worker per CPU core). On platforms that require a ``if __name__ == '__main__'`` guard
    (Windows), ensure callers use it.
    """
    runs = build_run_specs(config)
    evaluate_kwargs = deepcopy(config["execution"].get("evaluate_kwargs", {}))
    max_workers = config["execution"].get("max_workers", None)
    LOGGER.info("experiments: %d runs configured", len(runs))

    loss_reference = config["design"]["loss_reference"]
    loss_reference_condition = loss_reference["condition"]
    loss_reference_scenario = loss_reference["scenario"]
    classifier_cache: dict[str, dict[str, Any]] = {}
    baseline_performance_cache: dict[str, dict[str, float]] = {}

    optimal_thresholds: dict[str, float] = {}
    thresholds_path = resolve_configured_output_path(config, "outputs.sparsification.optimal_thresholds_path")
    if thresholds_path.exists():
        optimal_thresholds = json.loads(thresholds_path.read_text())

    evaluate_kwargs["sparsification"] = optimal_thresholds

    results_rows = []

    def _is_loss_reference(run: Any) -> bool:
        return run.condition == loss_reference_condition and run.scenario_name == loss_reference_scenario

    baseline_runs = [run for run in runs if _is_loss_reference(run)]
    other_runs = [run for run in runs if not _is_loss_reference(run)]

    # Phase 1: the loss-reference runs are sequential — their outputs seed the parallel phase.
    LOGGER.info("experiments: running baseline reference")
    for run in baseline_runs:
        scenario_result, classifiers = evaluate_scenario(
            tree_path=run.tree_path,
            scenario_name=run.scenario_name,
            generation_parameters=run.generation_parameters,
            inference_parameters=run.inference_parameters,
            logistic_classifier=None,
            baseline_performance=None,
            **evaluate_kwargs,
        )

        if classifiers is not None:
            classifier_cache["loss_reference"] = classifiers

        baseline_performance_cache = {
            key: {"ap": model_result.ap, "best_f1": model_result.best_f1}
            for key, model_result in scenario_result.models.items()
        }

        for model_key, model_result in scenario_result.models.items():
            results_rows.append(_make_row(run, scenario_result, model_key, model_result))

    # Phase 2: all remaining runs in parallel.
    if other_runs:
        LOGGER.info("experiments: running remaining scenarios")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_run,
                    run,
                    evaluate_kwargs,
                    baseline_performance_cache or None,
                    (
                        classifier_cache.get("loss_reference")
                        if run.logit_training_source == BASELINE_SCENARIO_NAME
                        else None
                    ),
                ): run
                for run in other_runs
            }

            for future in as_completed(futures):
                run, scenario_result = future.result()
                for model_key, model_result in scenario_result.models.items():
                    results_rows.append(_make_row(run, scenario_result, model_key, model_result))

    return pd.DataFrame(results_rows)

def main(config_path: str | Path = "config.yaml") -> None:
    configure_logging()
    LOGGER.info("experiments: starting")
    config = load_config(config_path)
    out_dir = resolve_configured_output_path(config, "outputs.synthetic.directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = run_experiment(config)
    results.to_parquet(out_dir / "results.parquet", index=False)
    LOGGER.info("experiments: done")

if __name__ == "__main__":
    main()
