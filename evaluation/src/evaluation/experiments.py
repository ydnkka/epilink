from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Dict, Optional

import pandas as pd

try:
    from .config import build_run_specs
    from .evaluate import ScenarioResult, evaluate_scenario
except ImportError:
    from config import build_run_specs
    from evaluate import ScenarioResult, evaluate_scenario


def _evaluate_run(
        run: Any,
        evaluate_kwargs: Dict[str, Any],
        baseline_performance: Optional[Dict[str, Dict[str, float]]],
        logistic_classifier: Optional[Dict[str, Any]],
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


def _make_row(run: Any, scenario_result: ScenarioResult, model_key: str, model_result: Any) -> Dict[str, Any]:
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
        "generation_incubation_shape": run.generation_parameters["incubation_shape"],
        "generation_incubation_scale": run.generation_parameters["incubation_scale"],
        "generation_testing_delay_shape": run.generation_parameters["testing_delay_shape"],
        "generation_testing_delay_scale": run.generation_parameters["testing_delay_scale"],
        "generation_substitution_rate": run.generation_parameters["substitution_rate"],
        "generation_relaxation": run.generation_parameters["relaxation"],
        "inference_incubation_shape": run.inference_parameters["incubation_shape"],
        "inference_incubation_scale": run.inference_parameters["incubation_scale"],
        "inference_testing_delay_shape": run.inference_parameters["testing_delay_shape"],
        "inference_testing_delay_scale": run.inference_parameters["testing_delay_scale"],
        "inference_substitution_rate": run.inference_parameters["substitution_rate"],
        "inference_relaxation": run.inference_parameters["relaxation"],
    }


def run_experiment(config: Dict[str, Any]) -> pd.DataFrame:
    """Run all configured scenarios and return a tidy results table.

    The baseline run (matched condition + loss_reference scenario) executes first so that its
    classifier and performance metrics are available to all subsequent runs. All other runs are
    then dispatched in parallel via ProcessPoolExecutor.

    Set ``execution.max_workers`` in the config to cap the process-pool size (default: one
    worker per CPU core). On platforms that require a ``if __name__ == '__main__'`` guard
    (Windows), ensure callers use it.
    """
    runs = build_run_specs(config)
    evaluate_kwargs = deepcopy(config["execution"].get("evaluate_kwargs", {}))
    max_workers = config["execution"].get("max_workers", None)

    baseline_reference = config["design"]["loss_reference"]["scenario"]
    classifier_cache: Dict[str, Dict[str, Any]] = {}
    baseline_performance_cache: Dict[str, Dict[str, float]] = {}

    results_rows = []

    def _is_baseline(r: Any) -> bool:
        return r.condition == "matched" and r.scenario_name == baseline_reference

    baseline_runs = [r for r in runs if _is_baseline(r)]
    other_runs = [r for r in runs if not _is_baseline(r)]

    # Phase 1: baseline runs are sequential — their outputs seed the parallel phase.
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
            classifier_cache["baseline"] = classifiers

        baseline_performance_cache = {
            key: {"ap": model_result.ap, "best_f1": model_result.best_f1}
            for key, model_result in scenario_result.models.items()
        }

        for model_key, model_result in scenario_result.models.items():
            results_rows.append(_make_row(run, scenario_result, model_key, model_result))

    # Phase 2: all remaining runs in parallel.
    if other_runs:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_run,
                    run,
                    evaluate_kwargs,
                    baseline_performance_cache or None,
                    classifier_cache.get("baseline") if run.condition == "generation_varied_inference_fixed" else None,
                ): run
                for run in other_runs
            }

            for future in as_completed(futures):
                run, scenario_result = future.result()
                for model_key, model_result in scenario_result.models.items():
                    results_rows.append(_make_row(run, scenario_result, model_key, model_result))

    return pd.DataFrame(results_rows)
