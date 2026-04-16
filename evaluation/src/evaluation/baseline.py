"""Baseline evaluation: run evaluation at baseline and compute a rich summary table.

Two public entry-points
-----------------------
evaluate_baseline(config)
    Runs the EpiLink pipeline at the matched-baseline scenario and returns raw
    scores plus basic per-model metrics (AP, best-F1, stability).

analyse_baseline(df, basic_metrics)
    Accepts the raw-scores dataframe and the basic metrics list returned by
    evaluate_baseline, then computes a richer summary table: stratified-bootstrap
    AP confidence intervals, PRG / AUPRG, isotonic-regression calibration (Brier),
    and F-beta at fixed recall levels.

main() saves both outputs to the configured results directory:
    baseline_scores.parquet      – raw prediction scores + labels
    baseline_summary.parquet     – full statistical summary table
"""

from __future__ import annotations

import json
import logging
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from config import (
    build_run_specs,
    configure_logging,
    get_pipeline_log_path,
    load_config,
    resolve_configured_output_path,
)
from evaluate import evaluate_scenario
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)
from specs import DEFAULT_SEED

LOGGER = logging.getLogger(__name__)

# ─── Analysis defaults ────────────────────────────────────────────────────────

N_BOOTSTRAP: int = 200
CI_ALPHA: float = 0.05
FIXED_RECALL_LEVELS: tuple[float, ...] = (0.7, 0.8, 0.9)
BETA: int = 2

# ─── Statistical helpers ──────────────────────────────────────────────────────


def stratified_bootstrap_auprc(
    y_true,
    scores,
    n_boot=N_BOOTSTRAP,
    alpha=CI_ALPHA,
    seed=DEFAULT_SEED,
) -> tuple[float, float]:
    """Stratified bootstrap CI on AP (positives and negatives resampled separately)."""
    LOGGER.debug("baseline: stratified_bootstrap_auprc n_boot=%d alpha=%s", n_boot, alpha)
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    # Generate all bootstrap index arrays in one vectorised call.
    b_pos = rng.choice(pos_idx, size=(n_boot, len(pos_idx)), replace=True)  # (n_boot, n_pos)
    b_neg = rng.choice(neg_idx, size=(n_boot, len(neg_idx)), replace=True)  # (n_boot, n_neg)
    b_all = np.concatenate([b_pos, b_neg], axis=1)  # (n_boot, n)

    boot_aps = np.array([average_precision_score(y_true[b], scores[b]) for b in b_all])
    return (
        float(np.percentile(boot_aps, 100 * alpha / 2)),
        float(np.percentile(boot_aps, 100 * (1 - alpha / 2))),
    )


def calibrate_scores(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Map raw model weights → calibrated probabilities via isotonic regression."""
    LOGGER.debug("baseline: calibrate_scores fitting isotonic regression")
    ir = IsotonicRegression(out_of_bounds="clip")
    order = np.argsort(scores)
    ir.fit(scores[order], y_true[order])
    return ir.predict(scores)


def compute_f_beta_at_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    recall_level: float,
    beta: int = BETA,
) -> tuple[float, float, float]:
    """Return (precision, recall, F-beta) at the operating point nearest to recall_level."""
    LOGGER.debug("baseline: compute_f_beta_at_recall recall_level=%s", recall_level)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    idx = np.argmin(np.abs(rec - recall_level))
    p, r = prec[idx], rec[idx]
    fb = 0.0 if (p + r == 0) else (1 + beta**2) * p * r / (beta**2 * p + r)
    return round(p, 4), round(r, 4), round(fb, 4)


def compute_prg_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Precision-Recall-Gain curve (Flach & Kull 2015).

    Returns (rec_gain, prec_gain, auprg).  The valid region is rec_gain >= 0
    and prec_gain >= 0.
    """
    LOGGER.debug("baseline: compute_prg_curve computing precision-recall-gain")
    pi = float(y_true.mean())
    prec, rec, _ = precision_recall_curve(y_true, scores)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prec_gain = np.where(prec > 0, (prec - pi) / ((1 - pi) * prec), 0.0)
        rec_gain = np.where(rec > 0, (rec - pi) / ((1 - pi) * rec), 0.0)
    mask = (rec_gain >= 0) & (prec_gain >= 0) & np.isfinite(prec_gain) & np.isfinite(rec_gain)
    rg, pg = rec_gain[mask], prec_gain[mask]
    if len(rg) > 1:
        order = np.argsort(rg)
        auprg = float(np.trapezoid(pg[order], rg[order]))
    else:
        auprg = 0.0
    return rg, pg, auprg


# ─── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_baseline(config: dict[str, Any]) -> dict[str, Any]:
    """Run the EpiLink pipeline at the matched-baseline scenario.

    Returns
    -------
    dict with keys:
        ``metrics`` – list of per-model basic metric dicts
                      (model, n_pairs, prevalence, ap, best_f1, mean/std_stability)
        ``scores``  – DataFrame with IsRelated + one score column per model
    """
    runs = build_run_specs(config)
    evaluate_kwargs = deepcopy(config["execution"].get("evaluate_kwargs", {}))

    optimal_thresholds: dict[str, float] = {}
    thresholds_path = resolve_configured_output_path(
        config, "outputs.sparsification.optimal_thresholds_path"
    )
    if thresholds_path.exists():
        optimal_thresholds = json.loads(thresholds_path.read_text())

    evaluate_kwargs["sparsification"] = optimal_thresholds
    evaluate_kwargs["return_scores"] = True
    evaluate_kwargs["return_classifier"] = False

    def _is_baseline(run: Any) -> bool:
        return run.condition == "matched" and run.scenario_name == "baseline"

    baseline_runs = [run for run in runs if _is_baseline(run)]
    if len(baseline_runs) == 0:
        raise ValueError("No baseline runs found in the configuration.")
    if len(baseline_runs) > 1:
        raise ValueError("Multiple baseline runs found in the configuration.")
    baseline = baseline_runs[0]

    results, _, scores = evaluate_scenario(
        tree_path=baseline.tree_path,
        scenario_name=baseline.scenario_name,
        generation_parameters=baseline.generation_parameters,
        inference_parameters=baseline.inference_parameters,
        **evaluate_kwargs,
    )

    metrics = [
        {
            "model": key,
            "n_pairs": results.n_pairs,
            "prevalence": results.prevalence,
            "ap": res.ap,
            "best_f1": res.best_f1,
            "mean_stability": res.mean_stability,
            "std_stability": res.std_stability,
        }
        for key, res in results.models.items()
    ]
    return {"metrics": metrics, "scores": scores}


def analyse_baseline(
    df: pd.DataFrame,
    basic_metrics: list[dict[str, Any]],
    *,
    model_keys: tuple[str, ...] = None,
    n_bootstrap: int = N_BOOTSTRAP,
    ci_alpha: float = CI_ALPHA,
    beta: int = BETA,
    fixed_recall_levels: tuple[float, ...] = FIXED_RECALL_LEVELS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Compute a rich per-model summary table from raw baseline scores.

    Parameters
    ----------
    df:
        DataFrame with an ``IsRelated`` column (true labels, 0/1) and one
        column per model containing raw prediction scores (not probabilities).
    basic_metrics:
        List of per-model dicts as returned by ``evaluate_baseline``; used
        to look up ``best_f1`` which is computed during the evaluation step.
    model_keys:
        Explicit model order. Defaults to the insertion order of basic_metrics.
    n_bootstrap:
        Number of bootstrap samples for AP CI estimation.
    ci_alpha:
        Significance level for AP confidence intervals (e.g. 0.05 for 95% CI).
    beta:
        Beta parameter for F-beta score at fixed recall levels.
    fixed_recall_levels:
        Tuple of recall levels at which to compute precision and F-beta scores.
    seed:
        Random seed for reproducibility of bootstrap sampling and isotonic regression.

    Returns
    -------
    DataFrame sorted descending by AP with columns:
        model, ap, best_f1, ci_lo, ci_hi, relative_ap, auprg, brier,
        prevalence, p_at_r{N}, f{beta}_at_r{N}  (one column per recall level)
    """
    y = df["IsRelated"].values
    prevalence = float(y.mean())
    basic = {m["model"]: m for m in basic_metrics}

    if model_keys is None:
        model_keys = tuple(basic.keys())

    rows = []
    for model in model_keys:
        LOGGER.info("baseline: computing summary for model %s", model)
        scores = df[model].values
        ci_lo, ci_hi = stratified_bootstrap_auprc(
            y,
            scores,
            n_boot=n_bootstrap,
            alpha=ci_alpha,
            seed=seed,
        )
        _, _, auprg = compute_prg_curve(y, scores)
        cal = calibrate_scores(y, scores)
        brier = float(brier_score_loss(y, cal))

        precision_at = {
            f"p_at_r{int(rl * 100)}": compute_f_beta_at_recall(y, scores, rl, beta)[0]
            for rl in fixed_recall_levels
        }
        fbeta_at = {
            f"f{beta}_at_r{int(rl * 100)}": compute_f_beta_at_recall(y, scores, rl, beta)[2]
            for rl in fixed_recall_levels
        }

        rows.append(
            {
                "model": model,
                "ap": round(basic[model]["ap"], 4),
                "best_f1": round(basic[model]["best_f1"], 4),
                "mean_stability": round(basic[model]["mean_stability"], 4),
                "std_stability": round(basic[model]["best_f1"], 4),
                "ci_lo": round(ci_lo, 4),
                "ci_hi": round(ci_hi, 4),
                "relative_ap": round(basic[model]["ap"] / prevalence, 3),
                "auprg": round(auprg, 4),
                "brier": round(brier, 4),
                "prevalence": round(prevalence, 6),
                **precision_at,
                **fbeta_at,
            }
        )

    return pd.DataFrame(rows).sort_values("ap", ascending=False).reset_index(drop=True)


# ─── Entry point ──────────────────────────────────────────────────────────────


def main(config_path: str | Path = "config.yaml") -> None:
    """Run baseline evaluation and write scores and rich summary parquet outputs."""
    config = load_config(config_path)
    configure_logging(log_file=get_pipeline_log_path(config))
    LOGGER.info("baseline: starting")
    out_dir = resolve_configured_output_path(config, "outputs.synthetic.directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_baseline(config)
    metrics = results["metrics"]
    scores_df = results["scores"]
    LOGGER.info(
        "baseline: evaluated %d models on %d pairs",
        len(metrics),
        metrics[0]["n_pairs"] if metrics else 0,
    )

    summary = analyse_baseline(scores_df, metrics)

    scores_df.to_parquet(out_dir / "baseline_scores.parquet", index=False)
    summary.to_parquet(out_dir / "baseline_summary.parquet", index=False)
    LOGGER.info("baseline: written outputs to %s", out_dir)


if __name__ == "__main__":
    main()
