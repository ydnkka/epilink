"""Stage 4: Validation of scorer assumptions and independent benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.config import resolve_generation_baseline_parameters
from synthetic_exploration.observations import DISTANCES, feature_cells
from synthetic_exploration.scores import (
    average_precision,
    ranking_table,
    thresholds_for_selections,
)

from .common import log, save_table, write_json


def binned_joint(time, genetic):
    """Unit-width bins centred on integers, including negative time bins."""
    return (
        pd.DataFrame(
            {
                "time_days": np.rint(time).astype(int),
                "genetic_distance": np.rint(genetic).astype(int),
            }
        )
        .value_counts()
        .rename("n")
        .reset_index()
    )


def joint_total_variation(left, right):
    """Total variation distance between two joint distributions."""
    joined = left.merge(
        right,
        on=["time_days", "genetic_distance"],
        how="outer",
        suffixes=("_left", "_right"),
    ).fillna(0)
    return float(
        0.5
        * np.abs(
            joined.n_left / joined.n_left.sum() - joined.n_right / joined.n_right.sum()
        ).sum()
    )


def fit_feature_benchmarks(train_cells, test_cells, strength):
    """Train on another realization; aggregate identical inputs without changing weights."""
    columns = ["time_days", "genetic_distance"]
    train_x = train_cells[columns].to_numpy(float)
    test_x = test_cells[columns].to_numpy(float)
    prevalence = float(train_cells.n_target.sum() / train_cells.n_pairs.sum())
    if not 0 < prevalence < 1:
        raise ValueError(
            "Benchmark training requires both target and non-target pairs."
        )

    x = np.concatenate([train_x, train_x])
    y = np.r_[np.ones(len(train_x)), np.zeros(len(train_x))]
    weights = np.r_[train_cells.n_target.to_numpy(), train_cells.n_other.to_numpy()]
    valid = weights > 0

    scaler = StandardScaler().fit(x[valid], sample_weight=weights[valid])
    classifier = LogisticRegression(max_iter=1000).fit(
        scaler.transform(x[valid]), y[valid], sample_weight=weights[valid]
    )

    logistic = classifier.predict_proba(scaler.transform(test_x))[:, 1]
    lookup = (
        test_cells[columns]
        .merge(train_cells[columns + ["n_target", "n_pairs"]], on=columns, how="left")
        .fillna(0)
    )
    probability = (lookup.n_target + strength * prevalence) / (
        lookup.n_pairs + strength
    )

    return (
        logistic,
        probability.to_numpy(),
        {
            "training_prevalence": prevalence,
            "test_pair_fraction_in_unseen_cells": float(
                test_cells.loc[lookup.n_pairs == 0, "n_pairs"].sum()
                / test_cells.n_pairs.sum()
            ),
            "logistic_coefficients_standardised": classifier.coef_[0].tolist(),
            "logistic_intercept": float(classifier.intercept_[0]),
            "feature_mean": scaler.mean_.tolist(),
            "feature_scale": scaler.scale_.tolist(),
            "lookup_prior_strength": strength,
        },
    )


def investigate_validation(
    pairs,
    cases,
    scores,
    models,
    run,
    source,
    output,
    directory,
    settings,
    seed,
    config_directory,
):
    """Stage 4: Historical reproduction, scorer assumptions, and independent-observation benchmarks."""
    log(
        "4/4: historical reproduction, scorer assumptions, and independent-observation benchmarks"
    )
    config = settings["validation"]

    checks = {
        "pair_count": len(pairs),
        "sampled_case_count": int(cases.sampled.sum()),
        "transmission_components": int(cases.root_index.nunique()),
        "pairwise_self_pairs": int(np.sum(pairs.CaseID1 == pairs.CaseID2)),
        "truth_label_agreement": bool(
            np.all(pairs.IsRelated == (pairs.AD.fillna(0) == 1))
        ),
        "historical_comparison": "not_applicable",
    }

    if (
        config["compare_historical_baseline"]
        and run.scenario_name == "baseline"
        and seed == source["rng_seed"]
    ):
        historical_path = (config_directory / config["historical_scores"]).resolve()
        old_labels = pd.read_parquet(
            historical_path, columns=["IsRelated"]
        ).IsRelated.to_numpy(bool)
        np.testing.assert_array_equal(old_labels, pairs.IsRelated.to_numpy(bool))
        del old_labels

        errors = {}
        for model in settings["models"]:
            old = pd.read_parquet(historical_path, columns=[model])[model].to_numpy()
            current = scores[model].to_numpy()
            np.testing.assert_allclose(current, old, rtol=1e-12, atol=1e-14)
            errors[model] = float(np.max(np.abs(current - old)))
        checks["historical_comparison"] = (
            "all_labels_and_selected_epilink_scores_reproduced"
        )
        checks["maximum_absolute_score_difference"] = errors

    write_json(directory / "integrity_checks.json", checks)

    # Scorer generator checks
    summaries, draw_tables, joint_tables = [], [], []
    for data_process, genetic_column in DISTANCES.items():
        for inference_process, model in models.items():
            for relationship, scenario in [
                ("direct", "ad(0)"),
                ("shared_infector", "ca(0,0)"),
            ]:
                observed = pairs.loc[pairs.relationship == relationship]
                if not len(observed):
                    continue
                draws = model.draws_by_scenario[scenario]
                t, g = (
                    np.asarray(draws["time_draws"]),
                    np.asarray(draws["genetic_draws"]),
                )
                observed_time = observed.SamplingDateDistanceDays.to_numpy()
                observed_genetic = observed[genetic_column].to_numpy()

                base = {
                    "data_process": data_process,
                    "inference_process": inference_process,
                    "relationship": relationship,
                    "scenario": scenario,
                }
                observed_joint = binned_joint(observed_time, observed_genetic)
                raw_joint, folded_joint = binned_joint(t, g), binned_joint(np.abs(t), g)

                summaries.append(
                    {
                        **base,
                        "n_observed": len(observed),
                        "n_draws": len(t),
                        "draw_negative_time_fraction": float(np.mean(t < 0)),
                        "observed_mean_time": float(observed_time.mean()),
                        "raw_draw_mean_time": float(t.mean()),
                        "absolute_draw_mean_time": float(np.abs(t).mean()),
                        "observed_mean_genetic": float(observed_genetic.mean()),
                        "draw_mean_genetic": float(g.mean()),
                        "observed_zero_genetic_fraction": float(
                            np.mean(observed_genetic == 0)
                        ),
                        "draw_zero_genetic_fraction": float(np.mean(g == 0)),
                        "wasserstein_time_raw": wasserstein_distance(observed_time, t),
                        "wasserstein_time_absolute_rounded": wasserstein_distance(
                            observed_time, np.rint(np.abs(t))
                        ),
                        "wasserstein_genetic_raw": wasserstein_distance(
                            observed_genetic, g
                        ),
                        "joint_tv_raw_rounded": joint_total_variation(
                            observed_joint, raw_joint
                        ),
                        "joint_tv_absolute_time_rounded": joint_total_variation(
                            observed_joint, folded_joint
                        ),
                    }
                )
                joint_tables.extend(
                    frame.assign(source=label, **base)
                    for label, frame in [
                        ("observed_target_pairs", observed_joint),
                        ("raw_scorer_draws_rounded", raw_joint),
                        ("absolute_time_draws_rounded_diagnostic", folded_joint),
                    ]
                )

    for inference_process, model in models.items():
        for scenario, draws in model.draws_by_scenario.items():
            draw_tables.append(
                pd.DataFrame(
                    {
                        "inference_process": inference_process,
                        "scenario": scenario,
                        "time_draw": draws["time_draws"],
                        "genetic_draw": draws["genetic_draws"],
                        "branch_draw": draws["branch_draws"],
                    }
                )
            )

    save_table(directory, "scorer_generator_checks", summaries)
    save_table(
        directory,
        "target_joint_distributions",
        pd.concat(joint_tables, ignore_index=True),
    )
    save_table(directory, "scorer_draws", pd.concat(draw_tables, ignore_index=True))

    # Independent-observation benchmarks
    training_seed = int(settings["training_seed"])
    if training_seed == seed:
        raise ValueError(
            "Training and evaluation must use distinct observation-generation seeds."
        )

    training_parameters = (
        resolve_generation_baseline_parameters(source)
        if run.logit_training_source == "baseline"
        else run.generation_parameters
    )

    from synthetic_baseline.run import prepare_and_load_dataset

    training_pairs, _ = prepare_and_load_dataset(
        run.tree_path, training_parameters, training_seed, output
    )

    benchmark_rows, operating_rows, benchmark_metadata = [], [], {}
    for process, column in DISTANCES.items():
        train_cells, test_cells = (
            feature_cells(training_pairs, column),
            feature_cells(pairs, column),
        )
        save_table(directory, f"{process}_training_cells", train_cells)

        logistic, lookup, metadata = fit_feature_benchmarks(
            train_cells, test_cells, config["lookup_prior_strength"]
        )
        benchmark_metadata[process] = metadata

        test_cells["joint_logistic"], test_cells["joint_lookup"] = logistic, lookup
        save_table(directory, f"{process}_heldout_cell_predictions", test_cells)

        mapped = (
            pairs[["SamplingDateDistanceDays", column]]
            .rename(
                columns={
                    "SamplingDateDistanceDays": "time_days",
                    column: "genetic_distance",
                }
            )
            .merge(
                test_cells[
                    ["time_days", "genetic_distance", "joint_logistic", "joint_lookup"]
                ],
                on=["time_days", "genetic_distance"],
                how="left",
                sort=False,
                validate="many_to_one",
            )
        )

        candidate_scores = {
            "genetic_only": -pairs[column].to_numpy(float),
            "time_only": -pairs.SamplingDateDistanceDays.to_numpy(float),
            "joint_logistic": mapped.joint_logistic.to_numpy(),
            "joint_lookup": mapped.joint_lookup.to_numpy(),
        }

        for name, values in candidate_scores.items():
            ranks = ranking_table(pairs.IsRelated.to_numpy(bool), values)
            save_table(directory, f"{process}_{name}_precision_recall", ranks)
            benchmark_rows.append(
                {
                    "data_process": process,
                    "model": name,
                    "ap": average_precision(ranks),
                    "target_prevalence": float(pairs.IsRelated.mean()),
                    "training": "separate_observation_realization_same_tree"
                    if name.startswith("joint")
                    else "none",
                }
            )

            for selection, requested, threshold in thresholds_for_selections(
                ranks, settings["pairwise"]
            ):
                if selection == "fixed_threshold":
                    continue
                row = ranks.loc[ranks.score == threshold].iloc[0]
                operating_rows.append(
                    {
                        "data_process": process,
                        "model": name,
                        "selection": selection,
                        "requested": requested,
                        "threshold": threshold,
                        "precision": row.precision,
                        "recall": row.recall,
                        "selected_pairs": int(row.selected_pairs),
                        "selected_fraction": row.selected_fraction,
                    }
                )

    save_table(directory, "benchmark_ranking", benchmark_rows)
    save_table(directory, "benchmark_selections", operating_rows)
    write_json(
        directory / "benchmark_training.json",
        {
            "training_seed": training_seed,
            "evaluation_seed": seed,
            "training_source": run.logit_training_source,
            "models": benchmark_metadata,
            "limitation": "Independent dates and genomes on the same transmission topology, not independent epidemic trees.",
        },
    )
