"""Pairwise informativeness comparisons across near-transmission horizons."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.specs import SCORE_METADATA
from synthetic_exploration.common import save_table, write_json, log
from synthetic_exploration.observations import DISTANCES
from synthetic_exploration.scores import average_precision, ranking_table, thresholds_for_selections

from .common import ENDPOINTS, endpoint_mask, m_category_rows, selected_m_summary


@dataclass
class CandidateScore:
    name: str
    family: str
    values: np.ndarray
    data_process: str
    inference_process: str = "not_applicable"
    trained_endpoint: str = "not_applicable"


def feature_cells_for_target(pairs: pd.DataFrame, genetic_column: str, target: np.ndarray) -> pd.DataFrame:
    """Aggregate identical observed inputs for a binary endpoint."""
    frame = pd.DataFrame({
        "time_days": np.rint(pairs.SamplingDateDistanceDays.to_numpy()).astype(int),
        "genetic_distance": np.rint(pairs[genetic_column].to_numpy()).astype(int),
        "target": np.asarray(target, dtype=bool),
    })
    cells = frame.groupby(["time_days", "genetic_distance"], sort=True).target.agg(
        n_pairs="size", n_target="sum"
    ).reset_index()
    cells["n_other"] = cells.n_pairs - cells.n_target
    cells["target_fraction"] = cells.n_target / cells.n_pairs
    return cells


def fit_logistic_probability(train_cells: pd.DataFrame, test_cells: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Fit P(endpoint | GD, TD) on aggregated feature cells."""
    columns = ["time_days", "genetic_distance"]
    prevalence = float(train_cells.n_target.sum() / train_cells.n_pairs.sum())
    if not 0 < prevalence < 1:
        raise ValueError("Logistic training requires both positive and negative pairs.")
    x_train = train_cells[columns].to_numpy(float)
    x = np.concatenate([x_train, x_train])
    y = np.r_[np.ones(len(x_train)), np.zeros(len(x_train))]
    weights = np.r_[train_cells.n_target.to_numpy(), train_cells.n_other.to_numpy()]
    valid = weights > 0
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    model.fit(x[valid], y[valid], logisticregression__sample_weight=weights[valid])
    predictions = model.predict_proba(test_cells[columns].to_numpy(float))[:, 1]
    classifier = model.named_steps["logisticregression"]
    scaler = model.named_steps["standardscaler"]
    return predictions, {
        "training_prevalence": prevalence,
        "logistic_coefficients_standardised": classifier.coef_[0].tolist(),
        "logistic_intercept": float(classifier.intercept_[0]),
        "feature_mean": scaler.mean_.tolist(),
        "feature_scale": scaler.scale_.tolist(),
    }


def map_cell_predictions(pairs: pd.DataFrame, genetic_column: str, cells: pd.DataFrame, prediction_column: str) -> np.ndarray:
    observed = pd.DataFrame({
        "time_days": np.rint(pairs.SamplingDateDistanceDays.to_numpy()).astype(int),
        "genetic_distance": np.rint(pairs[genetic_column].to_numpy()).astype(int),
    })
    mapped = observed.merge(
        cells[["time_days", "genetic_distance", prediction_column]],
        on=["time_days", "genetic_distance"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if mapped[prediction_column].isna().any():
        raise ValueError("Every evaluation pair should map to an evaluation feature cell.")
    return mapped[prediction_column].to_numpy(float)


def build_candidate_scores(pairs: pd.DataFrame, training_pairs: pd.DataFrame,
                           epilink_scores: pd.DataFrame, model_names: list[str],
                           directory, settings) -> tuple[dict[str, CandidateScore], pd.DataFrame]:
    """Build compatibility, genetic-only, and horizon-specific logistic scores."""
    candidates: dict[str, CandidateScore] = {}
    rows = []
    for model in model_names:
        meta = SCORE_METADATA[model]
        name = f"compatibility_{model}"
        candidates[name] = CandidateScore(
            name=name,
            family="compatibility",
            values=epilink_scores[model].to_numpy(float),
            data_process=meta["data_process"],
            inference_process=meta["inference_process"],
        )
        rows.append({**meta, "score_name": name, "score_family": "compatibility",
                     "trained_endpoint": "not_applicable"})

    training_metadata = []
    for process, column in DISTANCES.items():
        name = f"genetic_only_{process}"
        candidates[name] = CandidateScore(
            name=name,
            family="genetic_only",
            values=-pairs[column].to_numpy(float),
            data_process=process,
        )
        rows.append({"score_name": name, "score_family": "genetic_only",
                     "data_process": process, "inference_process": "not_applicable",
                     "trained_endpoint": "not_applicable"})
        for endpoint in ENDPOINTS:
            train_y = endpoint_mask(training_pairs, endpoint)
            test_y = endpoint_mask(pairs, endpoint)
            train_cells = feature_cells_for_target(training_pairs, column, train_y)
            test_cells = feature_cells_for_target(pairs, column, test_y)
            cell_predictions, metadata = fit_logistic_probability(train_cells, test_cells)
            prediction_column = f"logistic_{endpoint.key}"
            test_cells[prediction_column] = cell_predictions
            name = f"logistic_{endpoint.key}_{process}"
            candidates[name] = CandidateScore(
                name=name,
                family="logistic_probability",
                values=map_cell_predictions(pairs, column, test_cells, prediction_column),
                data_process=process,
                trained_endpoint=endpoint.key,
            )
            save_table(directory, f"{name}_feature_cells", test_cells)
            training_metadata.append({"score_name": name, "data_process": process,
                                      "trained_endpoint": endpoint.key, **metadata})
            rows.append({"score_name": name, "score_family": "logistic_probability",
                         "data_process": process, "inference_process": "not_applicable",
                         "trained_endpoint": endpoint.key})
    metadata_frame = save_table(directory, "score_metadata", rows)
    write_json(directory / "logistic_training.json", {
        "training": "separate observation realization on the same transmission tree",
        "models": training_metadata,
    })
    return candidates, metadata_frame


def candidate_applies_to_endpoint(candidate: CandidateScore, endpoint_key: str) -> bool:
    return candidate.family != "logistic_probability" or candidate.trained_endpoint == endpoint_key


def investigate_pairwise(pairs: pd.DataFrame, training_pairs: pd.DataFrame,
                         epilink_scores: pd.DataFrame, model_names: list[str],
                         directory, settings) -> dict[str, CandidateScore]:
    log("1/3: pairwise horizon rankings and contamination summaries")
    directory.mkdir(parents=True, exist_ok=True)
    candidates, _ = build_candidate_scores(pairs, training_pairs, epilink_scores, model_names, directory, settings)
    score_settings = {"thresholds": [], **settings["scores"]}
    rank_rows, summary_rows, operating_rows, composition_rows = [], [], [], []
    calibration_rows = []
    for endpoint in ENDPOINTS:
        y = endpoint_mask(pairs, endpoint)
        prevalence = float(np.mean(y))
        for candidate in candidates.values():
            if not candidate_applies_to_endpoint(candidate, endpoint.key):
                continue
            ranks = ranking_table(y, candidate.values)
            rank_rows.extend({
                "endpoint": endpoint.key,
                "endpoint_label": endpoint.label,
                "score_name": candidate.name,
                "score_family": candidate.family,
                "data_process": candidate.data_process,
                "inference_process": candidate.inference_process,
                "trained_endpoint": candidate.trained_endpoint,
                **row,
            } for row in ranks.to_dict(orient="records"))
            ap = average_precision(ranks)
            summary = {
                "endpoint": endpoint.key,
                "endpoint_label": endpoint.label,
                "score_name": candidate.name,
                "score_family": candidate.family,
                "data_process": candidate.data_process,
                "inference_process": candidate.inference_process,
                "trained_endpoint": candidate.trained_endpoint,
                "ap": ap,
                "target_prevalence": prevalence,
                "unique_scores": len(ranks),
                "zero_score_fraction": float(np.mean(candidate.values == 0)),
            }
            summary_rows.append(summary)
            if candidate.family == "logistic_probability":
                calibration_rows.append({**summary,
                    "brier_score": brier_score_loss(y, candidate.values),
                    "log_loss": log_loss(y, np.clip(candidate.values, 1e-12, 1 - 1e-12))})
            for selection, requested, threshold in thresholds_for_selections(ranks, score_settings):
                if selection == "fixed_threshold":
                    continue
                selected = candidate.values >= threshold
                selected_target = int(np.sum(selected & y))
                base = {
                    "endpoint": endpoint.key,
                    "endpoint_label": endpoint.label,
                    "score_name": candidate.name,
                    "score_family": candidate.family,
                    "data_process": candidate.data_process,
                    "inference_process": candidate.inference_process,
                    "trained_endpoint": candidate.trained_endpoint,
                    "selection": selection,
                    "requested": requested,
                    "threshold_inclusive": threshold,
                }
                n = int(np.sum(selected))
                operating_rows.append({**base,
                    "selected_pairs": n,
                    "selected_fraction": n / len(pairs),
                    "precision": selected_target / n if n else np.nan,
                    "recall": selected_target / int(np.sum(y)) if np.sum(y) else np.nan,
                    "target_enrichment": (selected_target / n) / prevalence if n and prevalence else np.nan,
                    **selected_m_summary(pairs, selected),
                    "pairs_tied_at_cutoff": int(np.sum(candidate.values == threshold)),
                })
                composition_rows.extend(m_category_rows(base, pairs, selected))
    save_table(directory, "precision_recall", pd.DataFrame(rank_rows))
    save_table(directory, "ranking_summary", summary_rows)
    save_table(directory, "operating_points", operating_rows)
    save_table(directory, "selection_M_composition", composition_rows)
    save_table(directory, "logistic_calibration", calibration_rows)
    save_table(directory, "contamination_summary", pd.DataFrame(operating_rows)[[
        "endpoint", "score_name", "score_family", "data_process", "selection", "requested",
        "precision", "recall", "Mge3_contamination_fraction", "median_M_connected", "p90_M_connected",
    ]])
    return candidates
