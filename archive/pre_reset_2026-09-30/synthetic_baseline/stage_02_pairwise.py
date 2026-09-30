"""Stage 2: Pairwise informativeness comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.specs import SCORE_METADATA, EPILINK_SPECS
from synthetic_exploration.scores import ranking_table, average_precision, thresholds_for_selections
from synthetic_exploration.observations import DISTANCES

from .common import (
    log, save_table, write_json, ENDPOINTS, endpoint_mask,
    m_category_counts, selected_m_summary, relationship_category_counts,
)


def feature_cells_for_endpoint(pairs: pd.DataFrame, genetic_column: str, 
                                endpoint: str, target_mask: np.ndarray) -> pd.DataFrame:
    """Aggregate identical observed inputs for a binary endpoint."""
    frame = pd.DataFrame({
        "time_days": np.rint(pairs.SamplingDateDistanceDays.to_numpy()).astype(int),
        "genetic_distance": np.rint(pairs[genetic_column].to_numpy()).astype(int),
        "target": target_mask,
    })
    cells = frame.groupby(["time_days", "genetic_distance"], sort=True).agg(
        n_pairs=("target", "size"),
        n_target=("target", "sum"),
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
    
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
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


def map_cell_predictions(pairs: pd.DataFrame, genetic_column: str, 
                         cells: pd.DataFrame, prediction_column: str) -> np.ndarray:
    """Map cell-level predictions to all pairs."""
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
                           directory) -> dict:
    """Build compatibility, genetic-only, and horizon-specific logistic scores."""
    candidates = {}
    rows = []
    training_metadata = []
    
    # Compatibility scores
    for model in model_names:
        meta = SCORE_METADATA[model]
        name = f"compatibility_{model}"
        candidates[name] = {
            "name": name,
            "family": "compatibility",
            "values": epilink_scores[model].to_numpy(float),
            "data_process": meta["data_process"],
            "inference_process": meta["inference_process"],
            "trained_endpoint": "not_applicable",
        }
        rows.append({**meta, "score_name": name, "score_family": "compatibility",
                     "trained_endpoint": "not_applicable"})
    
    # Genetic-only and logistic scores
    for process, genetic_column in DISTANCES.items():
        # Genetic-only
        name = f"genetic_only_{process}"
        candidates[name] = {
            "name": name,
            "family": "genetic_only",
            "values": -pairs[genetic_column].to_numpy(float),
            "data_process": process,
            "inference_process": "not_applicable",
            "trained_endpoint": "not_applicable",
        }
        rows.append({"score_name": name, "score_family": "genetic_only",
                     "data_process": process, "inference_process": "not_applicable",
                     "trained_endpoint": "not_applicable"})
        
        # Logistic for each endpoint
        for endpoint in ENDPOINTS:
            train_y = endpoint_mask(training_pairs, endpoint)
            test_y = endpoint_mask(pairs, endpoint)
            
            train_cells = feature_cells_for_endpoint(training_pairs, genetic_column, endpoint.key, train_y)
            test_cells = feature_cells_for_endpoint(pairs, genetic_column, endpoint.key, test_y)
            
            predictions, metadata = fit_logistic_probability(train_cells, test_cells)
            
            prediction_column = f"logistic_{endpoint.key}"
            test_cells[prediction_column] = predictions
            save_table(directory, f"logistic_{endpoint.key}_{process}_cells", test_cells)
            
            name = f"logistic_{endpoint.key}_{process}"
            candidates[name] = {
                "name": name,
                "family": "logistic_probability",
                "values": map_cell_predictions(pairs, genetic_column, test_cells, prediction_column),
                "data_process": process,
                "inference_process": "not_applicable",
                "trained_endpoint": endpoint.key,
            }
            training_metadata.append({"score_name": name, "data_process": process,
                                      "trained_endpoint": endpoint.key, **metadata})
            rows.append({"score_name": name, "score_family": "logistic_probability",
                         "data_process": process, "inference_process": "not_applicable",
                         "trained_endpoint": endpoint.key})
    
    save_table(directory, "score_metadata", rows)
    write_json(directory / "logistic_training.json", {
        "training": "separate observation realization on the same transmission tree",
        "models": training_metadata,
    })
    
    return candidates


def candidate_applies_to_endpoint(candidate: dict, endpoint_key: str) -> bool:
    return candidate["family"] != "logistic_probability" or candidate["trained_endpoint"] == endpoint_key


def investigate_pairwise(pairs: pd.DataFrame, training_pairs: pd.DataFrame,
                         epilink_scores: pd.DataFrame, model_names: list[str],
                         directory, settings) -> dict:
    """Stage 2: Pairwise ranking and contamination analysis."""
    log("2/4: pairwise horizon rankings and contamination summaries")
    directory.mkdir(parents=True, exist_ok=True)
    
    candidates = build_candidate_scores(pairs, training_pairs, epilink_scores, model_names, directory)
    score_settings = {"thresholds": [], **settings["pairwise"]}
    
    rank_rows, summary_rows, operating_rows, composition_rows = [], [], [], []
    calibration_rows = []
    
    for endpoint in ENDPOINTS:
        y = endpoint_mask(pairs, endpoint)
        prevalence = float(np.mean(y))
        
        for candidate in candidates.values():
            if not candidate_applies_to_endpoint(candidate, endpoint.key):
                continue
            
            values = candidate["values"]
            ranks = ranking_table(y, values)
            
            # Ranking data
            for row_dict in ranks.to_dict(orient="records"):
                rank_rows.append({
                    "endpoint": endpoint.key,
                    "endpoint_label": endpoint.label,
                    "score_name": candidate["name"],
                    "score_family": candidate["family"],
                    "data_process": candidate["data_process"],
                    "inference_process": candidate["inference_process"],
                    "trained_endpoint": candidate["trained_endpoint"],
                    **row_dict,
                })
            
            # Summary
            ap = average_precision(ranks)
            summary_rows.append({
                "endpoint": endpoint.key,
                "endpoint_label": endpoint.label,
                "score_name": candidate["name"],
                "score_family": candidate["family"],
                "data_process": candidate["data_process"],
                "inference_process": candidate["inference_process"],
                "trained_endpoint": candidate["trained_endpoint"],
                "ap": ap,
                "target_prevalence": prevalence,
                "unique_scores": len(ranks),
                "zero_score_fraction": float(np.mean(values == 0)),
            })
            
            # Calibration (logistic only)
            if candidate["family"] == "logistic_probability":
                calibration_rows.append({
                    "endpoint": endpoint.key,
                    "score_name": candidate["name"],
                    "ap": ap,
                    "brier_score": brier_score_loss(y, values),
                    "log_loss": log_loss(y, np.clip(values, 1e-12, 1 - 1e-12)),
                })
            
            # Operating points
            for selection, requested, threshold in thresholds_for_selections(ranks, score_settings):
                if selection == "fixed_threshold":
                    continue
                
                selected = values >= threshold
                selected_target = int(np.sum(selected & y))
                n = int(np.sum(selected))
                
                base = {
                    "endpoint": endpoint.key,
                    "endpoint_label": endpoint.label,
                    "score_name": candidate["name"],
                    "score_family": candidate["family"],
                    "data_process": candidate["data_process"],
                    "inference_process": candidate["inference_process"],
                    "trained_endpoint": candidate["trained_endpoint"],
                    "selection": selection,
                    "requested": requested,
                    "threshold_inclusive": threshold,
                }
                
                operating_rows.append({
                    **base,
                    "selected_pairs": n,
                    "selected_fraction": n / len(pairs),
                    "precision": selected_target / n if n else np.nan,
                    "recall": selected_target / int(np.sum(y)) if np.sum(y) else np.nan,
                    "target_enrichment": (selected_target / n) / prevalence if n and prevalence else np.nan,
                    **selected_m_summary(pairs, selected),
                    "pairs_tied_at_cutoff": int(np.sum(values == threshold)),
                })
                
                # Relationship composition (8 categories)
                rel_counts = relationship_category_counts(pairs, selected)
                for cat_key, count in rel_counts.items():
                    if cat_key != "unknown":
                        composition_rows.append({
                            **base,
                            "relationship_category": cat_key,
                            "n_pairs": count,
                            "proportion": count / n if n else np.nan,
                        })
    
    save_table(directory, "precision_recall", pd.DataFrame(rank_rows))
    save_table(directory, "ranking_summary", summary_rows)
    save_table(directory, "operating_points", operating_rows)
    save_table(directory, "selection_M_composition", composition_rows)
    save_table(directory, "logistic_calibration", calibration_rows)
    
    # Contamination summary
    contamination = pd.DataFrame(operating_rows)[[
        "endpoint", "score_name", "score_family", "data_process", "selection", "requested",
        "precision", "recall", "Mge3_contamination_fraction", "median_M_connected", "p90_M_connected",
    ]]
    save_table(directory, "contamination_summary", contamination)
    
    return candidates
