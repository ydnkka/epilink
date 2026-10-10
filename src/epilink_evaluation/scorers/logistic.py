"""Count-weighted logistic fitting equivalent to the uncompressed pair table."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..schemas import DISTANCES


def training_cells(observations, truth, process):
    if not observations.pair_id.equals(truth.pair_id):
        raise ValueError("Training truth is not aligned with observations")
    frame = pd.DataFrame(
        {
            "GD": observations[DISTANCES[process]].to_numpy(float),
            "TD": observations.TD.to_numpy(float),
            "positive": truth.M.eq(0).fillna(False).to_numpy(bool),
        }
    )
    return (
        frame.groupby(["GD", "TD"], sort=True)
        .positive.agg(positives="sum", count="size")
        .reset_index()
    )


def fit_logistic(cells, regularization=1.0):
    cells = (
        cells.groupby(["GD", "TD"], sort=True)[["positives", "count"]]
        .sum()
        .reset_index()
    )
    x = cells[["GD", "TD"]].to_numpy(float)
    positives = cells.positives.to_numpy(float)
    negatives = cells["count"].to_numpy(float) - positives
    if positives.sum() <= 0 or negatives.sum() <= 0:
        raise ValueError("Logistic training requires both target and non-target pairs")
    scaler = StandardScaler().fit(x, sample_weight=cells["count"].to_numpy(float))
    scale = scaler.scale_
    mean = scaler.mean_
    if mean is None or scale is None:
        raise RuntimeError("Logistic scaler did not compute feature scales")
    scaled = scaler.transform(x)
    features = np.concatenate([scaled, scaled])
    labels = np.r_[np.ones(len(x)), np.zeros(len(x))]
    weights = np.r_[positives, negatives]
    keep = weights > 0
    max_iter = 2000
    classifier = LogisticRegression(
        C=regularization, solver="lbfgs", max_iter=max_iter, tol=1e-9
    ).fit(features[keep], labels[keep], sample_weight=weights[keep])
    if classifier.n_iter_[0] >= max_iter:
        raise RuntimeError("Logistic optimizer did not converge")
    return {
        "features": ["GD", "TD"],
        "target": "M0",
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "C": regularization,
        "training_pairs": int(weights.sum()),
        "training_prevalence": float(positives.sum() / weights.sum()),
    }


def predict_logistic(observations, process, model):
    x = observations[[DISTANCES[process], "TD"]].to_numpy(float)
    return expit(
        ((x - np.asarray(model["mean"])) / np.asarray(model["scale"]))
        @ np.asarray(model["coef"])
        + model["intercept"]
    )
