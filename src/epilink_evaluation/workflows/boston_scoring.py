"""Training-free scores on Boston's single observed TN93 distance table."""

from copy import deepcopy

import numpy as np
import pandas as pd

from ..provenance import fingerprint
from ..schemas import ScoreSpec

# S/D in the GD names identifies the synthetic origin of the operating rule,
# not separate empirical measurements. Both use the same observed GD values.
BASELINE_SCORES = {"ES": "ESS", "ED": "EDS", "GD_S": "GD_S", "GD_D": "GD_D"}
BOSTON_SPECS = {
    "ES": ScoreSpec("ES", "epilink", "empirical", "stochastic"),
    "ED": ScoreSpec("ED", "epilink", "empirical", "deterministic"),
    "GD_S": ScoreSpec("GD_S", "genetic", "empirical", target="none", higher_is_better=False),
    "GD_D": ScoreSpec("GD_D", "genetic", "empirical", target="none", higher_is_better=False),
}


def validate_scorers(names):
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or name not in BOSTON_SPECS for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Boston scorers must be a unique nonempty subset of ES, ED, GD_S, GD_D")


def score_observations(observations, context, names):
    """Evaluate compatibility or distance without fitting or labelled pairs."""
    validate_scorers(names)
    gd, td = observations.GD.to_numpy(float), observations.TD.to_numpy(float)
    if not (np.isfinite(gd).all() and np.isfinite(td).all()
            and (gd >= 0).all() and (td >= 0).all()):
        raise ValueError("Boston GD and TD must be finite nonnegative observations")
    scores = {}
    for name in names:
        spec = BOSTON_SPECS[name]
        if spec.family == "genetic" or not len(observations):
            values = gd.copy()
        else:
            values = np.asarray(context.epilink(spec.inference_process).score_target(
                sample_time_difference=td, genetic_distance=gd,
            ), dtype=float).reshape(-1)
        if values.shape != gd.shape or not (np.isfinite(values) & (values >= 0)).all():
            raise ValueError(f"Invalid Boston scores from {name}")
        scores[name] = values
    return pd.DataFrame(scores, index=observations.index)


def operating_settings(reference, names):
    """Rename supported frozen decisions while retaining their source identity."""
    validate_scorers(names)
    aliases = {BASELINE_SCORES[name]: name for name in names}
    missing = set(aliases) - set(reference.config["scorers"])
    if missing:
        raise ValueError(f"Baseline lacks the requested Boston source scorers: {sorted(missing)}")
    definitions, points = {}, []
    for source in reference.frozen["operating_points"]:
        parts = source["pipeline"].split("/")
        if parts[0] not in ("pairwise", "components", "leiden") or parts[1] not in aliases:
            continue
        name = aliases[parts[1]]
        point = deepcopy(source)
        point["pipeline"] = "/".join([parts[0], name, *parts[2:]])
        point["baseline_setting_id"] = source["setting_id"]
        if source["status"] == "selected":
            definition = deepcopy(reference.selected[source["setting_id"]])
            definition.update(
                baseline_setting_id=source["setting_id"],
                baseline_score_name=definition["score_name"],
                baseline_data_process=definition["data_process"],
                score_name=name, data_process="empirical", pipeline=point["pipeline"],
            )
            key = fingerprint(definition)[:20]
            definitions[key] = definition
            point.update(setting_id=key, definition=definition)
        points.append(point)
    if not any(d["kind"] in ("components", "leiden") for d in definitions.values()):
        raise ValueError("Reference has no selected graph settings for the requested Boston scorers")
    selection = {
        "reference_selection_fingerprint": reference.identity["selection_fingerprint"],
        "criteria": deepcopy(reference.frozen["criteria"]),
        "operating_points": points,
    }
    return definitions, selection
