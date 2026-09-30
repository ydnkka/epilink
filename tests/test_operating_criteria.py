import numpy as np
import pandas as pd
import pytest

from epilink_evaluation.selection.operating import aggregate_settings, pareto_frontier, select_operating_points


def evidence():
    return pd.DataFrame([
        {"split": "development", "pipeline": "method", "setting_id": setting, "seed": seed,
         "M0_f1": f1, "M0_precision": precision}
        for setting, results in {
            "a": [(0.8, 0.9), (0.6, 0.8)],
            "b": [(0.7, 0.8), (0.7, 0.8)],
            "c": [(0.9, 0.95), (0.9, 0.5)],
        }.items()
        for seed, (f1, precision) in zip([11, 12], results)
    ])


def test_constraints_hold_on_every_realization_and_ties_prefer_stability():
    frame = evidence()
    definitions = {name: {"name": name} for name in "abc"}
    criteria = [{"name": "precise", "objective": "M0_f1", "constraints": {"M0_precision": {"min": 0.7}}}]
    point, = select_operating_points(frame, definitions, criteria, [11, 12])
    assert point["setting_id"] == "b"
    assert point["development_objective_sd"] == 0
    assert point["development_objective_mean"] == pytest.approx(0.7)
    # A bound above every setting's worst realization stays explicitly infeasible.
    criteria[0]["constraints"]["M0_precision"]["min"] = 0.85
    point, = select_operating_points(frame, definitions, criteria, [11, 12])
    assert point["status"] == "infeasible"
    assert point["setting_id"] is None


@pytest.mark.parametrize("corruption", ["evaluation", "missing_seed", "duplicate"])
def test_selection_rejects_invalid_development_evidence(corruption):
    frame = evidence()
    if corruption == "evaluation":
        frame.loc[0, "split"] = "evaluation"
    elif corruption == "missing_seed":
        frame = frame.loc[frame.seed == 11]
    else:
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError):
        select_operating_points(frame, {}, [{"name": "test", "objective": "M0_f1"}], [11, 12])


def test_nonfinite_and_incomplete_settings_cannot_win():
    frame = evidence()
    frame.loc[frame.setting_id == "c", "M0_f1"] = np.nan
    frame = frame.loc[~((frame.setting_id == "a") & (frame.seed == 12))]
    point, = select_operating_points(frame, {"b": {}}, [{"name": "test", "objective": "M0_f1"}], [11, 12])
    assert point["setting_id"] == "b"
    summary = aggregate_settings(evidence()).set_index("setting_id")
    assert summary.loc["a", "M0_f1_mean"] == pytest.approx(0.7)
    assert summary.loc["a", "M0_f1_std"] == pytest.approx(np.std([0.8, 0.6], ddof=1))


def test_frontier_keeps_exact_ties_and_excludes_dominated_settings():
    frame = pd.DataFrame({
        "pipeline": ["method"] * 5, "setting_id": list("abcde"),
        "M0_precision": [0.9, 0.9, 0.8, 0.8, 0.7],
        "M0_recall": [0.2, 0.1, 0.4, 0.4, 0.3],
    })
    assert set(pareto_frontier(frame).setting_id) == {"a", "c", "d"}
