import numpy as np
import pandas as pd
import pytest

from epilink_evaluation.config import validate
from epilink_evaluation.metrics.pairwise import metrics_at_thresholds
from epilink_evaluation.reporting.baseline_tables import operating_summary
from epilink_evaluation.selection.operating import (
    endpoint_frontiers,
    select_operating_points,
)
from epilink_evaluation.workflows.settings import settings_registry


def test_one_shared_development_cutoff_beats_coarse_grid_without_per_seed_optimization(
    small_config,
):
    # In each seed the positive outranks the negative, but score scales differ.
    # Independently optimizing each seed would misleadingly report F1=1.
    curves = {
        11: pd.DataFrame(
            {"threshold": [0.8, 0.7], "M0_f1": [1.0, 2 / 3], "M0_precision": [1, 0.5]}
        ),
        12: pd.DataFrame(
            {"threshold": [0.6, 0.4], "M0_f1": [1.0, 2 / 3], "M0_precision": [1, 0.5]}
        ),
    }
    small_config["scorers"] = ["LOGIT_D"]
    small_config["thresholds"]["logistic"] = [0.1, 0.9]
    definitions = settings_registry(small_config, {"LOGIT_D": [0.4, 0.6, 0.7, 0.8]})
    choices = {key: d for key, d in definitions.items() if d["kind"] == "pairwise"}
    rows = []
    for seed, curve in curves.items():
        rows.append(
            metrics_at_thresholds(
                curve,
                [d["threshold"] for d in choices.values()],
                True,
                {"M0_f1": 0, "M0_precision": np.nan},
            ).assign(
                seed=seed,
                split="development",
                pipeline="pairwise/LOGIT_D",
                setting_id=list(choices),
            )
        )
    frame = pd.concat(rows, ignore_index=True)
    criteria = [{"name": "custom", "objective": "M0_f1", "constraints": {}}]
    (point,) = select_operating_points(frame, definitions, criteria, [11, 12])
    assert point["definition"]["threshold"] == 0.6
    assert point["development_objective_mean"] == pytest.approx(5 / 6)
    assert point["development_objective_mean"] > 2 / 3  # best coarse-grid F1
    criteria[0]["constraints"] = {"M0_precision": {"min": 0.75}}
    (point,) = select_operating_points(frame, definitions, criteria, [11, 12])
    assert point["status"] == "infeasible"  # undefined precision also fails bounds


def test_exact_cutoffs_do_not_expand_full_graph_resolutions(
    small_config,
):
    small_config["scorers"] = ["LOGIT_D"]
    small_config["clustering"]["algorithms"] = ["components", "leiden"]
    leiden = small_config["clustering"]["leiden"]
    leiden["resolutions"] = [0.2, 0.8]
    assert "pairwise" not in small_config
    assert all(d["kind"] != "pairwise" for d in settings_registry(small_config).values())
    definitions = settings_registry(
        small_config, {"LOGIT_D": np.linspace(0, 1, 101).tolist()}
    )
    assert sum(d["kind"] == "pairwise" for d in definitions.values()) == 102
    for definition in definitions.values():
        if definition["kind"] == "components" and not definition["empty"]:
            assert definition["threshold"] in small_config["thresholds"]["logistic"]
        if definition["kind"] == "leiden":
            assert definition["resolution"] in [0.2, 0.8]
            assert definition["threshold"] is None
            assert definition["graph_mode"] == "full"
    assert sum(d["kind"] == "leiden" for d in definitions.values()) == 2


def test_operating_summary_uses_frozen_objective_not_criterion_name():
    criterion = "M0_in_name_but_secondary_objective"
    rows = [
        {
            "criterion": criterion,
            "pipeline": "p",
            "setting_id": "s",
            "seed": seed,
            "M0_f1": 0.2,
            "Mle1_f1": f1,
            "Mle2_f1": 0.9,
            "Mle1_precision": precision,
        }
        for seed, f1, precision in [(11, 0.6, 0.7), (12, 0.8, np.nan)]
    ]
    frozen = {
        "operating_points": [
            {
                "criterion": criterion,
                "pipeline": "p",
                "setting_id": "s",
                "status": "selected",
                "rule": {"objective": "Mle1_f1"},
            }
        ]
    }
    summary = operating_summary(pd.DataFrame(rows), frozen).iloc[0]
    assert summary.objective_endpoint == "Mle1"
    assert summary.objective_mean == pytest.approx(0.7)
    assert summary.objective_std == pytest.approx(np.std([0.6, 0.8], ddof=1))
    assert summary.M0_f1_mean == 0.2
    assert summary.Mle2_f1_mean == 0.9
    assert summary.Mle1_precision_count == 1
    assert summary.n_realizations == 2
    rows[1]["setting_id"] = "unexpected"
    with pytest.raises(ValueError, match="frozen"):
        operating_summary(pd.DataFrame(rows), frozen)


def test_frontiers_use_each_endpoints_precision_and_recall():
    frame = pd.DataFrame(
        {
            "pipeline": ["p", "p"],
            "setting_id": ["a", "b"],
            "M0_precision_mean": [0.9, 0.8],
            "M0_recall_mean": [0.6, 0.5],
            "Mle1_precision_mean": [0.2, 0.9],
            "Mle1_recall_mean": [0.2, 0.8],
            "Mle2_precision_mean": [0.8, 0.6],
            "Mle2_recall_mean": [0.6, 0.8],
        }
    )
    frontier = endpoint_frontiers(frame)
    assert list(frontier.loc[frontier.endpoint == "M0", "setting_id"]) == ["a"]
    assert list(frontier.loc[frontier.endpoint == "Mle1", "setting_id"]) == ["b"]
    assert set(frontier.loc[frontier.endpoint == "Mle2", "setting_id"]) == {"a", "b"}
    assert "M0_precision_mean" in frontier


@pytest.mark.parametrize("resolutions", [[], [0], [-0.1], [np.inf], [np.nan]])
def test_invalid_search_configuration_is_rejected(small_config, resolutions):
    small_config["clustering"]["leiden"]["resolutions"] = resolutions
    with pytest.raises(ValueError):
        validate(small_config)
