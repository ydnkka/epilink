from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from epilink import EpiLink, InfectiousnessToTransmission

from ..config import natural_history
from ..schemas import DISTANCES, ScoreSpec
from .logistic import predict_logistic


@dataclass
class ScoringContext:
    config: dict
    logistic_models: dict
    epilink_models: dict = field(default_factory=dict)

    def epilink(self, process):
        if process not in self.epilink_models:
            profile = InfectiousnessToTransmission(
                parameters=natural_history(self.config["inference"]),
                rng_seed=self.config["scorer"]["seed"],
            )
            self.epilink_models[process] = EpiLink(
                mutation_process=process,
                transmission_profile=profile,
                maximum_depth=0,
                mc_samples=self.config["scorer"]["mc_samples"],
                target=("ad(0)", "ca(0,0)"),
            )
        return self.epilink_models[process]


class Scorer(Protocol):
    spec: ScoreSpec

    def predict(self, observations, context: ScoringContext) -> np.ndarray: ...


@dataclass(frozen=True)
class EpiLinkScorer(Scorer):
    spec: ScoreSpec

    def predict(self, observations, context):
        return np.asarray(
            context.epilink(self.spec.inference_process).score_target(
                sample_time_difference=observations.TD.to_numpy(float),
                genetic_distance=observations[
                    DISTANCES[self.spec.data_process]
                ].to_numpy(float),
            )
        )


@dataclass(frozen=True)
class GeneticScorer(Scorer):
    spec: ScoreSpec

    def predict(self, observations, context):
        return observations[DISTANCES[self.spec.data_process]].to_numpy(float)


@dataclass(frozen=True)
class LogisticScorer(Scorer):
    spec: ScoreSpec

    def predict(self, observations, context):
        return predict_logistic(
            observations,
            self.spec.data_process,
            context.logistic_models[self.spec.data_process],
        )


SCORERS: dict[str, Scorer] = {}
for name, inference, data in (
    ("EDD", "deterministic", "deterministic"),
    ("EDS", "deterministic", "stochastic"),
    ("ESD", "stochastic", "deterministic"),
    ("ESS", "stochastic", "stochastic"),
):
    SCORERS[name] = EpiLinkScorer(ScoreSpec(name, "epilink", data, inference))
for suffix, process in (("D", "deterministic"), ("S", "stochastic")):
    name = f"GD_{suffix}"
    SCORERS[name] = GeneticScorer(
        ScoreSpec(name, "genetic", process, target="none", higher_is_better=False)
    )
    name = f"LOGIT_{suffix}"
    SCORERS[name] = LogisticScorer(ScoreSpec(name, "logistic", process))
