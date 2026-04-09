from __future__ import annotations

from typing import Any, Mapping

from epilink import EpiLink, InfectiousnessToTransmission, NaturalHistoryParameters

try:
    from .specs import (
        DEFAULT_MAXIMUM_DEPTH,
        DEFAULT_NUM_SIMULATIONS,
        DEFAULT_TARGET,
        expand_baseline_parameters,
        resolve_target_labels,
    )
except ImportError:  # pragma: no cover - support direct script execution
    from specs import (
        DEFAULT_MAXIMUM_DEPTH,
        DEFAULT_NUM_SIMULATIONS,
        DEFAULT_TARGET,
        expand_baseline_parameters,
        resolve_target_labels,
    )


def build_natural_history_parameters(parameters: Mapping[str, Any]) -> NaturalHistoryParameters:
    return NaturalHistoryParameters(
        incubation_shape=float(parameters.get("incubation_shape", 5.807)),
        incubation_scale=float(parameters.get("incubation_scale", 0.948)),
        latent_shape=float(parameters.get("latent_shape", 3.38)),
        symptomatic_rate=float(parameters.get("symptomatic_rate", 0.37)),
        symptomatic_shape=float(parameters.get("symptomatic_shape", 1.0)),
        transmission_rate_ratio=float(parameters.get("transmission_rate_ratio", 2.29)),
        testing_delay_shape=float(parameters.get("testing_delay_shape", 1.0)),
        testing_delay_scale=float(parameters.get("testing_delay_scale", 1.0)),
        substitution_rate=float(parameters.get("substitution_rate", 1e-3)),
        relaxation=float(parameters.get("relaxation", 0.33)),
        genome_length=int(parameters.get("genome_length", 29903)),
    )


def nhp_from_baseline(
    baseline: Mapping[str, Any],
    fixed_parameters: Mapping[str, Any],
) -> NaturalHistoryParameters:
    return build_natural_history_parameters(expand_baseline_parameters(baseline, fixed_parameters))


def build_linkage_model(
    parameters: Mapping[str, Any],
    *,
    mutation_process: str,
    rng_seed: int,
) -> EpiLink:
    transmission_profile = InfectiousnessToTransmission(
        parameters=build_natural_history_parameters(parameters),
        rng_seed=rng_seed,
    )
    return EpiLink(
        mutation_process=mutation_process,
        transmission_profile=transmission_profile,
        maximum_depth=int(parameters.get("maximum_depth", DEFAULT_MAXIMUM_DEPTH)),
        mc_samples=int(parameters.get("num_simulations", DEFAULT_NUM_SIMULATIONS)),
        target=resolve_target_labels(parameters.get("target", DEFAULT_TARGET)),
    )


def build_linkage_models(
    parameters: Mapping[str, Any],
    *,
    rng_seed: int,
) -> dict[str, EpiLink]:
    return {
        mutation_process: build_linkage_model(
            parameters,
            mutation_process=mutation_process,
            rng_seed=rng_seed,
        )
        for mutation_process in ("deterministic", "stochastic")
    }
