"""Epidemic date simulation utilities."""

from __future__ import annotations

import networkx as nx
from scipy.stats import gamma

from ..model.profiles import InfectiousnessToTransmissionTime


def simulate_epidemic_dates(
    transmission_profile: InfectiousnessToTransmissionTime,
    tree: nx.DiGraph,
    prop_sampled: float = 1.0,
    sampling_scale: float = 1.0,
    sampling_shape: float = 3.0,
    root_start_range: int = 30,
) -> nx.DiGraph:
    """Populate a transmission tree with simulated epidemic dates.

    Parameters
    ----------
    transmission_profile : InfectiousnessToTransmissionTime
        Transmission-time model used to sample latent, presymptomatic, and
        transmission intervals.
    tree : networkx.DiGraph
        Directed transmission tree.
    prop_sampled : float, default=1.0
        Fraction of nodes marked as sampled.
    sampling_scale : float, default=1.0
        Scale parameter for the Gamma-distributed testing delay.
    sampling_shape : float, default=3.0
        Shape parameter for the Gamma-distributed testing delay.
    root_start_range : int, default=30
        Root exposure dates are drawn uniformly from
        ``range(root_start_range)`` when positive.

    Returns
    -------
    networkx.DiGraph
        Copy of ``tree`` with simulated epidemic annotations.
    """

    annotated_tree = tree.copy()
    rng = transmission_profile.rng

    num_nodes = annotated_tree.number_of_nodes()
    num_sampled = int(round(prop_sampled * num_nodes))
    sampled_node_ids = set(
        rng.choice(list(annotated_tree.nodes()), size=num_sampled, replace=False)
    )
    nx.set_node_attributes(
        annotated_tree,
        {node: (node in sampled_node_ids) for node in annotated_tree},
        "sampled",
    )

    def sample_stage_intervals() -> tuple[float, float, float]:
        if sampling_scale <= 0:
            latent_periods = (
                transmission_profile.parameters.latent_shape
                * transmission_profile.parameters.incubation_scale
            )
            presymptomatic_periods = (
                transmission_profile.parameters.presymptomatic_shape
                * transmission_profile.parameters.incubation_scale
            )
            testing_delays = 0.0
        else:
            latent_periods = transmission_profile.sample_latent_periods().item()
            presymptomatic_periods = transmission_profile.sample_presymptomatic_periods().item()
            testing_delays = gamma.rvs(
                a=sampling_shape,
                scale=sampling_scale,
                random_state=rng,
            )

        return latent_periods, presymptomatic_periods, testing_delays

    roots = [node for node, degree in annotated_tree.in_degree(annotated_tree.nodes) if degree == 0]

    for root in roots:
        exposure_date = int(rng.choice(range(root_start_range))) if root_start_range > 0 else 0
        latent_period_days, presymptomatic_period_days, testing_delay_days = (
            sample_stage_intervals()
        )

        annotated_tree.nodes[root].update(
            {
                "exposure_date": exposure_date,
                "date_infectious": exposure_date + latent_period_days,
                "date_symptom_onset": (
                    exposure_date + latent_period_days + presymptomatic_period_days
                ),
                "sample_date": (
                    exposure_date
                    + latent_period_days
                    + presymptomatic_period_days
                    + testing_delay_days
                ),
                "seed": True,
            }
        )

        for parent, child in nx.dfs_edges(annotated_tree, source=root):
            transmission_interval_days = (
                0.0 if sampling_scale <= 0 else transmission_profile.rvs().item()
            )
            latent_period_days, presymptomatic_period_days, testing_delay_days = (
                sample_stage_intervals()
            )
            parent_infectious_date = annotated_tree.nodes[parent]["date_infectious"]

            child_exposure_date = parent_infectious_date + transmission_interval_days
            child_infectious_date = child_exposure_date + latent_period_days
            child_symptom_onset_date = child_infectious_date + presymptomatic_period_days
            child_sample_date = child_symptom_onset_date + testing_delay_days

            annotated_tree.nodes[child].update(
                {
                    "exposure_date": child_exposure_date,
                    "date_infectious": child_infectious_date,
                    "date_symptom_onset": child_symptom_onset_date,
                    "sample_date": child_sample_date,
                    "seed": False,
                }
            )

    return annotated_tree


__all__ = ["simulate_epidemic_dates"]
