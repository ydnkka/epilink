"""Run the empirical Boston clustering workflow."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import igraph as ig
import numpy as np
import pandas as pd
from epilink import EpiLink, InfectiousnessToTransmission, NaturalHistoryParameters

try:
    from .config import gamma_mean_cv_to_shape_scale, load_config, resolve_path
    from .leiden import run_leiden_partition
    from .metrics import analyse_partition_composition
except ImportError:
    from config import gamma_mean_cv_to_shape_scale, load_config, resolve_path
    from leiden import run_leiden_partition
    from metrics import analyse_partition_composition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def add_temporal_distance(
    pairwise_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    id_col_1: str,
    id_col_2: str,
    date_col: str,
    out_col: str,
) -> pd.DataFrame:
    date_map = metadata_df.set_index("SeqID")[date_col]
    d1 = pd.to_datetime(pairwise_df[id_col_1].map(date_map))
    d2 = pd.to_datetime(pairwise_df[id_col_2].map(date_map))
    pairwise_df[out_col] = (d1 - d2).abs().dt.days.astype(int)
    return pairwise_df


def build_graph(
    pairwise_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    id_col_1: str,
    id_col_2: str,
    exposure_col: str,
    minimum_weight: float = 0.0001,
    weight_column: str = "EpiLinkStochasticScore",
) -> ig.Graph:
    """Build an igraph Graph with vertex metadata and weighted edges."""
    all_ids = pd.unique(pairwise_df[[id_col_1, id_col_2]].values.ravel())
    id_to_index = {seq_id: i for i, seq_id in enumerate(all_ids)}
    metadata_dict = metadata_df.set_index("SeqID").to_dict(orient="index")

    g = ig.Graph(n=len(all_ids))
    g.vs["SeqID"] = all_ids.tolist()
    g.vs["Date"] = [metadata_dict.get(sid, {}).get("Date") for sid in all_ids]
    g.vs["Clade"] = [metadata_dict.get(sid, {}).get("Clade") for sid in all_ids]
    g.vs[exposure_col] = [metadata_dict.get(sid, {}).get(exposure_col) for sid in all_ids]

    filtered = pairwise_df[pairwise_df[weight_column] >= minimum_weight]
    edges = list(zip(filtered[id_col_1].map(id_to_index), filtered[id_col_2].map(id_to_index)))
    g.add_edges(edges)

    for col in ["TN93_Distance", "SNP_Distance", "Temporal_Distance", weight_column]:
        if col in filtered.columns:
            g.es[col] = filtered[col].tolist()

    return g


def summarise_cluster_sizes(cluster_results: pd.DataFrame, focus_cluster_ids: set[int]) -> pd.DataFrame:
    """Return all reported cluster sizes and flag the focus clusters."""
    sizes = cluster_results[["cluster_id", "size"]].copy()
    sizes = sizes.sort_values(["size", "cluster_id"], ascending=[False, True]).reset_index(drop=True)
    sizes["rank"] = np.arange(1, len(sizes) + 1, dtype=int)
    sizes["is_focus_cluster"] = sizes["cluster_id"].isin(focus_cluster_ids)
    return sizes


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_linkage_model(config: dict[str, Any], *, mutation_process: str) -> EpiLink:
    """Build a single EpiLink scorer from config inference_baseline."""
    fixed = config["fixed_parameters"]
    inf = config["inference_baseline"]

    inc = gamma_mean_cv_to_shape_scale(float(inf["incubation"]["mean"]), float(inf["incubation"]["cv"]))
    td = gamma_mean_cv_to_shape_scale(float(inf["testing_delay"]["mean"]), float(inf["testing_delay"]["cv"]))

    nhp = NaturalHistoryParameters(
        incubation_shape=float(inc["shape"]),
        incubation_scale=float(inc["scale"]),
        latent_shape=float(fixed.get("latent_shape", 3.38)),
        symptomatic_rate=float(fixed.get("symptomatic_rate", 0.37)),
        symptomatic_shape=float(fixed.get("symptomatic_shape", 1.0)),
        transmission_rate_ratio=float(fixed.get("transmission_rate_ratio", 2.29)),
        testing_delay_shape=float(td["shape"]),
        testing_delay_scale=float(td["scale"]),
        substitution_rate=float(inf.get("substitution_rate", 1e-3)),
        relaxation=float(inf.get("relaxation", 0.33)),
        genome_length=int(fixed.get("genome_length", 29903)),
    )
    rng_seed = int(config.get("experiment", {}).get("seed", 12345))
    profile = InfectiousnessToTransmission(parameters=nhp, rng_seed=rng_seed)
    return EpiLink(
        transmission_profile=profile,
        maximum_depth=int(fixed.get("maximum_depth", 0)),
        mc_samples=int(fixed.get("num_simulations", 10_000)),
        target=tuple(str(v) for v in fixed.get("target", ("ad(0)", "ca(0,0)"))),
        mutation_process=mutation_process,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_path: str | Path = "config.yaml") -> None:
    parser = argparse.ArgumentParser(description="Boston empirical clustering workflow.")
    parser.add_argument("--config", default=config_path, help="Path to YAML configuration.")
    args = parser.parse_args()

    config = load_config(args.config)
    inputs = config["boston"]["inputs"]
    analysis = config["boston"]["analysis"]

    metadata_path = resolve_path(inputs["metadata"])
    pairwise_path = resolve_path(inputs["pairwise_distances"])

    if not metadata_path.exists():
        raise FileNotFoundError(f"Boston metadata file not found: {metadata_path}")
    if not pairwise_path.exists():
        raise FileNotFoundError(f"Boston pairwise distances file not found: {pairwise_path}")

    metadata = pd.read_parquet(metadata_path)
    pair_data = pd.read_parquet(pairwise_path)

    id_col_1 = str(inputs.get("id_col_1", "SeqID1"))
    id_col_2 = str(inputs.get("id_col_2", "SeqID2"))
    date_col = str(inputs.get("date_col", "Date"))
    exposure_col = str(inputs.get("exposure_col", "Exposure"))
    temporal_col = str(inputs.get("temporal_col", "Temporal_Distance"))
    genetic_col = str(inputs.get("genetic_col", "SNP_Distance"))

    missing = {id_col_1, id_col_2, genetic_col} - set(pair_data.columns)
    if missing:
        raise ValueError(f"Missing required columns in pairwise file: {missing}")

    pair_data = add_temporal_distance(
        pairwise_df=pair_data,
        metadata_df=metadata,
        id_col_1=id_col_1,
        id_col_2=id_col_2,
        date_col=date_col,
        out_col=temporal_col,
    )

    weight_column = "EpiLinkStochasticScore"
    stochastic_model = build_linkage_model(config, mutation_process="stochastic")
    pair_data[weight_column] = np.asarray(
        stochastic_model.score_target(
            sample_time_difference=pair_data[temporal_col].values,
            genetic_distance=pair_data[genetic_col].values,
        ),
        dtype=float,
    )

    resolution = float(analysis.get("resolution", 0.3))
    minimum_weight = float(analysis.get("min_edge_weight", 0.0001))
    min_cluster_size = int(analysis.get("min_cluster_size", 2))
    focus_exposures = list(analysis.get("focus_exposures", []))
    rng_seed = int(config.get("experiment", {}).get("seed", 12345))
    n_restarts = int(config["execution"]["evaluate_kwargs"].get("n_restarts", 10))

    g = build_graph(
        pairwise_df=pair_data,
        metadata_df=metadata,
        id_col_1=id_col_1,
        id_col_2=id_col_2,
        exposure_col=exposure_col,
        minimum_weight=minimum_weight,
        weight_column=weight_column,
    )

    partition, _ = run_leiden_partition(
        g,
        weight_column=weight_column,
        resolution=resolution,
        num_restarts=n_restarts,
        rng_seed=rng_seed,
    )

    cluster_results = analyse_partition_composition(
        partition,
        node_attribute=exposure_col,
        edge_attributes=[genetic_col, temporal_col],
        min_cluster_size=min_cluster_size,
    )

    if focus_exposures:
        focus_cols = [f"count::{label}" for label in focus_exposures if f"count::{label}" in cluster_results.columns]
        if not focus_cols:
            raise ValueError(f"Focus exposures not found in cluster summary: {focus_exposures}")
        focus_results = cluster_results[cluster_results[focus_cols].sum(axis=1) > 0]
    else:
        focus_results = cluster_results

    summary = focus_results[[
        "cluster_id",
        "size",
        f"intra_mean_{genetic_col}",
        f"intra_max_{genetic_col}",
        f"intra_mean_{temporal_col}",
        f"intra_max_{temporal_col}",
        f"inter_mean_{genetic_col}",
    ]].copy()
    summary.rename(columns={
        "cluster_id": "Cluster ID",
        "size": "Size",
        f"intra_mean_{genetic_col}": "Intra-SNP (Mean)",
        f"intra_max_{genetic_col}": "Intra-SNP (Max)",
        f"intra_mean_{temporal_col}": "Intra-Time (Mean)",
        f"intra_max_{temporal_col}": "Intra-Time (Max)",
        f"inter_mean_{genetic_col}": "Inter-SNP (Mean)",
    }, inplace=True)

    cluster_sizes = summarise_cluster_sizes(cluster_results, set(focus_results["cluster_id"].astype(int)))

    results_dir = resolve_path(config["outputs"]["directory"])
    results_dir.mkdir(parents=True, exist_ok=True)

    summary.to_parquet(results_dir / "boston_cluster_summary.parquet", index=False)
    focus_results.to_parquet(results_dir / "boston_cluster_composition.parquet", index=False)
    cluster_sizes.to_parquet(results_dir / "boston_cluster_sizes.parquet", index=False)

    logger.info("Saved Boston outputs to: %s", results_dir)


if __name__ == "__main__":
    main()
