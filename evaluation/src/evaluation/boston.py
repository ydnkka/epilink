"""Run the empirical Boston clustering workflow."""
from __future__ import annotations

from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd

try:
    from .config import (
        get_config_value,
        load_config,
        resolve_configured_output_path,
        resolve_configured_path,
        resolve_inference_baseline_parameters,
    )
    from .specs import DEFAULT_SEED
    from .leiden import run_leiden_partition
    from .metrics import analyse_partition_composition
    from .models import build_linkage_model
except ImportError:
    from config import (
        get_config_value,
        load_config,
        resolve_configured_output_path,
        resolve_configured_path,
        resolve_inference_baseline_parameters,
    )
    from specs import DEFAULT_SEED
    from leiden import run_leiden_partition
    from metrics import analyse_partition_composition
    from models import build_linkage_model


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def add_temporal_distance(
        pairwise_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        metadata_id_col: str,
        id_col_1: str,
        id_col_2: str,
        date_col: str,
        out_col: str,
) -> pd.DataFrame:
    date_map = metadata_df.set_index(metadata_id_col)[date_col]
    d1 = pd.to_datetime(pairwise_df[id_col_1].map(date_map))
    d2 = pd.to_datetime(pairwise_df[id_col_2].map(date_map))
    pairwise_df[out_col] = (d1 - d2).abs().dt.days.astype(int)
    return pairwise_df


def build_graph(
        pairwise_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        metadata_id_col: str,
        metadata_date_col: str,
        metadata_clade_col: str,
        id_col_1: str,
        id_col_2: str,
        exposure_col: str,
        weight_column: str,
        minimum_weight: float = 0.0001,
        edge_attribute_columns: tuple[str, ...] = (),
) -> ig.Graph:
    """Build an igraph Graph with vertex metadata and weighted edges."""
    all_ids = pd.unique(pairwise_df[[id_col_1, id_col_2]].values.ravel())
    id_to_index = {seq_id: i for i, seq_id in enumerate(all_ids)}
    metadata_dict = metadata_df.set_index(metadata_id_col).to_dict(orient="index")

    g = ig.Graph(n=len(all_ids))
    g.vs[metadata_id_col] = all_ids.tolist()
    g.vs[metadata_date_col] = [metadata_dict.get(sid, {}).get(metadata_date_col) for sid in all_ids]
    g.vs[metadata_clade_col] = [metadata_dict.get(sid, {}).get(metadata_clade_col) for sid in all_ids]
    g.vs[exposure_col] = [metadata_dict.get(sid, {}).get(exposure_col) for sid in all_ids]

    filtered = pairwise_df[pairwise_df[weight_column] >= minimum_weight]
    edges = list(zip(filtered[id_col_1].map(id_to_index), filtered[id_col_2].map(id_to_index)))
    g.add_edges(edges)

    for col in (*edge_attribute_columns, weight_column):
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
# Entry point
# ---------------------------------------------------------------------------


def main(config_path: str | Path = "config.yaml") -> None:
    config = load_config(config_path)
    workflow = get_config_value(config, "workflows.boston", default={})
    schema = get_config_value(config, "workflows.boston.schema", default={})
    metadata_path = resolve_configured_path(config, "paths.boston.metadata_path")
    pairwise_path = resolve_configured_path(config, "paths.boston.pairwise_path")
    metadata_id_col = str(schema.get("metadata_id_col"))
    metadata_date_col = str(schema.get("metadata_date_col"))
    metadata_clade_col = str(schema.get("metadata_clade_col"))
    exposure_col = str(schema.get("exposure_col"))
    pairwise_id_col_1 = str(schema.get("pairwise_id_col_1"))
    pairwise_id_col_2 = str(schema.get("pairwise_id_col_2"))
    temporal_col = str(schema.get("temporal_col"))
    genetic_col = str(schema.get("genetic_col"))
    tn93_col = str(schema.get("tn93_col"))
    weight_column = str(schema.get("weight_column"))
    min_edge_weight = float(workflow.get("minimum_edge_weight"))
    resolution = float(workflow.get("resolution"))
    min_cluster_size = int(workflow.get("min_cluster_size"))
    focus_exposures_raw = workflow.get("focus_exposures")
    focus_exposures = tuple(str(value) for value in focus_exposures_raw)
    n_restarts = int(workflow.get("n_restarts"))
    rng_seed = int(get_config_value(config, "rng_seed", default=DEFAULT_SEED))

    if not metadata_path.exists():
        raise FileNotFoundError(f"Boston metadata file not found: {metadata_path}")
    if not pairwise_path.exists():
        raise FileNotFoundError(f"Boston pairwise distances file not found: {pairwise_path}")

    metadata = pd.read_parquet(metadata_path)
    pair_data = pd.read_parquet(pairwise_path)

    missing_pairwise = {pairwise_id_col_1, pairwise_id_col_2, genetic_col} - set(pair_data.columns)
    if missing_pairwise:
        raise ValueError(f"Missing required columns in pairwise file: {missing_pairwise}")

    missing_metadata = {metadata_id_col, metadata_date_col, exposure_col} - set(metadata.columns)
    if missing_metadata:
        raise ValueError(f"Missing required columns in metadata file: {missing_metadata}")

    pair_data = add_temporal_distance(
        pairwise_df=pair_data,
        metadata_df=metadata,
        metadata_id_col=metadata_id_col,
        id_col_1=pairwise_id_col_1,
        id_col_2=pairwise_id_col_2,
        date_col=metadata_date_col,
        out_col=temporal_col,
    )

    inference_parameters = resolve_inference_baseline_parameters(config)
    stochastic_model = build_linkage_model(
        inference_parameters,
        mutation_process="stochastic",
        rng_seed=rng_seed,
    )
    pair_data[weight_column] = np.asarray(
        stochastic_model.score_target(
            sample_time_difference=pair_data[temporal_col].values,
            genetic_distance=pair_data[genetic_col].values,
        ),
        dtype=float,
    )

    g = build_graph(
        pairwise_df=pair_data,
        metadata_df=metadata,
        metadata_id_col=metadata_id_col,
        metadata_date_col=metadata_date_col,
        metadata_clade_col=metadata_clade_col,
        id_col_1=pairwise_id_col_1,
        id_col_2=pairwise_id_col_2,
        exposure_col=exposure_col,
        minimum_weight=min_edge_weight,
        edge_attribute_columns=(tn93_col, genetic_col, temporal_col),
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

    results_dir = resolve_configured_output_path(config, "outputs.boston.directory")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary.to_parquet(results_dir / "cluster_summary.parquet", index=False)
    focus_results.to_parquet(results_dir / "cluster_composition.parquet", index=False)
    cluster_sizes.to_parquet(results_dir / "cluster_sizes.parquet", index=False)


if __name__ == "__main__":
    main()
