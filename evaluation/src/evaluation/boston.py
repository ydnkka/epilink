"""Run the empirical Boston clustering workflow."""
from __future__ import annotations

from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd

try:
    from .config import load_config, project_root, resolve_inference_baseline_parameters, resolve_path
    from .leiden import run_leiden_partition
    from .metrics import analyse_partition_composition
    from .models import build_linkage_model
except ImportError:
    from config import load_config, project_root, resolve_inference_baseline_parameters, resolve_path
    from leiden import run_leiden_partition
    from metrics import analyse_partition_composition
    from models import build_linkage_model

METADATA_PATH = "data/processed/boston/boston_metadata.parquet"
PAIRWISE_PATH = "data/processed/boston/boston_pairwise_distances.parquet"
RESULTS_DIR = "results/boston"
ID_COL_1 = "SeqID1"
ID_COL_2 = "SeqID2"
DATE_COL = "Date"
EXPOSURE_COL = "Exposure"
TEMPORAL_COL = "Temporal_Distance"
GENETIC_COL = "SNP_Distance"
MIN_EDGE_WEIGHT = 0.0001
RESOLUTION = 0.3
MIN_CLUSTER_SIZE = 2
FOCUS_EXPOSURES = ("Conference", "SNF")
N_RESTARTS = 10
RNG_SEED = 12345


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
# Entry point
# ---------------------------------------------------------------------------


def main(config_path: str | Path = "config.yaml") -> None:
    config = load_config(config_path)
    metadata_path = resolve_path(METADATA_PATH)
    pairwise_path = resolve_path(PAIRWISE_PATH)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Boston metadata file not found: {metadata_path}")
    if not pairwise_path.exists():
        raise FileNotFoundError(f"Boston pairwise distances file not found: {pairwise_path}")

    metadata = pd.read_parquet(metadata_path)
    pair_data = pd.read_parquet(pairwise_path)

    missing = {ID_COL_1, ID_COL_2, GENETIC_COL} - set(pair_data.columns)
    if missing:
        raise ValueError(f"Missing required columns in pairwise file: {missing}")

    pair_data = add_temporal_distance(
        pairwise_df=pair_data,
        metadata_df=metadata,
        id_col_1=ID_COL_1,
        id_col_2=ID_COL_2,
        date_col=DATE_COL,
        out_col=TEMPORAL_COL,
    )

    weight_column = "EpiLinkStochasticScore"
    inference_parameters = resolve_inference_baseline_parameters(config)
    stochastic_model = build_linkage_model(
        inference_parameters,
        mutation_process="stochastic",
        rng_seed=RNG_SEED,
    )
    pair_data[weight_column] = np.asarray(
        stochastic_model.score_target(
            sample_time_difference=pair_data[TEMPORAL_COL].values,
            genetic_distance=pair_data[GENETIC_COL].values,
        ),
        dtype=float,
    )

    g = build_graph(
        pairwise_df=pair_data,
        metadata_df=metadata,
        id_col_1=ID_COL_1,
        id_col_2=ID_COL_2,
        exposure_col=EXPOSURE_COL,
        minimum_weight=MIN_EDGE_WEIGHT,
        weight_column=weight_column,
    )

    partition, _ = run_leiden_partition(
        g,
        weight_column=weight_column,
        resolution=RESOLUTION,
        num_restarts=N_RESTARTS,
        rng_seed=RNG_SEED,
    )

    cluster_results = analyse_partition_composition(
        partition,
        node_attribute=EXPOSURE_COL,
        edge_attributes=[GENETIC_COL, TEMPORAL_COL],
        min_cluster_size=MIN_CLUSTER_SIZE,
    )

    if FOCUS_EXPOSURES:
        focus_cols = [f"count::{label}" for label in FOCUS_EXPOSURES if f"count::{label}" in cluster_results.columns]
        if not focus_cols:
            raise ValueError(f"Focus exposures not found in cluster summary: {FOCUS_EXPOSURES}")
        focus_results = cluster_results[cluster_results[focus_cols].sum(axis=1) > 0]
    else:
        focus_results = cluster_results

    summary = focus_results[[
        "cluster_id",
        "size",
        f"intra_mean_{GENETIC_COL}",
        f"intra_max_{GENETIC_COL}",
        f"intra_mean_{TEMPORAL_COL}",
        f"intra_max_{TEMPORAL_COL}",
        f"inter_mean_{GENETIC_COL}",
    ]].copy()
    summary.rename(columns={
        "cluster_id": "Cluster ID",
        "size": "Size",
        f"intra_mean_{GENETIC_COL}": "Intra-SNP (Mean)",
        f"intra_max_{GENETIC_COL}": "Intra-SNP (Max)",
        f"intra_mean_{TEMPORAL_COL}": "Intra-Time (Mean)",
        f"intra_max_{TEMPORAL_COL}": "Intra-Time (Max)",
        f"inter_mean_{GENETIC_COL}": "Inter-SNP (Mean)",
    }, inplace=True)

    cluster_sizes = summarise_cluster_sizes(cluster_results, set(focus_results["cluster_id"].astype(int)))

    results_dir = resolve_path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary.to_parquet(results_dir / "cluster_summary.parquet", index=False)
    focus_results.to_parquet(results_dir / "cluster_composition.parquet", index=False)
    cluster_sizes.to_parquet(results_dir / "cluster_sizes.parquet", index=False)


if __name__ == "__main__":
    main()
