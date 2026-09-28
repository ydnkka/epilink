"""Build Chapter 3 presentation and overlap summaries from existing Boston inputs.

The original workflow saved EpiLink cluster summaries, not sequence memberships.
--recover-memberships reconstructs the fixed Boston analysis in memory and
requires it to reproduce every saved cluster size and focus-cluster summary.
It does not fit parameters, select thresholds, or overwrite evaluation outputs.
Subsequent runs can use the recovered membership CSV directly.

Example (from the evaluation root, using the PhD Python environment):
    python src/evaluation/chapter3_descriptives.py \
        --output-dir /tmp/ch3-descriptives --recover-memberships
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ch3-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recover_memberships(root: Path, output: Path) -> tuple[pd.DataFrame, dict]:
    from boston import add_temporal_distance, build_graph, summarise_cluster_sizes
    from config import load_config, resolve_inference_baseline_parameters
    from leiden import run_leiden_partition
    from metrics import analyse_partition_composition
    from models import build_linkage_model

    cfg = load_config(root / "config.yaml")
    settings = cfg["workflows"]["boston"]
    s = settings["schema"]
    metadata = pd.read_parquet(root / "data/processed/boston/boston_metadata.parquet")
    pairs = pd.read_parquet(root / "data/processed/boston/boston_pairwise_distances.parquet")
    pairs = add_temporal_distance(
        pairs, metadata, s["metadata_id_col"], s["pairwise_id_col_1"],
        s["pairwise_id_col_2"], s["metadata_date_col"], s["temporal_col"],
    )
    model = build_linkage_model(
        resolve_inference_baseline_parameters(cfg),
        mutation_process="stochastic", rng_seed=int(cfg["rng_seed"]),
    )
    weight = s["weight_column"]
    pairs[weight] = model.score_target(
        sample_time_difference=pairs[s["temporal_col"]].to_numpy(),
        genetic_distance=pairs[s["genetic_col"]].to_numpy(),
    )
    graph = build_graph(
        pairs, metadata, s["metadata_id_col"], s["metadata_date_col"],
        s["metadata_clade_col"], s["pairwise_id_col_1"], s["pairwise_id_col_2"],
        s["exposure_col"], weight, settings["minimum_edge_weight"],
        (s["tn93_col"], s["genetic_col"], s["temporal_col"]),
    )
    partition, modularity = run_leiden_partition(
        graph, weight, settings["resolution"], settings["n_restarts"], cfg["rng_seed"],
    )
    summary = analyse_partition_composition(
        partition, node_attribute=s["exposure_col"],
        edge_attributes=[s["genetic_col"], s["temporal_col"]],
        min_cluster_size=settings["min_cluster_size"],
    )
    focus = summary.loc[summary[[f"count::{x}" for x in settings["focus_exposures"]]].sum(axis=1) > 0].copy()
    saved_focus = pd.read_parquet(root / "results/boston/cluster_composition.parquet")
    saved_sizes = pd.read_parquet(root / "results/boston/cluster_sizes.parquet")
    sizes = summarise_cluster_sizes(summary, set(focus.cluster_id))
    pd.testing.assert_frame_equal(sizes, saved_sizes, check_dtype=False)
    cols = [c for c in saved_focus if c != f'{s["exposure_col"]}_dist']
    pd.testing.assert_frame_equal(
        focus[cols].sort_values("cluster_id").reset_index(drop=True),
        saved_focus[cols].sort_values("cluster_id").reset_index(drop=True),
        check_dtype=False, rtol=1e-10, atol=1e-12,
    )
    membership = pd.DataFrame({
        "SeqID": graph.vs[s["metadata_id_col"]],
        "epilink_cluster": partition.membership,
    })
    assert membership.SeqID.is_unique and len(membership) == 772
    membership_sizes = membership.groupby("epilink_cluster").size()
    non_singletons = membership_sizes[membership_sizes >= settings["min_cluster_size"]]
    assert len(sizes) == len(non_singletons)
    assert sizes["size"].sum() == non_singletons.sum()
    membership.to_csv(output / "boston_epilink_memberships.csv", index=False)
    retained = pairs[weight] >= settings["minimum_edge_weight"]
    audit = {
        "membership_origin": "reconstructed with current configuration and checked against saved Boston summaries",
        "saved_cluster_sizes_reproduced": True,
        "saved_focus_composition_and_edge_summaries_reproduced": True,
        "samples": len(membership), "scored_pairs": len(pairs),
        "non_singleton_clusters": len(sizes), "non_singleton_samples": int(sizes["size"].sum()),
        "singletons": len(membership) - int(sizes["size"].sum()),
        "threshold": settings["minimum_edge_weight"], "resolution": settings["resolution"],
        "modularity": modularity,
        "retained_edges": int(retained.sum()),
        "retained_weight_fraction": float(pairs.loc[retained, weight].sum() / pairs[weight].sum()),
        "settings": settings, "seed": cfg["rng_seed"],
    }
    (output / "boston_membership_recovery.json").write_text(json.dumps(audit, indent=2) + "\n")
    return membership, audit


def compare_memberships(root: Path, members: pd.DataFrame, output: Path) -> dict:
    tc = pd.read_csv(root / "phylo/clusters.tsv", sep="\t")
    assert tc.SequenceName.is_unique and len(tc) == 772
    assert set(members.SeqID) == set(tc.SequenceName)
    joined = members.merge(tc, left_on="SeqID", right_on="SequenceName", validate="one_to_one")
    assert (joined.ClusterNumber == -1).sum() == 230
    assert joined.loc[joined.ClusterNumber != -1, "ClusterNumber"].nunique() == 111
    meta = pd.read_parquet(root / "data/processed/boston/boston_metadata.parquet")
    joined = joined.merge(meta[["SeqID", "Exposure"]], on="SeqID", validate="one_to_one")
    # Each TreeCluster -1 is a distinct singleton, never one pooled cluster.
    joined["tc_group"] = [f"singleton:{sid}" if c == -1 else f"cluster:{c}" for sid, c in zip(joined.SeqID, joined.ClusterNumber)]
    epi = joined.groupby("epilink_cluster").SeqID.agg(set).to_dict()
    tree = joined.groupby("tc_group").SeqID.agg(set).to_dict()
    focus = pd.read_parquet(root / "results/boston/cluster_composition.parquet")
    rows = []
    for eid in focus.cluster_id:
        a = epi[eid]
        for tid, b in tree.items():
            shared = a & b
            if shared:
                rows.append({
                    "epilink_cluster": int(eid), "treecluster_group": tid,
                    "epilink_size": len(a), "treecluster_size": len(b),
                    "shared": len(shared), "union": len(a | b),
                    "epilink_overlap_percent": 100 * len(shared) / len(a),
                    "treecluster_overlap_percent": 100 * len(shared) / len(b),
                    "jaccard": len(shared) / len(a | b),
                })
    overlaps = pd.DataFrame(rows).sort_values(["epilink_cluster", "shared", "jaccard", "treecluster_group"], ascending=[True, False, False, True])
    overlaps.to_csv(output / "boston_cluster_overlaps.csv", index=False)
    best = overlaps.groupby("epilink_cluster", sort=False).head(1)
    best.to_csv(output / "boston_best_cluster_overlaps.csv", index=False)
    table_start = r"""\begin{thesistablebody}{@{}rrrrrrrr@{}}
\toprule
EL ID & EL size & TC ID & TC size & Shared & EL (\%) & TC (\%) & Jaccard \\
\midrule
"""
    table_end = "\n" + r"\bottomrule" + "\n" + r"\end{thesistablebody}" + "\n"
    (output / "boston_overlap_table.tex").write_text(table_start + "\n".join(
        f"{r.epilink_cluster} & {r.epilink_size} & {r.treecluster_group.split(':')[-1]} & {r.treecluster_size} & {r.shared} & {r.epilink_overlap_percent:.1f} & {r.treecluster_overlap_percent:.1f} & {r.jaccard:.3f} " + chr(92) * 2 for r in best.itertuples()
    ) + table_end)
    # Identify named outbreaks by exposure counts, independently of cluster IDs.
    named = []
    for exposure in ("SNF", "Conference"):
        exposure_counts = joined.loc[joined.Exposure == exposure].groupby("epilink_cluster").size()
        eid = max(exposure_counts.index, key=lambda x: (int(exposure_counts[x]), len(epi[x]), -int(x)))
        counts = joined.loc[(joined.ClusterNumber != -1) & (joined.Exposure == exposure)].groupby("tc_group").size()
        tid = max(counts.index, key=lambda x: (int(counts[x]), len(tree[x]), -int(x.split(":")[-1])))
        a, b = epi[eid], tree[tid]
        named.append({"exposure": exposure, "epilink_cluster": eid, "treecluster_group": tid, "epilink_size": len(a), "treecluster_size": len(b), "shared": len(a & b), "epilink_overlap_percent": 100*len(a & b)/len(a), "treecluster_overlap_percent": 100*len(a & b)/len(b), "jaccard": len(a & b)/len(a | b)})
    (output / "boston_named_cluster_overlaps.json").write_text(json.dumps(named, indent=2) + "\n")
    return {"shared_sample_universe": len(joined), "treecluster_clusters": 111, "treecluster_singletons": 230, "named_overlaps": named}


def presentation(root: Path, output: Path) -> None:
    import matplotlib.pyplot as plt
    from figures import make_fig_boston
    from plotting import set_plos_theme

    set_plos_theme()
    fig = make_fig_boston()  # Preserve the existing histogram, bins and observations.
    fig.savefig(output / "boston.pdf", bbox_inches="tight")
    fig.savefig(output / "boston.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "boston.tif", dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recover-memberships", action="store_true")
    parser.add_argument("--memberships", type=Path)
    args = parser.parse_args()
    root, output = args.evaluation_root.resolve(), args.output_dir.resolve()
    protected = [root / "data", root / "phylo", root / "src", root / "results/boston"]
    if output == root or any(output == p or p in output.parents for p in protected):
        raise ValueError("Do not overwrite original inputs, source, or Boston summaries")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "src/evaluation"))
    audit = {}
    if args.recover_memberships:
        members, audit = recover_memberships(root, output)
    elif args.memberships:
        members = pd.read_csv(args.memberships)
        recovery = args.memberships.parent / "boston_membership_recovery.json"
        if recovery.exists():
            audit.update(json.loads(recovery.read_text()))
        audit["membership_sha256"] = digest(args.memberships)
    else:
        parser.error("Provide --memberships or --recover-memberships")
    audit.update(compare_memberships(root, members, output))
    presentation(root, output)
    inputs = [root / p for p in ["config.yaml", "results/boston/cluster_composition.parquet", "results/boston/cluster_sizes.parquet", "phylo/clusters.tsv", "data/processed/boston/boston_metadata.parquet", "data/processed/boston/boston_pairwise_distances.parquet", "src/evaluation/boston.py", "src/evaluation/leiden.py"]]
    audit["source_sha256"] = {str(p): digest(p) for p in inputs}
    (output / "boston_descriptive_validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
