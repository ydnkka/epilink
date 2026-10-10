"""Descriptive Boston evidence and optional TreeCluster membership comparisons.

Exposure flags and an independently constructed phylogenetic partition are
external evidence, not complete transmission links or pairwise ground truth.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from ..provenance import read_json, valid_artifact


def load_treecluster(path, cases):
    """Require the same sample universe; treat every -1 as its own singleton."""
    if path is None:
        return None
    path = Path(path)
    table = pd.read_csv(path, sep="\t", dtype={"SequenceName": str})
    if (
        table.SequenceName.isna().any()
        or table.SequenceName.duplicated().any()
        or table.ClusterNumber.isna().any()
        or set(table.SequenceName) != set(cases.case_id)
    ):
        raise ValueError(
            "TreeCluster and Boston cases must have the same unique sample IDs"
        )
    table["treecluster_group"] = [
        f"singleton:{case_id}" if number == -1 else f"cluster:{number}"
        for case_id, number in zip(table.SequenceName, table.ClusterNumber)
    ]
    return table[["SequenceName", "treecluster_group"]].rename(
        columns={"SequenceName": "case_id"}
    )


def assess_partitions(
    directory,
    cases,
    definitions,
    focus_exposures,
    min_cluster_size,
    comparator,
    n_observed_pairs,
    n_all_pairs,
):
    """Rebuild assessments from completed membership artifacts, including resumed runs."""
    directory = Path(directory)
    output = directory / "assessment"
    output.mkdir(exist_ok=True)
    if "Exposure" not in cases:
        raise ValueError("Boston assessment requires an Exposure column in cases")
    totals = cases.Exposure.value_counts()
    evidence, summary, overlaps, named, partitions, named_groups = (
        [],
        [],
        [],
        [],
        {},
        {},
    )
    if comparator is not None:
        reference = cases[["case_id", "Exposure"]].merge(
            comparator, on="case_id", validate="one_to_one"
        )
        reference_groups = {
            group: set(rows.case_id)
            for group, rows in reference.groupby("treecluster_group")
        }
        reference_named = {}
        for exposure in focus_exposures:
            counts = reference.loc[
                reference.Exposure.eq(exposure)
                & ~reference.treecluster_group.str.startswith("singleton:"),
                "treecluster_group",
            ].value_counts()
            if not counts.empty:
                reference_named[exposure] = min(
                    counts.index,
                    key=lambda group: (
                        -counts[group],
                        -len(reference_groups[group]),
                        group,
                    ),
                )

    for setting_id, definition in definitions.items():
        if definition["kind"] not in ("components", "leiden", "treecluster"):
            continue
        artifact = (
            directory
            / ("trees" if definition["kind"] == "treecluster" else "clusters")
            / setting_id
        )
        manifest = artifact / "manifest.json"
        if not manifest.exists():
            continue
        signature = read_json(manifest).get("signature")
        if signature is None or not valid_artifact(artifact, signature):
            continue
        members = pd.read_parquet(artifact / "memberships.parquet")
        joined = cases.merge(members, on="case_id", validate="one_to_one")
        if len(joined) != len(cases) or joined.cluster_id.isna().any():
            raise ValueError(f"Boston memberships do not cover cases: {setting_id}")
        partitions[setting_id] = joined
        groups = {
            int(group): set(rows.case_id)
            for group, rows in joined.groupby("cluster_id")
        }
        base = {
            "setting_id": setting_id,
            "pipeline": definition["pipeline"],
            "score_name": definition["score_name"],
            "baseline_setting_id": definition["baseline_setting_id"],
        }
        sizes = np.array([len(group) for group in groups.values()])
        summary.append(
            {
                **base,
                "n_clusters": len(groups),
                "n_non_singleton_clusters": int((sizes >= min_cluster_size).sum()),
                "n_singleton_cases": int((sizes == 1).sum()),
                "largest_cluster": int(sizes.max()),
                "n_observed_pairs": n_observed_pairs,
                "n_all_pairs": n_all_pairs,
                "candidate_coverage": n_observed_pairs / n_all_pairs
                if n_all_pairs
                else 0.0,
            }
        )
        for field in ("Exposure", "Clade", "Mutation"):
            if field not in joined:
                continue
            for (group, label), count in (
                joined.groupby(["cluster_id", field]).size().items()
            ):
                evidence.append(
                    {
                        **base,
                        "cluster_id": int(group),
                        "n_cases": len(groups[int(group)]),
                        "evidence_type": field,
                        "evidence_label": label,
                        "n_evidence": int(count),
                        "fraction_in_cluster": count / len(groups[int(group)]),
                    }
                )
        for exposure in focus_exposures:
            counts = joined.loc[
                joined.Exposure.eq(exposure), "cluster_id"
            ].value_counts()
            if counts.empty:
                continue
            focus_groups = [
                int(group)
                for group in counts.index
                if len(groups[int(group)]) >= min_cluster_size
            ]
            if not focus_groups:
                continue
            chosen = min(
                focus_groups,
                key=lambda group: (-counts[group], -len(groups[group]), group),
            )
            named_groups[setting_id, exposure] = chosen
            primary = {
                **base,
                "exposure": exposure,
                "cluster_id": chosen,
                "n_cases": len(groups[chosen]),
                "n_exposure": int(counts[chosen]),
                "exposure_total": int(totals[exposure]),
                "exposure_fraction": counts[chosen] / len(groups[chosen]),
                "exposure_recovery": counts[chosen] / totals[exposure],
            }
            if comparator is not None:
                for group in focus_groups:
                    a = groups[group]
                    for tree_group, b in reference_groups.items():
                        shared = len(a & b)
                        if shared:
                            overlaps.append(
                                {
                                    **base,
                                    "exposure": exposure,
                                    "cluster_id": group,
                                    "treecluster_group": tree_group,
                                    "n_cases": len(a),
                                    "treecluster_size": len(b),
                                    "shared": shared,
                                    "union": len(a | b),
                                    "model_overlap_percent": 100 * shared / len(a),
                                    "treecluster_overlap_percent": 100
                                    * shared
                                    / len(b),
                                    "jaccard": shared / len(a | b),
                                }
                            )
                if exposure in reference_named:
                    tree_group = reference_named[exposure]
                    a, b = groups[chosen], reference_groups[tree_group]
                    shared = len(a & b)
                    primary.update(
                        treecluster_group=tree_group,
                        treecluster_size=len(b),
                        shared=shared,
                        model_overlap_percent=100 * shared / len(a),
                        treecluster_overlap_percent=100 * shared / len(b),
                        jaccard=shared / len(a | b),
                    )
            named.append(primary)

    agreement, named_tree = [], []
    for model_id, model in partitions.items():
        model_def = definitions[model_id]
        if model_def["kind"] == "treecluster":
            continue
        for tree_id, tree in partitions.items():
            tree_def = definitions[tree_id]
            if tree_def["kind"] != "treecluster":
                continue
            common = {
                "setting_id": model_id,
                "pipeline": model_def["pipeline"],
                "score_name": model_def["score_name"],
                "tree_setting_id": tree_id,
                "tree_pipeline": tree_def["pipeline"],
                "tree_kind": tree_def["tree_kind"],
                "baseline_data_process": tree_def["baseline_data_process"],
            }
            agreement.append(
                {
                    **common,
                    "n_cases": len(cases),
                    "adjusted_rand": adjusted_rand_score(
                        model.cluster_id, tree.cluster_id
                    ),
                    "adjusted_mutual_information": adjusted_mutual_info_score(
                        model.cluster_id,
                        tree.cluster_id,
                    ),
                }
            )
            model_groups = {
                int(group): set(rows.case_id)
                for group, rows in model.groupby("cluster_id")
            }
            tree_groups = {
                int(group): set(rows.case_id)
                for group, rows in tree.groupby("cluster_id")
            }
            for exposure in focus_exposures:
                a_id = named_groups.get((model_id, exposure))
                b_id = named_groups.get((tree_id, exposure))
                if a_id is None or b_id is None:
                    continue
                a, b = model_groups[a_id], tree_groups[b_id]
                shared = len(a & b)
                named_tree.append(
                    {
                        **common,
                        "exposure": exposure,
                        "model_cluster_id": a_id,
                        "tree_cluster_id": b_id,
                        "model_size": len(a),
                        "tree_size": len(b),
                        "shared": shared,
                        "model_overlap_percent": 100 * shared / len(a),
                        "tree_overlap_percent": 100 * shared / len(b),
                        "jaccard": shared / len(a | b),
                    }
                )

    columns = {
        "summary.csv": (
            summary,
            [
                "setting_id",
                "pipeline",
                "score_name",
                "baseline_setting_id",
                "n_clusters",
                "n_non_singleton_clusters",
                "n_singleton_cases",
                "largest_cluster",
                "n_observed_pairs",
                "n_all_pairs",
                "candidate_coverage",
            ],
        ),
        "cluster_composition.csv": (
            evidence,
            [
                "setting_id",
                "pipeline",
                "score_name",
                "baseline_setting_id",
                "cluster_id",
                "n_cases",
                "evidence_type",
                "evidence_label",
                "n_evidence",
                "fraction_in_cluster",
            ],
        ),
        "cluster_overlaps.csv": (
            overlaps,
            [
                "setting_id",
                "pipeline",
                "score_name",
                "baseline_setting_id",
                "exposure",
                "cluster_id",
                "treecluster_group",
                "n_cases",
                "treecluster_size",
                "shared",
                "union",
                "model_overlap_percent",
                "treecluster_overlap_percent",
                "jaccard",
            ],
        ),
        "named_cluster_overlaps.csv": (
            named,
            [
                "setting_id",
                "pipeline",
                "score_name",
                "baseline_setting_id",
                "exposure",
                "cluster_id",
                "n_cases",
                "n_exposure",
                "exposure_total",
                "exposure_fraction",
                "exposure_recovery",
                "treecluster_group",
                "treecluster_size",
                "shared",
                "model_overlap_percent",
                "treecluster_overlap_percent",
                "jaccard",
            ],
        ),
        "tree_agreement.csv": (
            agreement,
            [
                "setting_id",
                "pipeline",
                "score_name",
                "tree_setting_id",
                "tree_pipeline",
                "tree_kind",
                "baseline_data_process",
                "n_cases",
                "adjusted_rand",
                "adjusted_mutual_information",
            ],
        ),
        "named_tree_overlaps.csv": (
            named_tree,
            [
                "setting_id",
                "pipeline",
                "score_name",
                "tree_setting_id",
                "tree_pipeline",
                "tree_kind",
                "baseline_data_process",
                "exposure",
                "model_cluster_id",
                "tree_cluster_id",
                "model_size",
                "tree_size",
                "shared",
                "model_overlap_percent",
                "tree_overlap_percent",
                "jaccard",
            ],
        ),
    }
    frames = {}
    for filename, (rows, names) in columns.items():
        frames[filename] = pd.DataFrame(rows).reindex(columns=names)
        frames[filename].to_csv(output / filename, index=False)
    overlap = frames["cluster_overlaps.csv"]
    if not overlap.empty:
        overlap = (
            overlap.sort_values(
                [
                    "setting_id",
                    "exposure",
                    "cluster_id",
                    "shared",
                    "jaccard",
                    "treecluster_group",
                ],
                ascending=[True, True, True, False, False, True],
            )
            .groupby(["setting_id", "exposure", "cluster_id"], sort=False)
            .head(1)
        )
    overlap.to_csv(output / "best_cluster_overlaps.csv", index=False)
    return frames["summary.csv"]
