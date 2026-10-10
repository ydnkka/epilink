"""Evaluate every within-cluster pair and summarize cluster structure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pairwise import PairTruth


class PartitionEvaluator:
    def __init__(self, observations, cases, truth):
        if not observations.pair_id.equals(truth.pair_id):
            raise ValueError("Partition truth is not aligned")
        self.truth = PairTruth(truth)
        self.a, self.b = observations.a.to_numpy(), observations.b.to_numpy()
        self.n = len(cases)

    def evaluate(self, labels):
        if len(labels) != self.n:
            raise ValueError("A partition must cover every sampled case")
        _, labels, sizes = np.unique(labels, return_inverse=True, return_counts=True)
        within = labels[self.a] == labels[self.b]
        summary = self.truth.statistics(within)
        summary.update(
            {
                "n_cases": self.n,
                "n_clusters": len(sizes),
                "n_singletons": int(np.sum(sizes == 1)),
                "singleton_fraction": float(np.sum(sizes == 1) / self.n),
                "largest_cluster": int(sizes.max()),
                "largest_cluster_fraction": float(sizes.max() / self.n),
                "within_pairs": int(within.sum()),
            }
        )
        cluster_pairs = sizes.astype(np.int64) * (sizes - 1) // 2
        clusters = pd.DataFrame(
            {
                "cluster_id": np.arange(len(sizes)),
                "n_cases": sizes,
                "within_pairs": cluster_pairs,
            }
        )
        for name, mask in self.truth.categories.items():
            clusters[f"n_{name}"] = np.bincount(
                labels[self.a[within & mask]], minlength=len(sizes)
            )
        for name, mask in self.truth.masks.items():
            positive = np.bincount(labels[self.a[within & mask]], minlength=len(sizes))
            clusters[f"{name}_precision"] = np.divide(
                positive,
                cluster_pairs,
                out=np.full(len(sizes), np.nan),
                where=cluster_pairs > 0,
            )
        summary["cluster_mean_M0_precision"] = float(
            clusters.loc[sizes > 1, "M0_precision"].mean()
        )
        return summary, clusters
