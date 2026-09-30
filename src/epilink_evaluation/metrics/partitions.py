"""Every within-cluster pair, plus exact hard-vs-overlapping extended BCubed."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .pairwise import PairTruth


class ReferenceIndex:
    def __init__(self, case_ids, memberships):
        if set(map(str, case_ids)) != set(memberships):
            raise ValueError("Reference case universe differs from observed cases")
        codes, rows, cols = {}, [], []
        for i, case in enumerate(case_ids):
            if not memberships[str(case)]:
                raise ValueError("Empty reference membership")
            for label in memberships[str(case)]:
                rows.append(i)
                cols.append(codes.setdefault(label, len(codes)))
        incidence = csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(len(case_ids), len(codes))
        )
        counts = (incidence @ incidence.T).tocoo()
        self.row, self.col, self.count = counts.row, counts.col, counts.data
        self.neighbors = np.bincount(self.row, minlength=len(case_ids))
        self.n = len(case_ids)

    def score(self, labels):
        _, labels, sizes = np.unique(labels, return_inverse=True, return_counts=True)
        same = labels[self.row] == labels[self.col]
        # Predicted partitions share exactly one label. Retain reference overlap
        # multiplicity, self-pairs and equal case weighting from extended BCubed.
        precision = np.mean(
            np.bincount(self.row[same], minlength=self.n) / sizes[labels]
        )
        recall = np.mean(
            np.bincount(self.row[same], weights=1 / self.count[same], minlength=self.n)
            / self.neighbors
        )
        return {
            "bcubed_precision": float(precision),
            "bcubed_recall": float(recall),
            "bcubed_f1": float(2 * precision * recall / (precision + recall))
            if precision + recall
            else 0.0,
        }


class PartitionEvaluator:
    def __init__(self, observations, cases, truth, reference):
        if not observations.pair_id.equals(truth.pair_id):
            raise ValueError("Partition truth is not aligned")
        self.truth = PairTruth(truth)
        self.a, self.b = observations.a.to_numpy(), observations.b.to_numpy()
        self.n = len(cases)
        self.reference = ReferenceIndex(cases.case_id, reference)

    def evaluate(self, labels):
        if len(labels) != self.n:
            raise ValueError("A partition must cover every sampled case")
        _, labels, sizes = np.unique(labels, return_inverse=True, return_counts=True)
        within = labels[self.a] == labels[self.b]
        summary = self.truth.statistics(within)
        summary.update(self.reference.score(labels))
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
