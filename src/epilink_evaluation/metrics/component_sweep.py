"""Incremental connected components with exact all-within-pair truth counts."""

import numpy as np
import pandas as pd

from .pairwise import CATEGORIES, count_statistics


class ComponentSweep:
    """Add whole score ties in strict-to-permissive order, counting each pair once.

    The baseline's complete, canonically ordered pair universe permits compact
    triangular truth lookup. Cross-component pairs become within pairs only at
    their first merge; no threshold rescans the complete pair universe.
    """

    def __init__(self, evaluator, values, higher_is_better):
        self.truth, self.n = evaluator.truth, evaluator.n
        self.a, self.b = evaluator.a, evaluator.b
        a, b = np.triu_indices(self.n, 1)
        if not (np.array_equal(a, self.a) and np.array_equal(b, self.b)):
            raise ValueError(
                "Component sweeps require canonical complete sampled pairs"
            )
        values = np.asarray(values, dtype=float)
        if (
            self.n < 1
            or len(values) != len(self.a)
            or not np.isfinite(values).all()
            or np.any(values < 0)
        ):
            raise ValueError("Invalid component sweep scores")
        self.higher_is_better = higher_is_better
        oriented = -values if higher_is_better else values
        self.order = np.argsort(oriented, kind="stable")
        self.values = oriented[self.order]
        self.categories = np.empty(len(values), dtype=np.uint8)
        for i, name in enumerate(CATEGORIES):
            self.categories[self.truth.categories[name]] = i
        self.parent = np.arange(self.n)
        self.members = [np.array([i], dtype=np.int32) for i in range(self.n)]
        self.counts = np.zeros((self.n, len(CATEGORIES)), dtype=np.int64)
        self.total = np.zeros(len(CATEGORIES), dtype=np.int64)
        self.position, self.revision, self.n_clusters = 0, 0, self.n

    def _root(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def advance(self, threshold):
        """Return whether the partition changed; None is the initial empty graph."""
        if threshold is None:
            target = 0
        else:
            oriented = -threshold if self.higher_is_better else threshold
            target = int(np.searchsorted(self.values, oriented, side="right"))
        if target < self.position:
            raise ValueError("Component cutoffs must run from strict to permissive")
        revision = self.revision
        for edge in self.order[self.position : target]:
            if self.n_clusters == 1:
                break
            x, y = self._root(self.a[edge]), self._root(self.b[edge])
            if x == y:
                continue
            if len(self.members[x]) < len(self.members[y]):
                x, y = y, x
            left, right = self.members[x][:, None], self.members[y]
            low, high = np.minimum(left, right), np.maximum(left, right)
            indices = (
                low.astype(np.int64) * (2 * self.n - low - 1) // 2 + high - low - 1
            )
            added = np.bincount(
                self.categories[indices].ravel(), minlength=len(CATEGORIES)
            )
            self.total += added
            self.counts[x] += self.counts[y] + added
            self.parent[y] = x
            self.members[x] = np.concatenate((self.members[x], self.members[y]))
            self.members[y] = np.empty(0, dtype=np.int32)
            self.n_clusters -= 1
            self.revision += 1
        self.position = target
        return self.revision != revision

    def evaluate(self):
        """Produce the same metrics and canonical tables as PartitionEvaluator."""
        roots = sorted(
            (i for i in range(self.n) if self.parent[i] == i),
            key=lambda i: self.members[i].min(),
        )
        labels = np.empty(self.n, dtype=np.int32)
        sizes = np.array([len(self.members[i]) for i in roots], dtype=np.int64)
        for label, root in enumerate(roots):
            labels[self.members[root]] = label
        pairs = sizes * (sizes - 1) // 2
        clusters = pd.DataFrame(
            {
                "cluster_id": np.arange(len(roots)),
                "n_cases": sizes,
                "within_pairs": pairs,
            }
        )
        for j, name in enumerate(CATEGORIES):
            clusters[f"n_{name}"] = self.counts[roots, j]
        positive = {"M0": (0, 4), "Mle1": (0, 4, 1, 5), "Mle2": (0, 4, 1, 5, 2, 6, 7)}
        for name, indices in positive.items():
            tp = self.counts[roots][:, indices].sum(axis=1)
            clusters[f"{name}_precision"] = np.divide(
                tp, pairs, out=np.full(len(pairs), np.nan), where=pairs > 0
            )
        summary = count_statistics(
            dict(zip(CATEGORIES, map(int, self.total))), self.truth
        )
        summary.update(
            n_cases=self.n,
            n_clusters=len(sizes),
            n_singletons=int((sizes == 1).sum()),
            singleton_fraction=float((sizes == 1).sum() / self.n),
            largest_cluster=int(sizes.max()),
            largest_cluster_fraction=float(sizes.max() / self.n),
            within_pairs=int(self.total.sum()),
            cluster_mean_M0_precision=float(
                clusters.loc[sizes > 1, "M0_precision"].mean()
            ),
        )
        return labels, summary, clusters
