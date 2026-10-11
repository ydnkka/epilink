import logging
import random
from time import perf_counter

import igraph as ig
import numpy as np

from ..progress import log_progress, logged_stage

LOG = logging.getLogger(__name__)


def components(graph):
    return np.asarray(graph.connected_components().membership, dtype=np.int32), {}


@logged_stage("Leiden partition", fields=("resolution", "objective", "restarts"))
def leiden(graph, resolution, objective, restarts, seed):
    if graph.ecount() == 0:
        return np.arange(graph.vcount(), dtype=np.int32), {
            "objective": objective,
            "quality": 0.0,
        }
    best_labels, best_quality = None, -np.inf
    qualities = []
    try:
        for restart in range(restarts):
            started = perf_counter()
            LOG.info("Leiden restart=%s/%s seed=%s", restart + 1, restarts, seed + restart)
            ig.set_random_number_generator(random.Random(seed + restart))
            partition = graph.community_leiden(
                weights="weight",
                resolution=resolution,
                objective_function=objective,
                n_iterations=-1,
            )
            # igraph quality is the objective optimized by community_leiden.
            quality = float(partition.quality)
            if not np.isfinite(quality):
                raise ValueError("Leiden returned a non-finite objective")
            qualities.append(quality)
            labels = np.asarray(partition.membership, dtype=np.int32)
            if quality > best_quality:
                best_labels, best_quality = labels, quality
            log_progress(
                LOG, "Leiden restarts", restart + 1, restarts,
                quality=quality, elapsed_seconds=f"{perf_counter() - started:.3f}",
            )
    finally:
        ig.set_random_number_generator(None)
    return best_labels, {
        "objective": objective,
        "quality": best_quality,
        "restart_qualities": qualities,
        "seed": seed,
    }
