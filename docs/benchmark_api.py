from __future__ import annotations

import argparse
import statistics
import time

import networkx as nx
import numpy as np

from epilink import (
    EpiLink,
    InfectiousnessToTransmission,
    build_pairwise_case_table,
    simulate_epidemic_dates,
    simulate_genomic_sequences,
)


def _time_call(fn, repeats: int) -> tuple[float, float]:
    durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    return statistics.mean(durations), min(durations)


def _build_tree(node_count: int) -> nx.DiGraph:
    tree = nx.balanced_tree(r=2, h=max(1, int(np.ceil(np.log2(max(2, node_count))))))
    relabeled = nx.DiGraph()
    selected_nodes = list(tree.nodes())[:node_count]
    mapping = {node: f"case-{idx}" for idx, node in enumerate(selected_nodes)}
    for node in selected_nodes:
        relabeled.add_node(mapping[node])
    for parent, child in tree.edges():
        if parent in mapping and child in mapping:
            relabeled.add_edge(mapping[parent], mapping[child])
    return relabeled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark EpiLink scoring and simulation helpers."
    )
    parser.add_argument("--mc-samples", type=int, default=20000)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--genome-length", type=int, default=500)
    parser.add_argument("--tree-nodes", type=int, default=63)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rng-seed", type=int, default=2026)
    args = parser.parse_args()

    profile = InfectiousnessToTransmission(rng_seed=args.rng_seed)

    def build_model() -> EpiLink:
        return EpiLink(
            transmission_profile=profile,
            maximum_depth=2,
            mc_samples=args.mc_samples,
            target=["ad(0)", "ca(0,0)"],
            mutation_process="stochastic",
        )

    init_mean, init_best = _time_call(build_model, args.repeats)
    model = build_model()

    score_pair_mean, score_pair_best = _time_call(
        lambda: model.score_pair(sample_time_difference=3.0, genetic_distance=2.0),
        args.repeats,
    )

    pairwise = model.pairwise_model()
    grid = np.linspace(0.0, 10.0, args.grid_size)
    time_grid = grid[:, np.newaxis]
    distance_grid = grid[np.newaxis, :]
    score_target_mean, score_target_best = _time_call(
        lambda: pairwise(time_grid, distance_grid),
        args.repeats,
    )

    tree = _build_tree(args.tree_nodes)
    simulated_tree = simulate_epidemic_dates(profile, tree, fraction_sampled=1.0)
    sequence_result = simulate_genomic_sequences(
        profile,
        simulated_tree,
        genome_length=args.genome_length,
        return_raw=False,
    )
    pair_table_mean, pair_table_best = _time_call(
        lambda: build_pairwise_case_table(sequence_result.packed, simulated_tree),
        args.repeats,
    )

    print(f"model_init_mean_seconds={init_mean:.6f}")
    print(f"model_init_best_seconds={init_best:.6f}")
    print(f"score_pair_mean_seconds={score_pair_mean:.6f}")
    print(f"score_pair_best_seconds={score_pair_best:.6f}")
    print(f"score_target_grid_shape={time_grid.shape[0]}x{distance_grid.shape[1]}")
    print(f"score_target_mean_seconds={score_target_mean:.6f}")
    print(f"score_target_best_seconds={score_target_best:.6f}")
    print(f"pairwise_table_tree_nodes={args.tree_nodes}")
    print(f"pairwise_table_mean_seconds={pair_table_mean:.6f}")
    print(f"pairwise_table_best_seconds={pair_table_best:.6f}")


if __name__ == "__main__":
    main()
