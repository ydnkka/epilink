"""Reproduce synthetic observations and persist an auditable pair table."""
from __future__ import annotations

import json
from pathlib import Path

import epilink
import networkx as nx
import numpy as np
import pandas as pd
from epilink import (InfectiousnessToTransmission, simulate_epidemic_dates,
                     simulate_genomic_sequences, build_pairwise_case_table)
from evaluation.models import build_natural_history_parameters, build_linkage_models
from evaluation.specs import EPILINK_SPECS

from .common import digest_file, fingerprint, write_json, log, versions
from .truth import TreeIndex, prepare_relationships


def prepare_dataset(tree_path, parameters, seed, output_root):
    truth_directory = prepare_relationships(tree_path, output_root)
    package_root = Path(epilink.__file__).parent
    signature = {
        "schema": 2, "tree_sha256": digest_file(tree_path),
        "parameters": parameters, "seed": seed, "versions": versions(),
        "implementation": {str(p): digest_file(p) for p in [
            Path(__file__),
            *package_root.rglob("*.py"),
        ]},
    }
    key = fingerprint(signature)
    directory = Path(output_root) / "datasets" / key[:20]
    manifest = directory / "manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["fingerprint"] != key:
            raise ValueError("Dataset cache fingerprint mismatch.")
        for name, expected in saved["files"].items():
            if digest_file(directory / name) != expected:
                raise ValueError(f"Dataset cache was changed: {directory / name}")
        log(f"Reusing dataset seed={seed}: {directory.name}")
        # Truth annotations can evolve independently of observation simulation.
        saved["truth_directory"] = str(truth_directory.resolve())
        write_json(manifest, saved)
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    log(f"Simulating dates and genomes, seed={seed}")
    tree = nx.read_gml(tree_path)
    profile = InfectiousnessToTransmission(
        parameters=build_natural_history_parameters(parameters), rng_seed=seed)
    populated = simulate_epidemic_dates(
        transmission_profile=profile, tree=tree,
        fraction_sampled=float(parameters.get("fraction_sampled", 1)))
    genomes = simulate_genomic_sequences(
        transmission_profile=profile, tree=populated,
        genome_length=int(parameters.get("synthetic_genome_length", 5000)))
    log(f"Computing observed pairwise distances, seed={seed}")
    pairs = build_pairwise_case_table(genomes["packed"], populated)
    # Stable pair IDs reference the FULL tree, including when sampling is incomplete.
    pairs.insert(0, "pair_id", np.arange(len(pairs), dtype=np.int64))
    pairs = pairs.loc[pairs.BothSampled].copy().reset_index(drop=True)
    if pairs.empty:
        raise ValueError("At least two sampled cases are required.")
    index = TreeIndex(populated)
    cases = pd.DataFrame([
        {"case_id": str(node), "node_index": i, "tree_depth": int(index.depth[i]),
         "root_index": int(index.root[i]),
         "parent_index": int(index.up[0][i]) if populated.in_degree(node) else -1,
         **{k: attributes[k] for k in ("sampled", "exposure_date", "sample_date")}}
        for i, (node, attributes) in enumerate(populated.nodes(data=True))
    ])
    # Relationship and case-ID columns live once in the shared tree table.
    pairs.drop(columns=["CaseA", "CaseB"]).to_parquet(directory / "pairs.parquet", index=False)
    cases.to_parquet(directory / "cases.parquet", index=False)
    write_json(manifest, {
        "fingerprint": key, "signature": signature, "n_pairs": len(pairs),
        "truth_directory": str(truth_directory.resolve()),
        "n_cases": len(cases), "sampled_cases": int(cases.sampled.sum()),
        "components": nx.number_weakly_connected_components(populated),
        "target_prevalence": float(pairs.IsRelated.mean()),
        "files": {name: digest_file(directory / name)
                  for name in ("pairs.parquet", "cases.parquet")},
    })
    log(f"Dataset saved: {directory}")
    return directory


def load_dataset(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    observations = pd.read_parquet(directory / "pairs.parquet")
    truth = pd.read_parquet(Path(manifest["truth_directory"]) / "relationships.parquet")
    # Pair IDs are explicitly joined, never inferred from a cached scores table.
    pairs = observations.merge(truth, on="pair_id", how="left", validate="one_to_one", suffixes=("", "_truth"))
    np.testing.assert_array_equal(pairs.IsRelated.to_numpy(bool), pairs.IsRelated_truth.to_numpy(bool))
    pairs = pairs.drop(columns="IsRelated_truth")
    return pairs, pd.read_parquet(directory / "cases.parquet")


def score_pairs(pairs, parameters, seed, model_names):
    log("Scoring all observed pairs")
    models = build_linkage_models(parameters, rng_seed=seed)
    scores = pd.DataFrame({"pair_id": pairs.pair_id})
    for spec in EPILINK_SPECS:
        if spec["key"] not in model_names:
            continue
        values = models[spec["mutation_process"]].score_target(
            sample_time_difference=pairs.SamplingDateDistanceDays.to_numpy(),
            genetic_distance=pairs[spec["distance_col"]].to_numpy())
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite scores for {spec['key']}")
        scores[spec["key"]] = values
    return scores, models
