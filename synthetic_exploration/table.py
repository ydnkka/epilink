"""One analysis table, physically partitioned by experiment and model."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from evaluation.specs import EPILINK_SPECS, SCORE_METADATA

from .common import log, write_json

ANALYSIS_COLUMNS = ["CaseID1", "CaseID2", "AD", "CA", "m", "m1", "m2", "M", "GD", "TD", "CS"]


def analysis_chunk(pairs, score, model, start, stop):
    spec = next(s for s in EPILINK_SPECS if s["key"] == model)
    cols = ["CaseID1", "CaseID2", "AD", "CA", "m", "m1", "m2", "M", "pair_id", "IsRelated",
            "relationship", "tree_hops", "lca_index", "lca_steps_a", "lca_steps_b"]
    frame = pairs.iloc[start:stop][cols].copy()
    frame["GD"] = pairs[spec["distance_col"]].iloc[start:stop].to_numpy()
    frame["TD"] = pairs.SamplingDateDistanceDays.iloc[start:stop].to_numpy()
    frame["CS"] = score.iloc[start:stop].to_numpy()
    frame["data_process"] = SCORE_METADATA[model]["data_process"]
    frame["inference_process"] = SCORE_METADATA[model]["inference_process"]
    return frame[ANALYSIS_COLUMNS + [c for c in frame if c not in ANALYSIS_COLUMNS]]


def export_analysis_table(pairs, scores, run, seed, output_root, model_names):
    root = Path(output_root) / "analysis_table"
    for model in model_names:
        directory = root / f"condition={run.condition}" / f"scenario={run.scenario_name}" / f"seed={seed}" / f"model={model}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "pairs.parquet"
        temporary = directory / "pairs.parquet.tmp"
        log(f"Writing unified analysis table: {model}")
        writer = None
        try:
            for start in range(0, len(pairs), 500_000):
                frame = analysis_chunk(pairs, scores[model], model, start, start + 500_000)
                arrow = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, arrow.schema, compression="zstd")
                writer.write_table(arrow)
            if writer is not None:
                writer.close()
                writer = None
            temporary.replace(path)
        finally:
            if writer is not None:
                writer.close()
        # Small example includes the partition identifiers explicitly for easy inspection.
        examples = pairs.groupby(["AD", "CA", "m", "m1", "m2"], dropna=False, observed=True).head(1).index[:100]
        selected = pairs.loc[examples].reset_index(drop=True)
        sample = analysis_chunk(selected, scores.loc[examples, model].reset_index(drop=True), model, 0, len(selected))
        sample = sample.assign(condition=run.condition, scenario=run.scenario_name, seed=seed, model=model)
        preview = Path(output_root) / "runs" / run.condition / run.scenario_name / f"seed_{seed}" / "table_preview"
        preview.mkdir(parents=True, exist_ok=True)
        sample.to_csv(preview / f"{model}.csv", index=False)
    write_json(Path(output_root) / "analysis_table_schema.json", {
        "format": "Hive-partitioned Parquet; read the root directory as one table",
        "row_key": ["condition", "scenario", "seed", "model", "pair_id"],
        "partition_columns": ["condition", "scenario", "seed", "model"],
        "required_columns": ANALYSIS_COLUMNS,
        "GD": "Observed genetic distance, for this row's data_process",
        "TD": "Absolute sampling-date difference, rounded to days as in the original pipeline",
        "CS": "Unmodified summed EpiLink target compatibility, not a probability",
        "m": "Number of AD intermediates; null when AD=0",
        "m1": "CA intermediates on branch to CaseID1; null when CA=0",
        "m2": "CA intermediates on branch to CaseID2; null when CA=0",
        "M": "Total intermediates: m for AD, m1+m2 for CA; null for separate trees",
        "edge_distance": "tree_hops = M+1 for AD and M+2 for CA",
        "case_order": "Unordered pair in tree input order; AD ancestor can be either case",
        "relationship_grouping": "CA(a,b)=CA(b,a). Group on sorted CA depths (min(m1,m2), max(m1,m2)); stored depths stay aligned with the two cases.",
        "null_counts": "Inactive depths are null, never encoded as zero",
    })


def read_analysis_table(output_root=None, *, condition="matched", scenario="baseline", seed=12345, model="ESS", columns=None):
    """Read one model efficiently; use PyArrow dataset scanners for streaming all models."""
    root = Path(output_root) if output_root is not None else Path(__file__).parent / "outputs"
    filters = [(key, "==", value) for key, value in {
        "condition": condition, "scenario": scenario, "seed": seed, "model": model,
    }.items() if value is not None]
    return pd.read_parquet(root / "analysis_table", columns=columns, filters=filters)
