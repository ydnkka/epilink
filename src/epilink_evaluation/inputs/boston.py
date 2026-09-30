"""Reproduce preserved Boston input transformations in a new output directory.

The source TN93 file is censored at 0.0005 substitutions/site. This adapter
records that restriction and never imputes missing pairs as zero distance.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ..provenance import complete_artifact, digest_file, valid_artifact


def prepare_boston(data_root, output):
    data_root, output = Path(data_root), Path(output)
    raw = data_root / "raw/boston"
    names = ["MGH_DPH_98percent_772samples_metadata.csv",
             "MGH_DPH_98percent_772samples_nextclade.tsv",
             "MGH_DPH_98percent_772samples_tn93_distances.csv"]
    signature = {"kind": "boston-inputs-v1", "inputs": {name: digest_file(raw / name) for name in names},
                 "implementation": digest_file(__file__), "distance_cutoff_per_site": 0.0005,
                 "reference_length": 29903}
    if valid_artifact(output, signature):
        return output
    metadata = pd.read_csv(raw / names[0], parse_dates=["collection_date"]).rename(
        columns={"seq_id": "case_id", "collection_date": "sample_date"})
    clades = pd.read_csv(raw / names[1], sep="\t").rename(columns={"seqName": "case_id", "clade": "Clade",
                                                                       "qc.overallStatus": "QC_OverallStatus"})
    metadata["case_id"] = metadata.case_id.astype(str)
    clades["case_id"] = clades.case_id.astype(str)
    cases = metadata.merge(clades[["case_id", "Clade", "substitutions", "QC_OverallStatus"]],
                           on="case_id", validate="one_to_one").sort_values(["sample_date", "case_id"])
    cases["Exposure"] = np.select([cases.CONF_A_EXPOSURE.eq("YES"), cases.SNF_A_EXPOSURE.eq("YES"),
                                    cases.BHCHP.eq("YES"), cases.CITY_A_EXPOSURE.eq("YES")],
                                   ["Conference", "SNF", "BHCHP", "City"], default="Unlabeled")
    marker_labels = [("C2416T", "C2416T (Conf, BHCHP)"), ("G105T", "G105T (BHCHP)"),
                     ("G28899T", "G28899T"), ("G3892T", "G3892T (SNF)"), ("C20099T", "C20099T (BHCHP)")]
    def mutation_label(value):
        substitutions = set(value.split(",")) if isinstance(value, str) else set()
        return next((label for marker, label in marker_labels if marker in substitutions), "Minor Lineages")
    cases["Mutation"] = cases.substitutions.map(mutation_label)
    pairs = pd.read_csv(raw / names[2], dtype={"ID1": str, "ID2": str}).rename(
        columns={"ID1": "CaseID1", "ID2": "CaseID2", "Distance": "TN93_distance"})
    pairs = pairs.loc[pairs.CaseID1 != pairs.CaseID2].copy()
    missing = (set(pairs.CaseID1) | set(pairs.CaseID2)) - set(cases.case_id)
    if missing:
        raise ValueError(f"Pair IDs lack metadata: {sorted(missing)[:5]}")
    canonical = np.sort(pairs[["CaseID1", "CaseID2"]].to_numpy(str), axis=1)
    pairs["CaseID1"], pairs["CaseID2"] = canonical[:, 0], canonical[:, 1]
    if pairs.duplicated(["CaseID1", "CaseID2"]).any():
        raise ValueError("Duplicate unordered Boston pairs")
    dates = cases.set_index("case_id").sample_date
    delta = pairs.CaseID2.map(dates) - pairs.CaseID1.map(dates)
    pairs["TD"] = delta.dt.total_seconds().abs() / 86400
    pairs["GD"] = pairs.TN93_distance * 29903
    if not np.isfinite(pairs[["GD", "TD"]].to_numpy()).all():
        raise ValueError("Boston pairs contain missing or non-finite observations")
    output.mkdir(parents=True, exist_ok=True)
    cases.to_parquet(output / "cases.parquet", index=False)
    pairs.to_parquet(output / "observed_pairs.parquet", index=False)
    complete_artifact(output, signature, ["cases.parquet", "observed_pairs.parquet"],
                      n_cases=len(cases), n_observed_pairs=len(pairs),
                      n_all_pairs=len(cases) * (len(cases) - 1) // 2,
                      candidate_universe="TN93-censored; missing pairs are unobserved, not zero")
    return output
