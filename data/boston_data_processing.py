import sys

import numpy as np
import pandas as pd


def main():
    metadata = pd.read_csv(
        "raw/boston/MGH_DPH_98percent_772samples_metadata.csv",
        parse_dates=["collection_date"],
    )
    nextclade_result = pd.read_table(
        "raw/boston/MGH_DPH_98percent_772samples_nextclade.tsv", index_col=0
    )

    pairwise = pd.read_csv("raw/boston/MGH_DPH_98percent_772samples_tn93_distances.csv")

    metadata.rename(
        columns={
            "seq_id": "SeqID",
            "collection_date": "Date",
        },
        inplace=True,
    )
    nextclade_result.rename(
        columns={
            "seqName": "SeqID",
            "clade": "Clade",
            "qc.overallStatus": "QC_OverallStatus",
        },
        inplace=True,
    )

    pairwise.rename(
        columns={
            "ID1": "SeqID1",
            "ID2": "SeqID2",
            "Distance": "TN93_Distance",
        },
        inplace=True,
    )

    pairwise["SNP_Distance"] = (
        pairwise["TN93_Distance"] * 29903
    )  # Length of SARS-CoV-2 genome

    seq1_dates = metadata.set_index("SeqID").loc[list(pairwise["SeqID1"]), "Date"]
    seq2_dates = metadata.set_index("SeqID").loc[list(pairwise["SeqID2"]), "Date"]
    temp_diff = (seq2_dates.to_numpy() - seq1_dates.to_numpy()) / np.timedelta64(1, "D")
    temporal_distances = temp_diff.astype(int)
    pairwise["Temporal_Distance"] = temporal_distances

    nextclade_result["substitutions"] = nextclade_result["substitutions"].apply(
        lambda x: x.split(",") if isinstance(x, str) else [x]
    )

    nextclade_result["aaSubstitutions"] = nextclade_result["aaSubstitutions"].apply(
        lambda x: x.split(",") if isinstance(x, str) else [x]
    )

    boston_metadata = metadata[
        [
            "SeqID",
            "Date",
            "CONF_A_EXPOSURE",
            "SNF_A_EXPOSURE",
            "CITY_A_EXPOSURE",
            "BHCHP",
        ]
    ].merge(
        nextclade_result[
            ["SeqID", "Clade", "substitutions", "QC_OverallStatus"]
        ],
        on="SeqID",
        how="inner",
    )

    # Ordered based on estimated TMRCA
    boston_metadata["Mutation"] = boston_metadata["substitutions"].apply(
        lambda substitutions: (
            "C2416T (Conf, BHCHP)"
            if "C2416T" in substitutions
            else (
                "G105T (BHCHP)"
                if "G105T" in substitutions
                else (
                    "G28899T"
                    if "G28899T" in substitutions
                    else (
                        "G3892T (SNF)"
                        if "G3892T" in substitutions
                        else (
                            "C20099T (BHCHP)"
                            if "C20099T" in substitutions
                            else "Minor Lineages"
                        )
                    )
                )
            )
        )
    )

    boston_metadata["Exposure"] = boston_metadata.apply(
        lambda row: (
            "Conference"
            if row["CONF_A_EXPOSURE"] == "YES"
            else (
                "SNF"
                if row["SNF_A_EXPOSURE"] == "YES"
                else (
                    "BHCHP"
                    if row["BHCHP"] == "YES"
                    else ("City" if row["CITY_A_EXPOSURE"] == "YES" else "Unlabeled")
                )
            )
        ),
        axis=1,
    )

    boston_metadata.sort_values(by="Date", inplace=True)

    boston_metadata[[
        "SeqID",
        "Date",
        "QC_OverallStatus",
        "BHCHP",
        "Clade",
        "Mutation",
        "Exposure"
    ]].to_parquet(
        "processed/boston/boston_metadata.parquet", index=False
    )
    pairwise.to_parquet(
        "processed/boston/boston_pairwise_distances.parquet", index=False
    )


if __name__ == "__main__":
    sys.exit(main())
