import numpy as np
import pandas as pd


metadata = pd.read_csv(
    "raw/boston/MGH_DPH_98percent_772samples_metadata.csv",
    parse_dates=["collection_date"]
)

pairwise = pd.read_csv(
    "raw/boston/MGH_DPH_98percent_772samples_tn93_distances.csv"
)

seq1_dates = metadata.set_index("seq_id").loc[list(pairwise["seq_id_1"]), "collection_date"]
seq2_dates = metadata.set_index("seq_id").loc[list(pairwise["seq_id_2"]), "collection_date"]
temp_diff = (seq2_dates.to_numpy() - seq1_dates.to_numpy()) / np.timedelta64(1, "D")
temporal_distances = temp_diff.astype(int)
pairwise["temporal_distance"] = temporal_distances

nextclade_result = pd.read_table(
    "raw/boston/MGH_DPH_98percent_772samples_nextclade.tsv",
    index_col=0
)

nextclade_result.rename(columns={"seqName": "seq_id"}, inplace=True)

nextclade_result["substitutions"] = (
    nextclade_result["substitutions"].apply(
        lambda x: x.split(',') if isinstance(x, str) else [x]
    )
)

nextclade_result["aaSubstitutions"] = (
    nextclade_result["aaSubstitutions"].apply(
        lambda x: x.split(',') if isinstance(x, str) else [x]
    )
)

boston_metadata = pd.merge(
    metadata[["seq_id", "collection_date", "CONF_A_EXPOSURE", "SNF_A_EXPOSURE", "CITY_A_EXPOSURE", "BHCHP"]],
    nextclade_result[["seq_id", "clade", "substitutions", "aaSubstitutions", "qc.overallStatus"]],
    on='seq_id', how='inner'
)

# Ordered based on estimated TMRCA
boston_metadata["mutation"] = boston_metadata["substitutions"].apply(
    lambda substitutions:
    "C2416T (Conf, BHCHP)" if "C2416T" in substitutions else (
        "G105T (BHCHP)" if "G105T" in substitutions else (
            "G28899T" if "G28899T" in substitutions else (
                "G3892T (SNF)" if "G3892T" in substitutions else (
                    "C20099T (BHCHP)" if "C20099T" in substitutions else "Minor Lineages"
                )
            )
        )
    )
)

boston_metadata["label"] = boston_metadata.apply(
    lambda row: "Conference" if row["CONF_A_EXPOSURE"] == "YES" else (
        "SNF" if row["SNF_A_EXPOSURE"] == "YES" else (
            "BHCHP" if row["BHCHP"] == "YES" else (
                "City" if row["CITY_A_EXPOSURE"] == "YES" else "Unlabeled"
            )
        )
    )
    , axis=1)

boston_metadata.sort_values(by="collection_date", inplace=True)

boston_metadata.to_parquet("../../processed/boston/boston_metadata.parquet", index=False)
pairwise.to_parquet("../../processed/boston/boston_pairwise_distances.parquet", index=False)
