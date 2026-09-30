"""Investigation 2: exact tie-aware ranking and relationship enrichment."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import log, save_table
from .truth import RELATIONSHIPS, ENCODING_COLUMNS, canonical_encoding


def weighted_quantile(values, weights, q):
    values, weights = np.asarray(values), np.asarray(weights)
    valid = (weights > 0) & np.isfinite(values)
    if not np.any(valid):
        return np.nan
    values, weights = values[valid], weights[valid]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    index = np.searchsorted(np.cumsum(weights), q * weights.sum(), side="left")
    return float(values[min(index, len(values) - 1)])


def ranking_table(y, score):
    if not np.all(np.isfinite(score)):
        raise ValueError("Scores must be finite.")
    ranks = pd.DataFrame({"score": score, "target": np.asarray(y, dtype=bool)}).groupby(
        "score", sort=True).target.agg(n_pairs="size", n_target="sum").iloc[::-1].reset_index()
    ranks["selected_pairs"] = ranks.n_pairs.cumsum()
    ranks["selected_target"] = ranks.n_target.cumsum()
    ranks["precision"] = ranks.selected_target / ranks.selected_pairs
    n_positive = ranks.n_target.sum()
    ranks["recall"] = ranks.selected_target / n_positive if n_positive else np.nan
    ranks["selected_fraction"] = ranks.selected_pairs / ranks.n_pairs.sum()
    return ranks


def average_precision(ranks):
    total = ranks.n_target.sum()
    return float((ranks.precision * ranks.n_target / total).sum()) if total else np.nan


def thresholds_for_selections(ranks, settings):
    selections = [("fixed_threshold", float(t), float(t)) for t in settings["thresholds"]]
    for fraction in settings["selected_fractions"]:
        row = ranks.loc[ranks.selected_fraction >= fraction].iloc[0]
        selections.append(("top_fraction", float(fraction), float(row.score)))
    if ranks.n_target.sum():
        for recall in settings["recall_levels"]:
            row = ranks.loc[ranks.recall >= recall].iloc[0]
            selections.append(("target_recall", float(recall), float(row.score)))
    return selections


def investigate_scores(pairs, scores, directory, settings):
    log("2/4: score bands, candidate selection, and relationship enrichment")
    labels = pairs.IsRelated.to_numpy(bool)
    total_target = labels.sum()
    prevalence = pairs.relationship.value_counts(sort=False).reindex(RELATIONSHIPS, fill_value=0)
    all_summary, all_selection, all_composition, all_bands, all_relation, all_encodings = [], [], [], [], [], []
    encoding_columns = ENCODING_COLUMNS
    by_m, selected_m = [], []
    for model in settings["models"]:
        values = scores[model].to_numpy()
        ranks = ranking_table(labels, values)
        save_table(directory, f"{model}_precision_recall", ranks)
        all_summary.append({"model": model, "ap": average_precision(ranks),
                            "target_prevalence": labels.mean(),
                            "zero_score_fraction": float(np.mean(values == 0)),
                            "unique_scores": len(ranks)})
        # Aggregate once; all following descriptive operations preserve multiplicities.
        atom_input = canonical_encoding(pairs)
        atom_input[["relationship", "tree_hops"]] = pairs[["relationship", "tree_hops"]]
        atom_input["score"] = values
        atoms = atom_input.groupby(
            ["score", "relationship", "tree_hops", *encoding_columns], observed=True, dropna=False
        ).size().rename("n_pairs").reset_index()
        del atom_input
        save_table(directory, f"{model}_score_relationship_hops", atoms)
        for (ad, ca, m), group in atoms.groupby(["AD", "CA", "M"], observed=True, dropna=False):
            n = group.n_pairs.sum()
            by_m.append({"model": model, "AD": ad, "CA": ca, "M": m, "n_pairs": int(n),
                "mean_CS": float(np.average(group.score, weights=group.n_pairs)),
                "median_CS": weighted_quantile(group.score, group.n_pairs, .5),
                "p90_CS": weighted_quantile(group.score, group.n_pairs, .9)})
        for relation in RELATIONSHIPS:
            group = atoms.loc[atoms.relationship == relation]
            n = group.n_pairs.sum()
            all_relation.append({"model": model, "relationship": relation, "n_pairs": int(n),
                "zero_score_fraction": group.loc[group.score == 0, "n_pairs"].sum() / n if n else np.nan,
                **{f"score_q{int(q * 100):02d}": weighted_quantile(group.score, group.n_pairs, q)
                   for q in (0.1, 0.5, 0.9, 0.99)}})
        bins = np.asarray(settings["scores"]["bins"])
        atoms["score_bin"] = np.searchsorted(bins, atoms.score, side="right") - 1
        for bin_id, group in atoms.groupby("score_bin"):
            if bin_id < 0 or bin_id >= len(bins) - 1:
                raise ValueError("Score bins do not cover all observed scores.")
            counts = group.groupby("relationship", observed=True).n_pairs.sum()
            for relation in RELATIONSHIPS:
                count = int(counts.get(relation, 0))
                all_bands.append({"model": model, "bin_id": bin_id,
                    "lower_inclusive": bins[bin_id], "upper_exclusive": bins[bin_id + 1],
                    "relationship": relation, "n_pairs": count,
                    "band_pairs": int(group.n_pairs.sum()),
                    "proportion": count / group.n_pairs.sum(),
                    "relationship_fraction_in_band": count / prevalence[relation] if prevalence[relation] else np.nan})
        for selection, requested, threshold in thresholds_for_selections(ranks, settings["scores"]):
            group = atoms.loc[atoms.score >= threshold]
            n = int(group.n_pairs.sum())
            counts = group.groupby("relationship", observed=True).n_pairs.sum()
            target = int(counts.get("direct", 0) + counts.get("shared_infector", 0))
            connected = group.loc[group.tree_hops >= 0]
            base = {"model": model, "selection": selection, "requested": requested,
                    "threshold_inclusive": threshold}
            all_selection.append({**base, "selected_pairs": n, "selected_fraction": n / len(pairs),
                "precision": target / n if n else np.nan,
                "recall": target / total_target if total_target else np.nan,
                "target_enrichment": (target / n) / labels.mean() if n and total_target else np.nan,
                "connected_pair_fraction": connected.n_pairs.sum() / n if n else np.nan,
                "median_tree_hops_connected": weighted_quantile(connected.tree_hops, connected.n_pairs, 0.5),
                "p90_tree_hops_connected": weighted_quantile(connected.tree_hops, connected.n_pairs, 0.9),
                "median_M_connected": weighted_quantile(connected.M.to_numpy(dtype=float), connected.n_pairs, .5),
                "p90_M_connected": weighted_quantile(connected.M.to_numpy(dtype=float), connected.n_pairs, .9),
                "pairs_tied_at_cutoff": int(atoms.loc[atoms.score == threshold, "n_pairs"].sum())})
            for relation in RELATIONSHIPS:
                count = int(counts.get(relation, 0))
                proportion = count / n if n else np.nan
                all_composition.append({**base, "relationship": relation, "n_pairs": count,
                    "proportion": proportion, "prevalence": prevalence[relation] / len(pairs),
                    "enrichment": proportion / (prevalence[relation] / len(pairs)) if prevalence[relation] else np.nan,
                    "relationship_recall": count / prevalence[relation] if prevalence[relation] else np.nan})
            encoded = group.groupby(encoding_columns, observed=True, dropna=False).n_pairs.sum().reset_index()
            encoded["proportion"] = encoded.n_pairs / n if n else np.nan
            all_encodings.append(encoded.assign(**base))
            m_counts = group.groupby(["AD", "CA", "M"], observed=True, dropna=False).n_pairs.sum().reset_index()
            m_counts["proportion"] = m_counts.n_pairs / n if n else np.nan
            selected_m.append(m_counts.assign(**base))
    save_table(directory, "ranking_summary", all_summary)
    save_table(directory, "selection_summary", all_selection)
    save_table(directory, "selection_relationships", all_composition)
    save_table(directory, "score_bands", all_bands)
    save_table(directory, "score_by_relationship", all_relation)
    save_table(directory, "selection_epilink_encodings", pd.concat(all_encodings, ignore_index=True))
    save_table(directory, "score_by_M", by_m)
    save_table(directory, "selection_M", pd.concat(selected_m, ignore_index=True))
