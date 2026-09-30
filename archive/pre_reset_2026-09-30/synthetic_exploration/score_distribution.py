"""Weighted P(M | compatibility-score band), computed from saved exact counts.

Replot without simulating or clustering:
python3 -m synthetic_exploration.score_distribution --run <run-directory>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

from .common import save_table, write_json, digest_file
from .scores import weighted_quantile


def distribution_tables(atoms_by_model, width=.05):
    """Count all pairs, not aggregate rows; zero is its own score band.

    Positive intervals are (lower, upper]. Missing M denotes separate trees
    and is excluded from the conditional distribution, with counts recorded.
    """
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Score-band width must be positive and finite.")
    maximum = max(float(frame.score.max()) for frame in atoms_by_model.values())
    n_positive = max(1, int(np.ceil(maximum / width)))
    upper_edges = np.arange(1, n_positive + 1) * width
    bands = [{"score_band": 0, "score_lower": 0., "score_upper": 0., "band_label": "0"}]
    bands += [{"score_band": i+1, "score_lower": i*width, "score_upper": (i+1)*width,
               "band_label": f"({i*width:g}, {(i+1)*width:g}]"} for i in range(n_positive)]
    metadata = {"positive_band_width": width, "positive_band_closure": "(lower, upper]",
                "zero_scores": "separate score_band=0", "bands": bands,
                "normalisation": "P(M | score band, relationship class, connected pair)",
                "pair_weights": "n_pairs from exact score/relationship count tables",
                "undefined_M_pairs": {}}
    distributions, summaries = [], []
    for model, original in atoms_by_model.items():
        if np.any(~np.isfinite(original.score)) or np.any(original.score < 0):
            raise ValueError("Compatibility scores must be finite and nonnegative.")
        metadata["undefined_M_pairs"][model] = int(original.loc[original.M.isna(), "n_pairs"].sum())
        frame = original.loc[original.M.notna()].copy()
        if np.any(frame.M < 0) or np.any(frame.M != np.floor(frame.M)):
            raise ValueError("M must be a nonnegative integer for connected pairs.")
        if np.any(frame.n_pairs <= 0):
            raise ValueError("Pair multiplicities must be positive.")
        frame["M"] = frame.M.astype(int)
        frame["score_band"] = np.where(frame.score == 0, 0,
            np.searchsorted(upper_edges, frame.score.to_numpy(), side="left") + 1)
        for kind, subset in [("pooled", frame), ("AD", frame.loc[frame.AD == 1]), ("CA", frame.loc[frame.CA == 1])]:
            counts = subset.groupby(["score_band", "M"]).n_pairs.sum().rename("n_pairs").reset_index()
            counts["band_pairs"] = counts.groupby("score_band").n_pairs.transform("sum")
            counts["probability"] = counts.n_pairs / counts.band_pairs
            distributions.append(counts.assign(model=model, relationship_class=kind))
            for band in bands:
                group = counts.loc[counts.score_band == band["score_band"]]
                summaries.append({"model": model, "relationship_class": kind, **band,
                    "n_pairs": int(group.n_pairs.sum()),
                    **{f"M_q{int(q*100):02d}": weighted_quantile(group.M, group.n_pairs, q)
                       for q in (.1, .25, .5, .75, .9)},
                    "M0_fraction": float(group.loc[group.M == 0, "probability"].sum()) if len(group) else np.nan})
    return pd.concat(distributions, ignore_index=True), pd.DataFrame(summaries), metadata


def make_M_score_distributions(directory, settings):
    directory = Path(directory)
    models = settings["models"]
    atoms = {model: pd.read_csv(directory / "02_scores" / f"{model}_score_relationship_hops.csv") for model in models}
    width = settings.get("figures", {}).get("M_score_band_width", .05)
    distribution, summary, metadata = distribution_tables(atoms, width)
    save_table(directory / "02_scores", "M_given_score_distribution", distribution)
    save_table(directory / "02_scores", "M_given_score_summary", summary)
    write_json(directory / "02_scores" / "M_given_score_definition.json", metadata)
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    rows = int(np.ceil(len(models) / 2))
    max_m = int(distribution.M.max())
    n_bands = len(metadata["bands"])
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("white")
    for kind, suffix, description in [("pooled", "", "AD and CA pooled"),
                                      ("AD", "_AD", "Ancestor–descendant pairs only"),
                                      ("CA", "_CA", "Common-ancestor pairs only")]:
        fig, axes = plt.subplots(rows, 2, figsize=(13, 4.3*rows + 1), squeeze=False, constrained_layout=True)
        for ax, model in zip(axes.flat, models):
            selected = distribution.loc[(distribution.model == model) & (distribution.relationship_class == kind)]
            stats = summary.loc[(summary.model == model) & (summary.relationship_class == kind)].sort_values("score_band")
            grid = selected.pivot(index="M", columns="score_band", values="probability").reindex(
                index=np.arange(max_m+1), columns=np.arange(n_bands)).fillna(0).to_numpy()
            heat = ax.imshow(np.ma.masked_less_equal(grid, 0), origin="lower", aspect="auto",
                interpolation="nearest", cmap=cmap, norm=LogNorm(vmin=1e-4, vmax=1),
                extent=[-.5, n_bands-.5, -.5, max_m+.5])
            # Grey columns mean no pairs; white cells mean zero probability in an occupied band.
            for empty in stats.loc[stats.n_pairs == 0, "score_band"]:
                ax.axvspan(empty-.5, empty+.5, color="#e4e8ed", zorder=2)
            for q, style in [("M_q10", "--"), ("M_q50", "-"), ("M_q90", "--")]:
                ax.plot(stats.score_band, stats[q], style, color="#ec694b", linewidth=1.6 if q == "M_q50" else 1,
                        marker="o" if q == "M_q50" else "_", markersize=2.6 if q == "M_q50" else 4,
                        path_effects=[pe.Stroke(linewidth=2.7 if q == "M_q50" else 1.8, foreground="white"), pe.Normal()])
            ax.axvline(.5, color="#5d6874", linewidth=1.1)
            total = stats.n_pairs.sum()
            zeros = int(stats.loc[stats.score_band == 0, "n_pairs"].sum())
            ax.set_title(f"{model} · {zeros / total:.1%} at score 0" if total else f"{model} · no pairs", fontsize=12)
            step = max(1, int(np.ceil((n_bands-1) / 5)))
            ticks = [0] + list(range(step, n_bands, step))
            labels = ["0"] + [f"{metadata['bands'][i]['score_lower']:g}–\n{metadata['bands'][i]['score_upper']:g}" for i in ticks[1:]]
            ax.set_xticks(ticks, labels, fontsize=9)
            ax.set(xlabel="Compatibility score (positive-score bands)", ylabel="M · total intermediates",
                   xlim=(-.5, n_bands-.5), ylim=(-.5, max_m+.5))
            ax.spines[["top", "right"]].set_visible(False)
        for ax in list(axes.flat)[len(models):]:
            ax.set_visible(False)
        bar = fig.colorbar(heat, ax=list(axes.flat)[:len(models)], shrink=.76, pad=.02, ticks=[1e-4, 1e-3, .01, .1, 1])
        bar.ax.set_yticklabels(["0.01%", "0.1%", "1%", "10%", "100%"])
        bar.set_label("Fraction of pairs within each score band · logarithmic colour scale")
        fig.legend([Line2D([0], [0], color="#ec694b", lw=1.6), Line2D([0], [0], color="#ec694b", lw=1, ls="--")],
                   ["Median M", "10th and 90th percentiles"], loc="outside lower center", ncols=2, frameon=False)
        fig.suptitle(f"Distribution of M against compatibility score\n{description} · each column is a distribution · grey columns have no pairs", fontsize=15)
        fig.savefig(figures / f"02_M_against_score{suffix}.png", dpi=180, bbox_inches="tight", facecolor="white")
        # SVG keeps text and discrete heatmap cells sharp when exported.
        fig.savefig(figures / f"02_M_against_score{suffix}.svg", bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return distribution, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path(__file__).parent / "outputs/runs/matched/baseline/seed_12345")
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    make_M_score_distributions(args.run, manifest["settings"])
    manifest["M_score_distribution_source_sha256"] = digest_file(__file__)
    write_json(args.run / "manifest.json", manifest)
    print(args.run / "figures/02_M_against_score.png")


if __name__ == "__main__":
    main()
