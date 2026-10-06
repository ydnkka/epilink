"""Read saved backbone evidence and draw descriptive transmission summaries."""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from ..diagnostics.backbone import backbone_settings
from ..provenance import read_json, valid_artifact


def load_backbone_evidence(run):
    """Require a complete, checksummed backbone stage, even in a partial run."""
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    signature = manifest["signature"]
    stage = run / "backbone"
    if "backbone" not in signature.get("implementation", {}) or not (stage / "index.json").exists():
        raise ValueError("No saved backbone diagnostics; run diagnostics --stage backbone")
    settings = backbone_settings(manifest["config"]["diagnostics"].get("backbone"))
    expected = {
        "kind": "diagnostics-backbone-coverage-v1",
        "experiment": manifest["experiment"],
        "datasets": {},
        "implementation": signature["implementation"]["backbone"],
        "settings": settings,
        "tools": {},
    }
    if not valid_artifact(stage, expected):
        raise ValueError("Incomplete or changed backbone stage")
    index = read_json(stage / "index.json")
    if index["status"] != "complete" or len(index["records"]) != 1 or index["datasets"]:
        raise ValueError("Backbone evidence must describe exactly one backbone")
    record = index["records"][0]
    if record["backbone"] != signature["backbone"]:
        raise ValueError("Backbone evidence differs from the pinned diagnostics run")
    artifact = Path(record["artifact"])
    expected = {
        "kind": "diagnostic-backbone-v1",
        "backbone": record["backbone"],
        "implementation": signature["implementation"]["backbone"],
        "settings": settings,
    }
    if not valid_artifact(artifact, expected):
        raise ValueError("Incomplete or changed backbone artefact")
    summary = read_json(artifact / "summary.json")
    tables = {name: pd.read_csv(artifact / f"{name}.csv")
              for name in ("offspring", "concentration", "generations", "components", "bootstrap")}
    return manifest, summary, tables


def draw_backbone(axes, summary, tables):
    """Draw empirical frequencies and saved model values without refitting."""
    ax, concentration_ax, depth_ax = axes
    offspring = tables["offspring"]
    for qualifies, label, colour in (
        (False, "Other cases", "#0072B2"),
        (True, "Superspreading cases", "#CC79A7"),
    ):
        points = offspring.loc[offspring.qualifies_superspreading.eq(qualifies) & offspring.n_cases.gt(0)]
        ax.bar(points.offspring, points.case_fraction, color=colour, width=0.8, label=label)
    ax.plot(offspring.offspring, offspring.poisson_probability, "--", color="0.2", label="Poisson reference")
    if offspring.negative_binomial_probability.notna().any():
        label = "NB (Poisson limit)" if summary["fit_method"] == "poisson_limit" else "Negative binomial fit"
        ax.plot(offspring.offspring, offspring.negative_binomial_probability, color="#D55E00", label=label)
    if summary["n_transmissions"]:
        ax.axvline(summary["poisson_percentile"] - 0.5, color="#CC79A7", ls=":", lw=1)
    ax.set(title="Offspring distribution", xlabel="Secondary infections", ylabel="Case fraction (log scale)",
           yscale="log", ylim=(max(1e-6, 0.4 / summary["n_cases"]), 1.3))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.legend(fontsize=6, loc="upper right")
    k = summary["dispersion_k"]
    fit_label = f"k = {k:.3g}" if k is not None and np.isfinite(k) else summary["fit_method"].replace("_", " ")
    ax.text(0.02, 0.03, f"Mean = {summary['mean_offspring']:.3g}\n{fit_label}",
            transform=ax.transAxes, fontsize=7, va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2})

    curve = tables["concentration"]
    concentration_ax.step(100 * curve.case_fraction, 100 * curve.transmission_fraction,
                          where="post", color="#0072B2")
    concentration_ax.plot([0, 100], [0, 100], ":", color="0.7", label="Equal contribution")
    if summary["n_transmissions"]:
        fraction = 100 * summary["fraction_for_80_percent"]
        concentration_ax.axhline(80, color="0.6", ls="--", lw=0.8)
        concentration_ax.axvline(fraction, color="#D55E00", ls="--", lw=0.8)
        concentration_ax.text(0.97, 0.04, f"80% from {fraction:.1f}% of cases",
                              transform=concentration_ax.transAxes, ha="right", fontsize=7)
    else:
        concentration_ax.text(0.5, 0.5, "No transmissions", ha="center",
                              transform=concentration_ax.transAxes)
    concentration_ax.set(title="Transmission concentration", xlabel="Cases ranked by offspring (%)",
                         ylabel="Cumulative transmissions (%)", xlim=(0, 100), ylim=(0, 105))
    generations = tables["generations"]
    depth_ax.step(generations.depth, generations.n_cases, where="mid", color="#009E73")
    depth_ax.fill_between(generations.depth, generations.n_cases, step="mid", color="#009E73", alpha=0.2)
    depth_ax.set(title="Generation profile", xlabel="Depth from introduction (hops)", ylabel="Cases", ylim=(0, None))
    depth_ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    for axis in axes:
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="0.92")


def backbone_caption(manifest, summary):
    """State the reference population, inclusive boundary and display scope."""
    scope = "truncated smoke backbone" if manifest["config"]["inputs"].get("smoke_cases") is not None else "full selected backbone"
    return (
        f"Transmission structure of the {scope}: {summary['n_cases']:,} cases, "
        f"{summary['n_transmissions']:,} transmission edges and {summary['n_roots']} introduction(s). "
        "Offspring counts include every backbone case, including zero offspring. "
        f"The superspreading rule is offspring >= the {100 * summary['superspreading_quantile']:g}th "
        f"percentile of Poisson(mean offspring = {summary['mean_offspring']:.6g}); "
        f"the percentile cutoff is {summary['poisson_percentile']} secondary infections. "
        "No superspreading events are assigned if the backbone has no transmissions. "
        "The mean is (cases - introductions) / cases, a descriptive backbone reference, "
        "not an estimate of the epidemic reproduction number. The negative-binomial fit is descriptive. "
        "Concentration ranks cases by direct offspring; generations count transmission hops from each root. "
        "These panels describe one fixed backbone, not independent observation-seed replicates."
    )
