from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..provenance import read_json


def read_table(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def markdown_table(frame):
    if frame.empty:
        return "No completed results."

    def format_value(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.4g}" if np.isfinite(value) else "undefined"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(format_value(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def figures(directory, definitions, development):
    output = directory / "figures"
    output.mkdir(exist_ok=True)
    saved = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=150)
        plt.close(fig)
        saved.append(f"figures/{name}.png")

    curves = []
    for path in sorted(
        (directory / "development").glob("seed_*/pairwise/precision_recall.parquet")
    ):
        frame = pd.read_parquet(path)
        if not frame.empty:
            curves.append(frame)
    if curves:
        curve = pd.concat(curves, ignore_index=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for axis, process in zip(axes, ("deterministic", "stochastic")):
            for i, (score, group) in enumerate(
                curve.loc[curve.data_process == process].groupby("score_name")
            ):
                for j, (_, replicate) in enumerate(group.groupby("seed")):
                    axis.step(
                        replicate.M0_recall,
                        replicate.M0_precision,
                        where="post",
                        alpha=0.65,
                        color=f"C{i}",
                        label=score if j == 0 else None,
                    )
            axis.set(
                title=f"{process}: each development realization",
                xlabel="M=0 recall",
                ylabel="M=0 precision",
                xlim=(0, 1),
                ylim=(0, 1),
            )
            axis.legend(fontsize=8)
        save(fig, "pairwise_precision_recall")
    if development.empty:
        return saved
    summary = development.groupby(["pipeline", "setting_id"], as_index=False)[
        [
            "M0_precision",
            "M0_recall",
            "M0_f1",
            "Mge3_contamination",
            "singleton_fraction",
            "largest_cluster_fraction",
        ]
    ].mean()
    for key in ("kind", "threshold", "resolution", "method", "tree_kind"):
        summary[key] = summary.setting_id.map(
            lambda identifier: definitions[identifier].get(key)
        )
    components = summary.loc[summary.kind == "components"]
    if not components.empty:
        pipelines = sorted(components.pipeline.unique())
        fig, axes = plt.subplots(
            len(pipelines), 1, figsize=(9, 2.5 * len(pipelines)), squeeze=False
        )
        for axis, pipeline in zip(axes.flat, pipelines):
            group = (
                components.loc[components.pipeline == pipeline]
                .dropna(subset=["threshold"])
                .sort_values("threshold")
            )
            for metric in (
                "M0_precision",
                "M0_recall",
                "Mge3_contamination",
                "largest_cluster_fraction",
            ):
                axis.plot(group.threshold, group[metric], ".-", label=metric)
            axis.set(
                title=pipeline,
                xlabel="Native threshold (genetic ≤; score ≥)",
                ylim=(0, 1),
            )
            axis.legend(fontsize=7, ncol=2)
        save(fig, "components_thresholds")
    leiden = summary.loc[(summary.kind == "leiden") & summary.threshold.notna()]
    if not leiden.empty:
        pipelines = sorted(leiden.pipeline.unique())
        fig, axes = plt.subplots(
            len(pipelines), 2, figsize=(12, 2.7 * len(pipelines)), squeeze=False
        )
        for row, pipeline in enumerate(pipelines):
            group = leiden.loc[leiden.pipeline == pipeline]
            for column, metric in enumerate(("M0_f1", "Mge3_contamination")):
                table = group.pivot(
                    index="resolution", columns="threshold", values=metric
                ).sort_index()
                axis = axes[row, column]
                image = axis.imshow(
                    table.to_numpy(float), origin="lower", aspect="auto", vmin=0, vmax=1
                )
                axis.set(
                    title=f"{pipeline}: {metric}",
                    xlabel="Graph threshold",
                    ylabel="Resolution",
                )
                axis.set_xticks(
                    range(len(table.columns)),
                    [f"{x:g}" for x in table.columns],
                    rotation=45,
                )
                axis.set_yticks(
                    range(len(table.index)), [f"{x:g}" for x in table.index]
                )
                fig.colorbar(image, ax=axis)
        save(fig, "leiden_threshold_resolution")
    tc = summary.loc[summary.kind == "treecluster"]
    if not tc.empty:
        pipelines = sorted(tc.pipeline.unique())
        fig, axes = plt.subplots(
            len(pipelines), 1, figsize=(9, 3 * len(pipelines)), squeeze=False
        )
        for axis, pipeline in zip(axes.flat, pipelines):
            group = tc.loc[tc.pipeline == pipeline]
            for method, subset in group.groupby("method"):
                subset = subset.sort_values("threshold")
                axis.plot(subset.threshold, subset.M0_f1, ".-", label=method)
            units = (
                "days" if group.tree_kind.iloc[0] == "dated" else "substitutions/site"
            )
            axis.set(
                title=pipeline,
                xlabel=f"Tree threshold ({units})",
                ylabel="M=0 F1",
                ylim=(0, 1),
            )
            axis.legend(fontsize=8)
        save(fig, "treecluster_thresholds")
    fig, axis = plt.subplots(figsize=(11, 7))
    for pipeline, group in summary.groupby("pipeline"):
        if not pipeline.startswith("pairwise/"):
            axis.scatter(
                group.M0_recall, group.M0_precision, s=12, alpha=0.5, label=pipeline
            )
    axis.set(
        xlabel="Mean development M=0 pair recall",
        ylabel="Mean development M=0 pair precision",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Clustering operating trade-offs",
    )
    axis.legend(fontsize=6, loc="center left", bbox_to_anchor=(1, 0.5))
    save(fig, "cluster_tradeoffs")
    return saved


def render_report(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    definitions = read_json(directory / "settings.json")
    config = manifest["config"]
    development = read_table(directory / "development/metrics.csv")
    # Pairwise-only reports do not have these cluster-specific columns yet.
    for column in ("singleton_fraction", "largest_cluster_fraction"):
        if not development.empty and column not in development:
            development[column] = np.nan
    image_paths = figures(directory, definitions, development)
    title = (
        "Synthetic baseline — smoke validation"
        if config["inputs"].get("smoke_cases")
        else "Synthetic baseline"
    )
    text = [
        f"# {title}",
        "Primary endpoint: M=0 (direct transmission or shared infector). Secondary: M≤1 and M≤2.",
        "Realizations share a fixed transmission backbone. Between-realization variation is conditional on that backbone.",
        f"Last command: `{manifest['requested_stage']}`; command status: **{manifest['status']}**.",
    ]
    body = [f"<h1>{html.escape(title)}</h1>"]
    body.extend(f"<p>{html.escape(paragraph)}</p>" for paragraph in text[1:])

    def section(heading, frame=None, explanation=None):
        text.append(f"## {heading}")
        body.append(f"<h2>{html.escape(heading)}</h2>")
        if explanation:
            text.append(explanation)
            body.append(f"<p>{html.escape(explanation)}</p>")
        if frame is not None:
            text.append(markdown_table(frame))
            body.append(
                frame.to_html(index=False, float_format=lambda v: f"{v:.4g}")
                if not frame.empty
                else "<p>No completed results.</p>"
            )

    coverage = []
    for split in ("development", "evaluation"):
        for seed in config["splits"][split]:
            path = directory / split / f"seed_{seed}" / "clusters/status.json"
            status = (
                read_json(path)
                if path.exists()
                else {"status": "not run", "completed": 0, "configured": None}
            )
            pairwise = directory / split / f"seed_{seed}" / "pairwise/manifest.json"
            coverage.append(
                {
                    "split": split,
                    "seed": seed,
                    "pairwise": "complete" if pairwise.exists() else "not run",
                    "clustering": status["status"],
                    "completed": status["completed"],
                    "configured": status["configured"],
                }
            )
    section(
        "Comparison completeness",
        pd.DataFrame(coverage),
        "Partial or failed tree comparisons are visible in each seed's clusters/status.json. Complete command status refers only to the requested stage.",
    )
    rankings = []
    for path in sorted(
        (directory / "development").glob("seed_*/pairwise/rankings.csv")
    ):
        rankings.append(read_table(path))
    if rankings:
        frame = pd.concat(rankings, ignore_index=True)
        columns = [
            name
            for name in ("M0_AP", "Mle1_AP", "Mle2_AP", "M0_prevalence", "brier_score")
            if name in frame
        ]
        section(
            "Development pairwise comparison",
            frame.groupby(["data_process", "score_name"])[columns].mean().reset_index(),
            "Equal-realization means; full per-seed rankings, tie-aware curves, budget composition and logistic calibration are saved alongside this report.",
        )
    frozen_path = directory / "selection/operating_points.json"
    if frozen_path.exists():
        frozen = read_json(frozen_path)
        points = pd.DataFrame(
            [
                {
                    key: value
                    for key, value in point.items()
                    if key not in ("definition", "rule")
                }
                for point in frozen["operating_points"]
            ]
        )
        section(
            "Frozen operating decisions",
            points,
            "Criteria were applied to development data. Definitions and the evidence fingerprint are in selection/operating_points.json.",
        )
    else:
        section(
            "Operating decisions",
            explanation="Settings have not been frozen. Inspect development trade-offs and choose the scientific operating criteria before running select.",
        )
    evaluation = read_table(directory / "evaluation/operating_results.csv")
    if not evaluation.empty:
        metrics = [
            "M0_precision",
            "M0_recall",
            "M0_f1",
            "Mge3_contamination",
            "bcubed_f1",
        ]
        present = [column for column in metrics if column in evaluation]
        summary = evaluation.groupby(["criterion", "pipeline"])[present].agg(
            ["mean", "std", "min", "max"]
        )
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary.reset_index().to_csv(
            directory / "evaluation/operating_summary.csv", index=False
        )
        section(
            "Held-out fixed-setting performance",
            summary.reset_index(),
            "Settings are replayed unchanged. SD and range summarize observation realizations, not independent epidemics or independent pair replicates.",
        )
    else:
        section(
            "Held-out performance",
            explanation="Evaluation has not completed; no held-out performance claim is available.",
        )
    section(
        "Scientific interpretation",
        explanation=(
            "For M=0, all M>0 pairs are false positives. M≥3 is a separate distant-contamination measure. "
            "Cluster metrics use every co-clustered unordered pair; zero within-pair precision is undefined for all-singleton partitions. "
            "Compatibility is not a calibrated probability. See the protocol for distance units, graph weights, roots and uncertainty scope."
        ),
    )
    section("Development figures")
    for path in image_paths:
        text.append(f"![{Path(path).stem}]({path})")
        body.append(
            f'<figure><img src="{html.escape(path)}" alt="{html.escape(Path(path).stem)}"></figure>'
        )
    (directory / "report.md").write_text("\n\n".join(text) + "\n")
    (directory / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>'
        + html.escape(title)
        + "</title><style>body{font:15px system-ui;max-width:1400px;margin:2em auto;padding:1em}"
        "table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}td,th{padding:.4em;border:1px solid #ccc}"
        "img{max-width:100%}figure{margin:1em 0}</style><body>"
        + "\n".join(body)
        + "</body></html>\n"
    )
