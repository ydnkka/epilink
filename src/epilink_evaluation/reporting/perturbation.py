"""Saved-table reports for the four EpiLink clustering perturbation arms."""

import html
from pathlib import Path

import pandas as pd

from ..provenance import read_json
from .report import markdown_table, read_table


def render_report(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    reference = read_json(directory / "reference.json")
    config = manifest["config"]
    title = "EpiLink clustering perturbation" + (" — smoke validation" if config["smoke_mode"] else "")
    paragraphs = [
        f"Status: {manifest['status']}. Reference baseline: {reference['run_directory']}.",
        f"Cases: {manifest['n_cases']} (reference backbone: {reference['n_cases']}). Development seeds: {config['development_seeds']}. Evaluation seeds: {config['seeds']}.",
        "Four arms cross baseline/matched EpiLink inference with baseline/updated score-weighted Leiden resolutions. Every observed pair is included, with its original score as weight, including zero-weight edges. Updated resolutions are selected on separate fresh development observations using the baseline grid and criterion; algorithm settings stay fixed.",
        "All arms evaluate the same paired observations. Deltas are perturbed minus fresh unperturbed controls in the same arm, seed and pipeline. Updated settings may differ between a scenario and its control; both setting IDs are retained in results_deltas.csv.",
        "Negative F1 deltas indicate lower recovery; positive M≥3 contamination deltas indicate more distant within-cluster pairs. SD/range describe observation variation conditional on one backbone.",
    ]
    if config["smoke_mode"]:
        paragraphs.append("This reduced-backbone study validates the pipeline; full-backbone conclusions require the full study.")
    text = [f"# {title}", *paragraphs]
    body = [f"<h1>{html.escape(title)}</h1>", *[f"<p>{html.escape(p)}</p>" for p in paragraphs]]

    def section(heading, table):
        text.extend([f"## {heading}", markdown_table(table)])
        body.extend([f"<h2>{html.escape(heading)}</h2>", table.to_html(index=False, float_format=lambda v: f"{v:.4g}") if not table.empty else "<p>No completed results.</p>"])

    section("Comparison coverage", read_table(directory / "coverage.csv"))
    section("Perturbations", pd.DataFrame([
        {k: v for k, v in row.items() if k != "generation"}
        for row in read_json(directory / "scenarios.json")
    ]))
    settings = []
    for scenario in read_json(directory / "scenarios.json"):
        for mode in config["modes"]:
            path = directory / "scenarios" / scenario["name"] / mode / "selection.json"
            if not path.exists():
                continue
            for point in read_json(path)["operating_points"]:
                definition = point["definition"]
                settings.append({
                    "scenario": scenario["name"], "mode": mode,
                    "pipeline": point["pipeline"], "criterion": point["criterion"],
                    "graph_mode": definition["graph_mode"], "resolution": definition["resolution"],
                    "setting_id": point["setting_id"],
                })
    section("Clustering settings used", pd.DataFrame(settings))
    for heading, filename, prefix in (
        ("Absolute clustering performance", "results_summary.csv", ""),
        ("Paired clustering changes", "results_delta_summary.csv", "delta_"),
    ):
        frame = read_table(directory / filename)
        if not frame.empty:
            columns = ["scenario", "mode", "pipeline", "setting_id", "n_realizations"]
            if prefix:
                columns.append("n_controls")
            columns.extend(f"{prefix}{metric}_{stat}" for metric in ("M0_f1", "Mge3_contamination") for stat in ("mean", "std", "count"))
            section(heading, frame[columns])
    (directory / "report.md").write_text("\n\n".join(text) + "\n")
    (directory / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>' + html.escape(title)
        + '</title><style>body{font:15px system-ui;margin:2em}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}td,th{padding:.4em;border:1px solid #ccc}</style><body>'
        + "\n".join(body) + "</body></html>\n"
    )
