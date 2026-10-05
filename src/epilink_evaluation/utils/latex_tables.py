"""Shared LaTeX table writers using the project table environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def latex_escape(value: Any) -> str:
    if _is_missing(value):
        return ""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def latex_column_spec(column_spec: str | None, n_columns: int) -> str:
    spec = (column_spec or ("l" * n_columns)).strip()
    if spec.startswith("@{"):
        return spec
    return f"@{{}}{spec}@{{}}"


def addlinespace_after_group_changes(values: Any) -> set[int]:
    addlinespace_after: set[int] = set()
    previous_value: str | None = None
    for row_idx, value in enumerate(values.astype(str)):
        if previous_value is not None and value != previous_value:
            addlinespace_after.add(row_idx - 1)
        previous_value = value
    return addlinespace_after


def _body_lines(
    rows: list[list[Any]],
    *,
    expected_columns: int,
    addlinespace_after: set[int],
) -> list[str]:
    body = []
    for row_idx, row in enumerate(rows):
        if len(row) != expected_columns:
            raise ValueError(
                f"Table row {row_idx} has {len(row)} cells; expected {expected_columns}."
            )
        body.append("    " + " & ".join(latex_escape(cell) for cell in row) + r" \\")
        if row_idx in addlinespace_after and row_idx < len(rows) - 1:
            body.append(r"    \addlinespace[0.35em]")
    return body


def render_latex_table(
    *,
    caption: str,
    label: str,
    columns: list[str],
    rows: list[list[Any]],
    column_spec: str | None = None,
    addlinespace_after: set[int] | None = None,
    short_caption: str | None = None,
    landscape: bool = False,
) -> str:
    addlinespace_after = addlinespace_after or set()
    column_spec = latex_column_spec(column_spec, len(columns))
    header = " & ".join(f"\\textbf{{{latex_escape(col)}}}" for col in columns)
    lines = []
    if landscape:
        lines.append(r"\begin{landscape}")
    lines.extend(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            (
                f"\\caption[{latex_escape(short_caption or caption)}]"
                f"{{{latex_escape(caption)}}}\\label{{{label}}}"
            ),
            f"\\begin{{thesistablebody}}{{{column_spec}}}",
            r"\toprule",
            f"{header} " + r"\\",
            r"\midrule",
            *_body_lines(
                rows,
                expected_columns=len(columns),
                addlinespace_after=addlinespace_after,
            ),
            r"\bottomrule",
            r"\end{thesistablebody}",
            r"\end{table}",
        ]
    )
    if landscape:
        lines.append(r"\end{landscape}")
    return "\n".join(lines)


def render_latex_longtable(
    *,
    caption: str,
    label: str,
    columns: list[str],
    rows: list[list[Any]],
    column_spec: str | None = None,
    short_caption: str | None = None,
    caption_is_latex: bool = False,
    addlinespace_after: set[int] | None = None,
    landscape: bool = True,
    dense: bool = True,
    tiny: bool = False,
) -> str:
    """Render a dense longtable that can span pages.

    Set caption_is_latex for authored LaTeX captions containing math or
    cross-references. An explicit short_caption is always treated as plain text.
    """
    rendered_caption = caption if caption_is_latex else latex_escape(caption)
    rendered_short_caption = (
        latex_escape(short_caption) if short_caption else rendered_caption
    )
    addlinespace_after = addlinespace_after or set()
    column_count = len(columns)
    column_spec = latex_column_spec(column_spec, column_count)
    header = " & ".join(f"\\textbf{{{latex_escape(col)}}}" for col in columns)

    lines = []
    if landscape:
        lines.append(r"\begin{landscape}")
    lines.append(r"\begingroup")
    if tiny:
        lines.extend(
            [
                r"\renewcommand{\thesistablesetup}{%",
                r"  \tiny",
                r"  \setlength{\tabcolsep}{1.5pt}%",
                r"  \renewcommand{\arraystretch}{1.08}%",
                r"}",
            ]
        )
    elif dense:
        lines.append(r"\let\thesistablesetup\thesisdensetablesetup")
    lines.extend(
        [
            f"\\begin{{longtable}}{{{column_spec}}}",
            f"    \\caption[{rendered_short_caption}]"
            f"{{{rendered_caption}}}\\label{{{label}}} " + r"\\",
            r"    \toprule",
            f"    {header} " + r"\\",
            r"    \midrule",
            r"    \endfirsthead % chktex 1",
            (
                f"    \\multicolumn{{{column_count}}}{{l}}"
                r"{\small\itshape Table~\thetable\ continued from previous page} \\"
            ),
            r"    \toprule",
            f"    {header} " + r"\\",
            r"    \midrule",
            r"    \endhead % chktex 1",
            r"    \midrule",
            (
                f"    \\multicolumn{{{column_count}}}{{r}}"
                r"{\small\itshape Continued on next page} \\"
            ),
            r"    \endfoot % chktex 1",
            r"    \bottomrule",
            r"    \endlastfoot % chktex 1",
            *_body_lines(
                rows,
                expected_columns=column_count,
                addlinespace_after=addlinespace_after,
            ),
            r"\end{longtable}",
            r"\endgroup",
        ]
    )
    if landscape:
        lines.append(r"\end{landscape}")
    return "\n".join(lines)


def latex_shortstack(value: object, *, words_per_line: int = 2) -> str:
    words = str(value).split()
    if not words:
        return ""
    lines = [
        " ".join(words[idx : idx + words_per_line])
        for idx in range(0, len(words), words_per_line)
    ]
    return r"\shortstack{" + r"\\".join(latex_escape(line) for line in lines) + "}"


def render_latex_grouped_column_table(
    *,
    caption: str,
    label: str,
    row_columns: list[str],
    column_groups: list[tuple[str, list[str]]],
    rows: list[list[Any]],
    column_spec: str,
    addlinespace_after: set[int] | None = None,
    short_caption: str | None = None,
    font_size: str = r"\scriptsize",
    tabcolsep: str = "2.5pt",
    landscape: bool = False,
) -> str:
    addlinespace_after = addlinespace_after or set()
    n_columns = len(row_columns) + sum(len(periods) for _, periods in column_groups)
    column_spec = latex_column_spec(column_spec, n_columns)

    top_header = [f"\\textbf{{{latex_escape(column)}}}" for column in row_columns]
    cmidrules = []
    col_start = len(row_columns) + 1
    for era, periods in column_groups:
        top_header.append(
            f"\\multicolumn{{{len(periods)}}}{{c}}{{\\textbf{{{latex_shortstack(era)}}}}}"
        )
        col_end = col_start + len(periods) - 1
        cmidrules.append(f"\\cmidrule(lr){{{col_start}-{col_end}}}")
        col_start = col_end + 1

    period_header = [""] * len(row_columns)
    for _, periods in column_groups:
        period_header.extend(
            f"\\textbf{{{latex_escape(period)}}}" for period in periods
        )

    lines = []
    if landscape:
        lines.append(r"\begin{landscape}")
    lines.extend(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            font_size,
            f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
            (
                f"\\caption[{latex_escape(short_caption or caption)}]"
                f"{{{latex_escape(caption)}}}\\label{{{label}}}"
            ),
            f"\\begin{{thesistablebody}}{{{column_spec}}}",
            r"\toprule",
            " & ".join(top_header) + r" \\",
            "".join(cmidrules),
            " & ".join(period_header) + r" \\",
            r"\midrule",
            *_body_lines(
                rows,
                expected_columns=n_columns,
                addlinespace_after=addlinespace_after,
            ),
            r"\bottomrule",
            r"\end{thesistablebody}",
            r"\end{table}",
        ]
    )
    if landscape:
        lines.append(r"\end{landscape}")
    return "\n".join(lines)


def _output_path(output: Any, name: str | None) -> Path:
    if name is None:
        return Path(output)
    return Path(output.figure_dir) / f"{name}.tex"


def write_latex_table(
    output: Any,
    name: str | None = None,
    **kwargs: Any,
) -> Path:
    output_path = _output_path(output, name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_latex_table(**kwargs) + "\n")
    return output_path


def write_latex_longtable(
    output: Any,
    name: str | None = None,
    **kwargs: Any,
) -> Path:
    output_path = _output_path(output, name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_latex_longtable(**kwargs) + "\n")
    return output_path


def write_latex_grouped_column_table(
    output: Any,
    name: str | None = None,
    **kwargs: Any,
) -> Path:
    output_path = _output_path(output, name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_latex_grouped_column_table(**kwargs) + "\n")
    return output_path
