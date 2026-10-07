"""Check authored maths and plain-text escaping in manuscript table writers."""

import pytest

from epilink_evaluation.utils.latex_tables import (
    latex_escape,
    render_latex_grouped_column_table,
    render_latex_longtable,
)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("caption_is_latex", [False, True])
@pytest.mark.parametrize("headers_are_latex", [False, True])
def test_authored_latex_is_opt_in_and_body_text_stays_escaped(
    grouped, caption_is_latex, headers_are_latex
):
    caption = r"Target $M=0$, distant $M\ge3$; see Table~\ref{tab:full}."
    columns = ["Case $n$", "$F_1$"]
    kwargs = {
        "caption": caption,
        "short_caption": "Performance (%)",
        "label": "tab:maths",
        "rows": [["case_1 & case_2", "50%"]],
        "caption_is_latex": caption_is_latex,
        "headers_are_latex": headers_are_latex,
    }
    if grouped:
        rendered = render_latex_grouped_column_table(
            row_columns=columns[:1],
            column_groups=[("Performance (%)", columns[1:])],
            column_spec="lr",
            **kwargs,
        )
        assert r"\shortstack{Performance (\%)}" in rendered
    else:
        rendered = render_latex_longtable(columns=columns, **kwargs)

    rendered_caption = caption if caption_is_latex else latex_escape(caption)
    assert rf"\caption[Performance (\%)]{{{rendered_caption}}}" in rendered
    for column in columns:
        rendered_column = column if headers_are_latex else latex_escape(column)
        header = rf"\textbf{{{rendered_column}}}"
        assert rendered.count(header) == (1 if grouped else 2)
    assert r"case\_1 \& case\_2 & 50\%" in rendered


@pytest.mark.parametrize("grouped", [False, True])
def test_authored_caption_without_short_caption_preserves_maths(grouped):
    kwargs = {
        "caption": r"Target $M=0$ and $F_1$",
        "caption_is_latex": True,
        "label": "tab:maths",
        "rows": [],
    }
    if grouped:
        rendered = render_latex_grouped_column_table(
            row_columns=["Method"],
            column_groups=[("Performance", ["F1"])],
            column_spec="lr",
            **kwargs,
        )
    else:
        rendered = render_latex_longtable(columns=["Method", "F1"], **kwargs)
    assert r"\caption[Target $M=0$ and $F_1$]{Target $M=0$ and $F_1$}" in rendered
