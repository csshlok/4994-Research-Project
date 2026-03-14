from __future__ import annotations

import re
from typing import Any, Mapping


# Spreadsheet apps may interpret these as formulas, even with leading whitespace.
_DANGEROUS_CSV_RE = re.compile(r"^[\t\r\n\f\v ]*[=+\-@]")


def sanitize_csv_cell(value: Any) -> Any:
    """Prefix potentially dangerous string cells to prevent formula execution."""
    if not isinstance(value, str):
        return value
    if not value or value.startswith("'"):
        return value
    if _DANGEROUS_CSV_RE.match(value):
        return "'" + value
    return value


def sanitize_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {k: sanitize_csv_cell(v) for k, v in row.items()}


def sanitize_dataframe_for_csv(df):
    """
    Return a copy with object/string columns sanitized.
    Kept untyped to avoid hard dependency on pandas at import time.
    """
    if df is None or getattr(df, "empty", False):
        return df

    out = df.copy()
    obj_cols = list(out.select_dtypes(include=["object", "string"]).columns)
    for col in obj_cols:
        out[col] = out[col].map(sanitize_csv_cell)
    return out
