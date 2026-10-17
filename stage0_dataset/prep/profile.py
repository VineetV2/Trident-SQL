"""Per-column profile facts: distinct count, null fraction, most frequent example values.

Adapted from an earlier version of this prep (the profile-card builder's profile_column). Changes:
all SQL goes through common.sql.execute.execute
(no threading.Timer watchdog); a failed query is returned as an error string for the caller to
count instead of being swallowed with `except: pass`; distinct and null counts come from one
query; examples are the <=5 most frequent non-null, non-empty values ordered deterministically
(count desc, then value) and kept in their native type, strings truncated to 60 chars; no
numeric min/max, siblings, FK verification or inferred joins (not in the B1 contract).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from common.sql.execute import execute

MAX_EXAMPLES = 5
EXAMPLE_MAX_CHARS = 60
PREP_TIMEOUT_S = 120.0


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@dataclass(frozen=True)
class ColumnProfile:
    n_distinct: Optional[int]
    null_frac: Optional[float]
    examples: list
    errors: tuple[str, ...]


def row_count(db_file: str, table: str) -> tuple[Optional[int], Optional[str]]:
    res = execute(db_file, f"SELECT COUNT(*) FROM {quote_ident(table)}", timeout_s=PREP_TIMEOUT_S)
    return (int(res.rows[0][0]), None) if res.ok else (None, res.error)


def _example(value):
    if isinstance(value, bytes):
        return None  # blobs make no useful prompt example
    if isinstance(value, str):
        return value[:EXAMPLE_MAX_CHARS]
    return value


def _examples(db_file: str, table: str, column: str) -> tuple[list, Optional[str]]:
    t, c = quote_ident(table), quote_ident(column)
    sql = (f"SELECT {c} FROM {t} WHERE {c} IS NOT NULL AND {c} != '' "
           f"GROUP BY {c} ORDER BY COUNT(*) DESC, {c} LIMIT {MAX_EXAMPLES}")
    res = execute(db_file, sql, timeout_s=PREP_TIMEOUT_S)
    if not res.ok:
        return [], res.error
    values = [_example(r[0]) for r in res.rows]
    return [v for v in values if v is not None], None


def profile_column(db_file: str, table: str, column: str, n_rows: Optional[int]) -> ColumnProfile:
    t, c = quote_ident(table), quote_ident(column)
    res = execute(db_file, f"SELECT COUNT(DISTINCT {c}), SUM({c} IS NULL) FROM {t}", timeout_s=PREP_TIMEOUT_S)
    errors = []
    n_distinct = null_frac = None
    if res.ok:
        n_distinct = int(res.rows[0][0] or 0)
        nulls = int(res.rows[0][1] or 0)
        null_frac = round(nulls / n_rows, 4) if n_rows else None
    else:
        errors.append(f"stats: {res.error}")
    examples, err = _examples(db_file, table, column)
    if err:
        errors.append(f"examples: {err}")
    return ColumnProfile(n_distinct=n_distinct, null_frac=null_frac, examples=examples, errors=tuple(errors))
