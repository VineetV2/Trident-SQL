"""Lexical value index: every stored text value per column, for literal retrieval.

Adapted from an earlier version of this prep (its value dump).
Kept: values <= 100 chars, stored verbatim (only end whitespace trimmed); columns whose sampled
values are all UUIDs are skipped. Changed: selection is by stored type (typeof = 'text'), so
DATE-declared and untyped columns holding text are included and all-number text columns (zip
codes, '0040') are KEPT -- this index is lexical, exact matches on such codes are its job;
no profile-card dependency and no hard-coded DBDIR; per-column cap (default 500k distinct values,
most frequent first) with capped columns recorded; output is one SQLite file `values.sqlite`
(table vals(tbl, col, value)) instead of JSONL; errors are returned for counting, never
swallowed. The earlier embedding step is not included: off by default (B1_DESIGN).
"""
from __future__ import annotations

import os
import re
import sqlite3
import uuid as uuid_mod
from dataclasses import dataclass, field

from stage0_dataset.prep.profile import PREP_TIMEOUT_S, quote_ident
from common.sql.execute import execute

VALUE_MAX_CHARS = 100
DEFAULT_MAX_VALUES_PER_COLUMN = 500_000
UUID_SAMPLE = 40
INDEX_FILE = "values.sqlite"
UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


@dataclass
class IndexReport:
    n_values: int = 0
    indexed_columns: list = field(default_factory=list)
    skipped_columns: list = field(default_factory=list)  # all-UUID columns
    capped_columns: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def _column_values(db_file: str, table: str, column: str, cap: int):
    t, c = quote_ident(table), quote_ident(column)
    sql = (f"SELECT TRIM({c}) AS v FROM {t} WHERE typeof({c}) = 'text' "
           f"AND length(TRIM({c})) BETWEEN 1 AND {VALUE_MAX_CHARS} "
           f"GROUP BY v ORDER BY COUNT(*) DESC, v")
    return execute(db_file, sql, timeout_s=PREP_TIMEOUT_S, max_rows=cap)


def _all_uuid(values: list[str]) -> bool:
    sample = values[:UUID_SAMPLE]
    return bool(sample) and all(UUID_PATTERN.match(v) for v in sample)


def _write_rows(out_dir: str, rows: list[tuple[str, str, str]]) -> None:
    final = os.path.join(out_dir, INDEX_FILE)
    tmp = f"{final}.{uuid_mod.uuid4().hex}.tmp"
    conn = sqlite3.connect(tmp)
    try:
        conn.execute("CREATE TABLE vals (tbl TEXT NOT NULL, col TEXT NOT NULL, value TEXT NOT NULL)")
        conn.executemany("INSERT INTO vals VALUES (?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()
    os.replace(tmp, final)


def build_value_index(db_file: str, columns: list[tuple[str, str]], out_dir: str,
                      max_values_per_column: int = DEFAULT_MAX_VALUES_PER_COLUMN) -> IndexReport:
    """columns: [(table, column)] to consider. Writes <out_dir>/values.sqlite."""
    report = IndexReport()
    rows: list[tuple[str, str, str]] = []
    for table, column in columns:
        key = f"{table}.{column}"
        res = _column_values(db_file, table, column, max_values_per_column)
        if not res.ok:
            report.errors.append(f"{key}: values: {res.error}")
            continue
        values = [r[0] for r in res.rows]
        if not values:
            continue
        if _all_uuid(values):
            report.skipped_columns.append(key)
            continue
        if res.rows_capped:
            report.capped_columns.append(key)
        report.indexed_columns.append(key)
        rows.extend((table, column, v) for v in values)
    _write_rows(out_dir, rows)
    report.n_values = len(rows)
    return report


def load_values(prep_dir: str) -> list[tuple[str, str, str]]:
    """All (table, column, value) rows of a prep directory's value index."""
    path = os.path.join(prep_dir, INDEX_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"value index missing: {path}")
    res = execute(path, "SELECT tbl, col, value FROM vals ORDER BY rowid", max_rows=50_000_000)
    if not res.ok:
        raise RuntimeError(f"cannot read value index {path}: {res.error}")
    return [tuple(r) for r in res.rows]
