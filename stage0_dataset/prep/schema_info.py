"""Tables, columns, primary and foreign keys of a SQLite database, via PRAGMA.

New code (the old profile-card builder read keys from tables.json, which BIRD train_sample does
not ship). Every statement goes through common.sql.execute.execute (read-only, timeout).
Composite foreign keys come back from PRAGMA foreign_key_list as one row per column sharing an
`id`; they are grouped here. A foreign key with an empty target column list references the
target table's primary key (SQLite rule) and is resolved to it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from common.sql.execute import execute

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    pk: int  # 0 = not in the PK, else 1-based position in the (composite) PK


@dataclass(frozen=True)
class ForeignKey:
    from_cols: tuple[str, ...]
    ref_table: str
    to_cols: tuple[str, ...]


def quote_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _pragma(db_file: str, sql: str) -> tuple:
    res = execute(db_file, sql)
    if not res.ok:
        raise RuntimeError(f"{sql} failed on {db_file}: {res.error}")
    return res.rows


def list_tables(db_file: str) -> list[str]:
    rows = _pragma(db_file, "SELECT name FROM sqlite_master WHERE type='table' "
                            "AND name NOT LIKE 'sqlite_%' ORDER BY name")
    return [r[0] for r in rows]


def table_columns(db_file: str, table: str) -> list[Column]:
    rows = _pragma(db_file, f"SELECT name, type, pk FROM pragma_table_info({quote_literal(table)})")
    return [Column(name=r[0], type=(r[1] or "").strip(), pk=int(r[2] or 0)) for r in rows]


def primary_key(db_file: str, table: str) -> tuple[str, ...]:
    cols = sorted((c for c in table_columns(db_file, table) if c.pk), key=lambda c: c.pk)
    return tuple(c.name for c in cols)


def _resolve_table(db_file: str, name: str) -> str:
    """BIRD declares some FK targets with different case than the real table name."""
    for table in list_tables(db_file):
        if table.lower() == name.lower():
            return table
    return name


def foreign_keys(db_file: str, table: str) -> list[ForeignKey]:
    rows = _pragma(db_file, 'SELECT id, seq, "table", "from", "to" '
                            f"FROM pragma_foreign_key_list({quote_literal(table)}) ORDER BY id, seq")
    groups: dict[int, list[tuple]] = {}
    for row in rows:
        groups.setdefault(row[0], []).append(row)
    out = []
    for parts in groups.values():
        ref = _resolve_table(db_file, parts[0][2])
        from_cols = tuple(p[3] for p in parts)
        to_cols = tuple(p[4] for p in parts)
        if any(c is None or c == "" for c in to_cols):
            to_cols = primary_key(db_file, ref)
        if len(to_cols) != len(from_cols):
            log.warning("skipping malformed foreign key %s.%s -> %s", table, from_cols, ref)
            continue
        out.append(ForeignKey(from_cols=from_cols, ref_table=ref, to_cols=to_cols))
    return out
