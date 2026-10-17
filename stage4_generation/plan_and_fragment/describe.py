"""Described schema for B1 prompts: CREATE TABLE text with per-column descriptions and values.

Adapted from an earlier version of this pipeline's schema renderer.
Kept: restrict to the linked columns + primary keys, per-column description, value statistics
(distinct count, % NULL) and value examples with this question's matched values leading them,
foreign keys only between shown tables. Changed: CREATE TABLE-like layout (one column per line
with a `--` comment) instead of the earlier bracketed profile list; reads the B1 prep output
(columns.json) and PRAGMA keys instead of the old bundle; also keeps FK columns joining two shown
tables; composite keys rendered as one clause; adds BIRD's value_description; value hits are the
B1 values.jsonl `hits` list (only hits on rendered columns are used) and are also listed in a "Relevant stored values:" block with
SQL-quoted literals; no capability cards, join-path block, env-flag hit gate or 6-value cap.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from stage0_dataset.prep.schema_info import ForeignKey, foreign_keys, list_tables, quote_literal, table_columns

log = logging.getLogger(__name__)

DESC_MAX_CHARS = 200
VALUE_DESC_MAX_CHARS = 300
MAX_HITS_PER_COLUMN = 3
MAX_BLOCK_VALUES_PER_COLUMN = 5
MAX_BLOCK_COLUMNS = 20
_PLAIN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
RESERVED = {
    "abort", "action", "add", "all", "alter", "and", "as", "asc", "between", "by", "case", "check",
    "collate", "column", "commit", "constraint", "create", "cross", "current", "default", "delete",
    "desc", "distinct", "drop", "else", "end", "escape", "except", "exists", "foreign", "from",
    "full", "glob", "group", "having", "if", "in", "index", "inner", "insert", "intersect", "into",
    "is", "join", "key", "left", "like", "limit", "match", "natural", "not", "null", "of", "offset",
    "on", "or", "order", "outer", "primary", "references", "regexp", "right", "row", "rows", "select",
    "set", "table", "then", "to", "transaction", "union", "unique", "update", "using", "values",
    "view", "when", "where", "with",
}


def quote_ident(name: str) -> str:
    if _PLAIN.fullmatch(name) and name.lower() not in RESERVED:
        return name
    return '"' + name.replace('"', '""') + '"'


def _collapse(text: str, limit: int) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."


def _load_columns(prep_dir: str) -> dict:
    path = os.path.join(prep_dir, "columns.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"prep output missing: {path} (run stage0_dataset.prep.build)")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- selection

def _keys_to_keep(schema: dict, fks: dict, shown: set[str]) -> dict[str, set[str]]:
    keep = {t: {c.name for c in schema[t] if c.pk} for t in shown}
    for t in shown:
        for fk in fks[t]:
            if fk.ref_table in shown:
                keep[t].update(fk.from_cols)
                keep[fk.ref_table].update(fk.to_cols)
    return keep


def _select(schema: dict, fks: dict, columns: Optional[list[str]]) -> dict[str, set[str]]:
    """table -> column names to show."""
    if columns is None:
        return {t: {c.name for c in cols} for t, cols in schema.items()}
    lookup = {f"{t}.{c.name}".lower(): (t, c.name) for t, cols in schema.items() for c in cols}
    chosen = [lookup[k.lower()] for k in columns if k.lower() in lookup]
    unknown = [k for k in columns if k.lower() not in lookup]
    if unknown:
        log.warning("describe: ignoring %d unknown columns: %s", len(unknown), unknown[:10])
    if not chosen:
        log.warning("describe: empty or unknown column selection (%d requested); "
                    "falling back to the full schema", len(columns))
        return _select(schema, fks, None)
    shown = {t for t, _ in chosen}
    keep = _keys_to_keep(schema, fks, shown)
    for t, c in chosen:
        keep[t].add(c)
    return keep


def _group_hits(value_hits: Optional[list[dict]], schema: dict, keep: dict[str, set[str]]) -> dict[tuple, list[str]]:
    """(table, column) -> matched values, best score first, only for rendered columns."""
    lookup = {f"{t}.{c.name}".lower(): (t, c.name) for t, cols in schema.items() for c in cols}
    grouped: dict[tuple, list[tuple[float, str]]] = {}
    for hit in value_hits or []:
        key = lookup.get(f"{hit.get('table', '')}.{hit.get('column', '')}".lower())
        if key is None or key[1] not in keep.get(key[0], ()):
            continue
        grouped.setdefault(key, []).append((-float(hit.get("score", 0.0)), str(hit.get("value", ""))))
    out = {}
    for key, items in grouped.items():
        values = []
        for _, v in sorted(items):
            if v not in values:
                values.append(v)
        out[key] = values
    return out


# ---------------------------------------------------------------- rendering

def _comment(name: str, meta: dict, hit_values: list[str], max_examples: int) -> str:
    parts = []
    desc = _collapse(meta.get("description", ""), DESC_MAX_CHARS)
    if desc and desc.lower().replace("_", " ") != name.lower().replace("_", " "):
        parts.append(desc)
    vdesc = _collapse(meta.get("value_description", ""), VALUE_DESC_MAX_CHARS)
    if vdesc:
        parts.append(f"value description: {vdesc}")
    stats = []
    if meta.get("n_distinct") is not None:
        stats.append(f"{meta['n_distinct']} distinct")
    if meta.get("null_frac"):
        stats.append(f"{meta['null_frac'] * 100:.3g}% null")
    if stats:
        parts.append(", ".join(stats))
    leading = hit_values[:MAX_HITS_PER_COLUMN]
    rest = [e for e in meta.get("examples", []) if str(e) not in leading]
    examples = (leading + rest)[: max(max_examples, len(leading))]
    if examples:
        parts.append(f"examples: {examples!r}")
    return "; ".join(parts)


def _fk_clause(fk: ForeignKey) -> str:
    src = ", ".join(quote_ident(c) for c in fk.from_cols)
    dst = ", ".join(quote_ident(c) for c in fk.to_cols)
    return f"FOREIGN KEY ({src}) REFERENCES {quote_ident(fk.ref_table)} ({dst})"


def _render_table(table: str, cols: list, keep: set[str], fks: list, shown: set[str],
                  meta: dict, hits: dict, max_examples: int) -> str:
    entries = []
    for col in cols:
        if col.name not in keep:
            continue
        body = f"{quote_ident(col.name)} {col.type}".rstrip()
        m = meta.get(f"{table}.{col.name}", {})
        entries.append((body, _comment(col.name, m, hits.get((table, col.name), []), max_examples)))
    pk = [c.name for c in sorted((c for c in cols if c.pk), key=lambda c: c.pk)]
    if pk:
        entries.append((f"PRIMARY KEY ({', '.join(quote_ident(c) for c in pk)})", ""))
    entries += [(_fk_clause(fk), "") for fk in fks if fk.ref_table in shown]
    lines = []
    for i, (body, comment) in enumerate(entries):
        sep = "," if i < len(entries) - 1 else ""
        lines.append(f"  {body}{sep}" + (f" -- {comment}" if comment else ""))
    return f"CREATE TABLE {quote_ident(table)} (\n" + "\n".join(lines) + "\n);"


def _hits_block(hits: dict) -> str:
    if not hits:
        return ""
    lines = ["Relevant stored values:"]
    for (table, column), values in list(hits.items())[:MAX_BLOCK_COLUMNS]:
        literals = ", ".join(quote_literal(v) for v in values[:MAX_BLOCK_VALUES_PER_COLUMN])
        lines.append(f"- {quote_ident(table)}.{quote_ident(column)}: {literals}")
    return "\n".join(lines)


def render_described_schema(prep_dir: str, db_file: str, columns: list[str] | None = None,
                            value_hits: list[dict] | None = None, max_examples: int = 3) -> str:
    """CREATE TABLE-like schema text for `columns` ("table.column", None = all) with descriptions,
    value statistics, examples, keys, and a block of this question's matched stored values."""
    meta = _load_columns(prep_dir)
    tables = list_tables(db_file)
    schema = {t: table_columns(db_file, t) for t in tables}
    fks = {t: foreign_keys(db_file, t) for t in tables}
    keep = _select(schema, fks, columns)
    shown = set(keep)
    hits = _group_hits(value_hits, schema, keep)
    blocks = [_render_table(t, schema[t], keep[t], fks[t], shown, meta, hits, max_examples)
              for t in tables if t in shown]
    block = _hits_block(hits)
    return "\n\n".join(blocks + ([block] if block else []))
