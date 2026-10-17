"""Read BIRD's database_description/*.csv files (one CSV per table).

Adapted from an earlier version of this prep (the profile-card builder's load_descriptions). Changes:
takes the description directory directly
(no hard-coded DBDIR); tries utf-8-sig -> cp1252 -> latin-1 per file (latin-1 never fails, so a
file is never silently dropped); collapses stray whitespace in values; keys are lower-cased
table (file stem) and column names so callers match case-insensitively; no 400-char truncation
here (the renderer decides what to cut); C1 control characters (U+0080..U+009F, e.g. U+0095 in
superstore's UTF-8 files: cp1252 bullets converted byte-for-byte) are mapped to their cp1252
characters.
"""
from __future__ import annotations

import csv
import io
import logging
import os

log = logging.getLogger(__name__)

ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
BOM = "\ufeff"
FIELDS = ("description", "value_description", "data_format")
_C1_TO_CP1252 = {i: bytes([i]).decode("cp1252", errors="ignore") or None for i in range(0x80, 0xA0)}
_SOURCE_KEYS = {"description": "column_description", "value_description": "value_description",
                "data_format": "data_format"}


def _read_text(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ENCODINGS:
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        if enc != "utf-8-sig":
            log.debug("%s decoded as %s", path, enc)
        return text.lstrip(BOM).translate(_C1_TO_CP1252)
    raise AssertionError("unreachable: latin-1 decodes any byte string")


def _clean(value: str | None) -> str:
    """Trim ends and normalise line endings; keep inner newlines (renderers collapse them)."""
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _parse_rows(text: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in csv.DictReader(io.StringIO(text, newline="")):
        norm = {(k or "").strip().lstrip(BOM).lower(): _clean(v) for k, v in row.items() if k}
        col = norm.get("original_column_name", "")
        if not col:
            continue
        out[col.lower()] = {field: norm.get(_SOURCE_KEYS[field], "") for field in FIELDS}
    return out


def load_descriptions(description_dir: str) -> dict[str, dict[str, dict[str, str]]]:
    """table_lower -> column_lower -> {description, value_description, data_format}."""
    if not os.path.isdir(description_dir):
        log.warning("no description directory at %s", description_dir)
        return {}
    out: dict[str, dict[str, dict[str, str]]] = {}
    for name in sorted(os.listdir(description_dir)):
        if not name.lower().endswith(".csv"):
            continue
        table = name[:-4].strip().lower()
        try:
            out[table] = _parse_rows(_read_text(os.path.join(description_dir, name)))
        except (OSError, csv.Error) as exc:
            log.error("could not read description file %s: %s", name, exc)
    return out
