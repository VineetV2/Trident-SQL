"""Per-database prep for B1 (CPU only, cached): columns.json + lexical value index.

Adapted from an earlier version of this prep (see profile.py, value_index.py, descriptions.py
for per-part changes). Changes here: one entry point per database with explicit paths (no env
vars, no tables.json -- keys come from PRAGMA); outputs only what the B1 contract asks for;
cached: skipped when all outputs exist unless force=True; outputs written atomically with
prep_meta.json last, so a crashed build is rebuilt on the next run; per-column failures are
recorded in prep_meta.json["errors"] instead of being swallowed.

Outputs in <out_dir>/:
  columns.json    {"table.column": {type, description, value_description, n_distinct,
                                    null_frac, examples}}
  values.sqlite   lexical value index (see value_index.py)
  prep_meta.json  counts, capped/skipped columns, errors (completion marker)

Usage:
  python -m stage0_dataset.prep.build --db-root <dir with <db_id>/<db_id>.sqlite> --out-root <dir> \
      [--db-ids a b ...] [--force] [--max-values-per-column 100000]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid

from stage0_dataset.prep.descriptions import load_descriptions
from stage0_dataset.prep.profile import profile_column, row_count
from stage0_dataset.prep.schema_info import list_tables, table_columns
from stage0_dataset.prep.value_index import DEFAULT_MAX_VALUES_PER_COLUMN, INDEX_FILE, build_value_index

log = logging.getLogger(__name__)

COLUMNS_FILE = "columns.json"
META_FILE = "prep_meta.json"
PREP_VERSION = 1


def _write_json(path: str, obj) -> None:
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def is_built(out_dir: str) -> bool:
    return all(os.path.exists(os.path.join(out_dir, n)) for n in (COLUMNS_FILE, INDEX_FILE, META_FILE))


def _column_record(db_file: str, table: str, col, desc: dict, n_rows, errors: list) -> dict:
    prof = profile_column(db_file, table, col.name, n_rows)
    errors.extend(f"{table}.{col.name}: {e}" for e in prof.errors)
    d = desc.get(col.name.lower(), {})
    return {
        "type": col.type,
        "description": d.get("description", ""),
        "value_description": d.get("value_description", ""),
        "n_distinct": prof.n_distinct,
        "null_frac": prof.null_frac,
        "examples": prof.examples,
    }


def build_columns(db_file: str, description_dir: str) -> tuple[dict, list[tuple[str, str]], list[str]]:
    """Profile every column of every table. Returns (columns dict, [(table, column)], errors)."""
    descriptions = load_descriptions(description_dir)
    columns: dict[str, dict] = {}
    pairs: list[tuple[str, str]] = []
    errors: list[str] = []
    for table in list_tables(db_file):
        desc = descriptions.get(table.lower(), {})
        if not desc:
            log.info("%s: no description CSV for table %s", os.path.basename(db_file), table)
        n_rows, err = row_count(db_file, table)
        if err:
            errors.append(f"{table}: row count: {err}")
        for col in table_columns(db_file, table):
            columns[f"{table}.{col.name}"] = _column_record(db_file, table, col, desc, n_rows, errors)
            pairs.append((table, col.name))
    return columns, pairs, errors


def build_prep(db_file: str, description_dir: str, out_dir: str, force: bool = False,
               max_values_per_column: int = DEFAULT_MAX_VALUES_PER_COLUMN) -> None:
    """Build (or reuse) the prep outputs for one database."""
    if not os.path.isfile(db_file):
        raise FileNotFoundError(f"database not found: {db_file}")
    if is_built(out_dir) and not force:
        log.info("prep cached: %s", out_dir)
        return
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, META_FILE)
    if os.path.exists(meta_path):
        os.remove(meta_path)  # marker goes first, so a crash below forces a rebuild
    start = time.monotonic()
    columns, pairs, errors = build_columns(db_file, description_dir)
    report = build_value_index(db_file, pairs, out_dir, max_values_per_column)
    _write_json(os.path.join(out_dir, COLUMNS_FILE), columns)
    meta = {
        "prep_version": PREP_VERSION,
        "db_file": os.path.abspath(db_file),
        "description_dir": os.path.abspath(description_dir),
        "n_tables": len({t for t, _ in pairs}),
        "n_columns": len(columns),
        "n_described": sum(1 for c in columns.values() if c["description"]),
        "n_values": report.n_values,
        "max_values_per_column": max_values_per_column,
        "indexed_columns": report.indexed_columns,
        "skipped_columns": report.skipped_columns,
        "capped_columns": report.capped_columns,
        "errors": errors + report.errors,
        "build_seconds": round(time.monotonic() - start, 2),
    }
    _write_json(meta_path, meta)
    if meta["errors"]:
        log.warning("%s: %d prep errors (see %s)", out_dir, len(meta["errors"]), meta_path)
    log.info("prep %s: %d columns, %d values, %.1fs", out_dir, len(columns), report.n_values, meta["build_seconds"])


def _db_ids(db_root: str, wanted: list[str] | None) -> list[str]:
    if wanted:
        return wanted
    return sorted(d for d in os.listdir(db_root) if os.path.isdir(os.path.join(db_root, d)))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--db-ids", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-values-per-column", type=int, default=DEFAULT_MAX_VALUES_PER_COLUMN)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for db_id in _db_ids(args.db_root, args.db_ids):
        db_dir = os.path.join(args.db_root, db_id)
        build_prep(os.path.join(db_dir, f"{db_id}.sqlite"), os.path.join(db_dir, "database_description"),
                   os.path.join(args.out_root, db_id), force=args.force,
                   max_values_per_column=args.max_values_per_column)


if __name__ == "__main__":
    main()
