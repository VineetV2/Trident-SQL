"""Query-tree targeted fix for DISTINCT. decide() is applied per candidate by stage 6 (run_query_tree, fix
`distinct`) and to the selected SQL by stage 8 (run_scoring).

Find -> decide with the data -> edit only that part (learned on 3,002 train questions, then frozen):
  1. the SQL's tree flags distinct_no_cue (DISTINCT, but the question/hint has no distinct cue);
  2. the DISTINCT sits inside an aggregate, COUNT(DISTINCT x) (SELECT DISTINCT never changes a set-based result);
  3. x is NOT a primary key of its table (PRAGMA table_info);
  4. the same SQL without that DISTINCT (recipes.drop_distinct) runs and returns a DIFFERENT result;
  -> the SQL becomes the edited SQL. Nothing else in the query changes. No gold is read.
"""
from __future__ import annotations

import sqlite3
from typing import Dict, Optional, Set, Tuple

from stage6_query_tree.qtree.match import match
from stage6_query_tree.qtree.recipes import repairs
from stage6_query_tree.qtree.sql_tree import SQLTree, parse_sql_tree
from common.sql.execute import execute

EXEC_TIMEOUT_S = 30


def primary_keys(db_file: str, cache: Dict[str, Dict[str, Set[str]]]) -> Dict[str, Set[str]]:
    """table (lower-case) -> primary-key columns (lower-case), read-only."""
    if db_file not in cache:
        out: Dict[str, Set[str]] = {}
        con = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        try:
            for (t,) in con.execute("select name from sqlite_master where type='table'"):
                q = '"' + t.replace('"', '""') + '"'
                out[t.lower()] = {c[1].lower() for c in con.execute(f"PRAGMA table_info({q})").fetchall() if c[5]}
        finally:
            con.close()
        cache[db_file] = out
    return cache[db_file]


def count_distinct_target(tree: SQLTree) -> Optional[Tuple[Optional[str], str]]:
    """(table, column) inside an aggregate with DISTINCT (COUNT(DISTINCT x) in practice; same test as the frozen
    rule's "count" form), or None when the DISTINCT is not inside an aggregate."""
    for o in tree.outputs:
        if o.distinct and o.columns:
            t, c = sorted(o.columns, key=lambda x: (x[0] or "", x[1]))[0]
            return (t, c)
    return None


def decide(sql: str, side, db_file: str, pk_cache: dict) -> Tuple[Optional[str], str]:
    """(edited SQL or None, reason)."""
    tree = parse_sql_tree(sql)
    if not tree.ok or "distinct_no_cue" not in {m.kind for m in match(tree, side)}:
        return None, "not_flagged"
    target = count_distinct_target(tree)
    if target is None:
        return None, "not_count_distinct"
    tab, col = target
    if str(col).lower() in primary_keys(db_file, pk_cache).get((tab or "").lower(), set()):
        return None, "target_is_key"
    new = next((s for r, s in repairs(sql, side, {"distinct_no_cue"}) if r == "drop_distinct"), None)
    if new is None:
        return None, "no_edit"
    a, b = execute(db_file, sql, timeout_s=EXEC_TIMEOUT_S), execute(db_file, new, timeout_s=EXEC_TIMEOUT_S)
    if not (a.ok and b.ok):
        return None, "does_not_run"
    if a.result_hash == b.result_hash:
        return None, "same_result"
    return new, "edited"
