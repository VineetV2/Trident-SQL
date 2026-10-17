"""Shared, safe SQL execution for pipeline stages (repair, voting, probes, value lookup).

Read-only connection (see harden_connection), wall-clock timeout via SQLite progress handler, row
cap, and a stable result hash for grouping candidates by execution result (same set semantics as
BIRD's scorer).
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_ROWS = 10_000

# mode=ro alone is not a sandbox: a read-only connection still runs CREATE TEMP VIEW/TABLE (which
# then shadows the real table for every later query on a reused connection), ATTACH of another
# file (opened read-write), and VACUUM INTO any path (refused here through its internal ATTACH).
# Every other action (SELECT, READ, FUNCTION, RECURSIVE, TRANSACTION, SAVEPOINT) stays allowed.
_DENIED_ACTIONS = frozenset({
    sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH,
    sqlite3.SQLITE_CREATE_INDEX, sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_CREATE_VIEW, sqlite3.SQLITE_CREATE_VTABLE,
    sqlite3.SQLITE_CREATE_TEMP_INDEX, sqlite3.SQLITE_CREATE_TEMP_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER, sqlite3.SQLITE_CREATE_TEMP_VIEW,
    sqlite3.SQLITE_DROP_INDEX, sqlite3.SQLITE_DROP_TABLE, sqlite3.SQLITE_DROP_TRIGGER,
    sqlite3.SQLITE_DROP_VIEW, sqlite3.SQLITE_DROP_VTABLE,
    sqlite3.SQLITE_DROP_TEMP_INDEX, sqlite3.SQLITE_DROP_TEMP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_TRIGGER, sqlite3.SQLITE_DROP_TEMP_VIEW,
    sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_ALTER_TABLE, sqlite3.SQLITE_REINDEX, sqlite3.SQLITE_ANALYZE,
})
# The only PRAGMAs allowed: the read-only schema ones the pipeline itself runs, as statements or as
# pragma_*() table functions (which reach the authorizer as the same PRAGMA, name in lower case).
_ALLOWED_PRAGMAS = frozenset({"table_info", "table_xinfo", "foreign_key_list"})


def _read_only_authorizer(action, arg1, arg2, db_name, trigger_or_view):
    # must never raise: Python turns an exception here into a silent deny
    if action in _DENIED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_PRAGMA and str(arg1 or "").lower() not in _ALLOWED_PRAGMAS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def harden_connection(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Lock a mode=ro connection down for model-written SQL: no ATTACH, no DDL, no writes.

    A refused statement raises sqlite3.DatabaseError ("not authorized"; "authorization denied" for
    VACUUM INTO), which every caller already reports as a failed execution, like a syntax error.
    """
    conn.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
    conn.set_authorizer(_read_only_authorizer)
    return conn


@dataclass(frozen=True)
class ExecResult:
    rows: tuple                 # at most max_rows rows
    error: Optional[str]        # None when the query ran
    timed_out: bool
    rows_capped: bool           # more rows existed than max_rows
    result_hash: Optional[str]  # hash of the *set* of rows (order/duplicates ignored);
                                # None on error AND when rows_capped (only a prefix was seen, and
                                # which prefix depends on row order, so it cannot identify the set)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def empty(self) -> bool:
        return self.ok and self.n_rows == 0

    @property
    def all_null(self) -> bool:
        if not self.ok or self.n_rows == 0:
            return False
        return all(v is None for row in self.rows for v in row)


def _canonical(value):
    """Map values Python's == treats as equal to one form, so equal sets hash equally.

    BIRD compares set(pred) == set(gold); there 5 == 5.0 (and -0.0 == 0), so an integral float is
    hashed as the int. bool is left alone (SQLite never returns one). Everything else keeps its
    own type: '5' != 5 and b'5' != '5' in Python too.
    """
    if isinstance(value, float) and not isinstance(value, bool) and value.is_integer():
        return int(value)
    return value


def result_hash(rows) -> str:
    """Order- and duplicate-insensitive hash, matching BIRD's set(pred) == set(gold) rule."""
    canon = sorted({repr(tuple(_canonical(v) for v in r)) for r in rows})
    return hashlib.md5("\n".join(canon).encode()).hexdigest()


def execute(db_file: str, sql: str, timeout_s: float = DEFAULT_TIMEOUT_S, max_rows: int = DEFAULT_MAX_ROWS) -> ExecResult:
    if not sql or not sql.strip():
        return ExecResult((), "empty_sql", False, False, None)
    deadline = time.monotonic() + timeout_s
    conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    harden_connection(conn)
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        capped = len(rows) > max_rows
        rows = tuple(rows[:max_rows])
        # a capped result has no hash: its visible prefix depends on row order (see ExecResult)
        return ExecResult(rows, None, False, capped, None if capped else result_hash(rows))
    except sqlite3.OperationalError as exc:
        if time.monotonic() > deadline:
            return ExecResult((), "timeout", True, False, None)
        return ExecResult((), f"OperationalError: {exc}", False, False, None)
    except Exception as exc:  # any other failure is a failed execution, never a crash
        return ExecResult((), f"{type(exc).__name__}: {exc}", False, False, None)
    finally:
        conn.close()
