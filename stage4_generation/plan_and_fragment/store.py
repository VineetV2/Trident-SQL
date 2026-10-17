"""JSONL storage, resume bookkeeping and stats for the generation stage.

Resume unit = (question_id, generator). A pair is done when the file holds exactly n rows for it
and not every row carries an error (all-error pairs are request-level failures: retried on the
next run). Incomplete or all-error pairs are dropped from the file at start-up (atomic rewrite).
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from collections import Counter, defaultdict

log = logging.getLogger(__name__)


ERROR_GATE_MAX_FAILED = 0.05  # > 5 % failed (question, generator) requests -> exit code 3


def read_jsonl(path: str) -> list[dict]:
    return read_jsonl_counted(path)[0]


def read_jsonl_counted(path: str) -> tuple[list[dict], int]:
    """(well-formed JSON object lines, number of malformed lines skipped, e.g. a crash mid-write)."""
    if not os.path.exists(path):
        return [], 0
    rows, bad = [], 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                bad += 1
    if bad:
        log.warning("%s: skipped %d malformed lines", path, bad)
    return rows, bad


def write_jsonl_atomic(path: str, rows: list[dict]) -> None:
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(tmp, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    os.replace(tmp, path)


def write_json_atomic(path: str, obj: object) -> None:
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _ends_mid_line(path: str) -> bool:
    """True if the file's last byte is not a newline (a partial line from a crash)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1) != b"\n"


def append_rows(path: str, rows: list[dict]) -> None:
    """One write call per batch (a pair's n rows land together), then flush + fsync.

    A partial trailing line left by a crash is terminated first, so it stays one malformed line
    (skipped and counted by the reader) instead of swallowing the first new row.
    """
    if not rows:
        return
    blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if _ends_mid_line(path):
        log.warning("%s: terminating a partial trailing line before appending", path)
        blob = "\n" + blob
    with open(path, "a") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())


def _pair(row: dict) -> tuple[int, str]:
    return int(row["question_id"]), str(row["generator"])


def prepare_candidates(path: str, n: int) -> tuple[set[tuple[int, str]], int]:
    """(done (question_id, generator) pairs, malformed lines skipped).

    Incomplete, all-error (retried by default) and malformed lines are dropped from the file.
    """
    parsed, bad = read_jsonl_counted(path)
    rows = [r for r in parsed if "question_id" in r and "generator" in r]
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[_pair(r)].append(r)
    done = {
        key for key, grp in groups.items()
        if len(grp) == n and sorted(r.get("sample") for r in grp) == list(range(n))
        and not all(r.get("error") for r in grp)
    }
    kept = [r for r in rows if _pair(r) in done]
    if os.path.exists(path):
        with open(path) as f:
            n_lines = sum(1 for line in f if line.strip())
        if len(kept) != n_lines:
            log.info("resume: dropping %d lines of incomplete or failed pairs from %s", n_lines - len(kept), path)
            write_jsonl_atomic(path, kept)
    return done, bad


def _request_tokens(rows: list[dict]) -> int:
    """completion_tokens is response-level (shared by the n samples): count it once per request."""
    return sum(int(r.get("completion_tokens") or 0) for r in rows if r.get("sample") == 0)


def pair_is_failed(rows: list[dict]) -> bool:
    """A (question, generator) request failed: every sample errored or none yielded SQL."""
    return all(r.get("error") for r in rows) or not any(r.get("sql") for r in rows)


def error_gate(rows: list[dict]) -> dict:
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[_pair(r)].append(r)
    failed = sum(1 for grp in groups.values() if pair_is_failed(grp))
    rate = failed / len(groups) if groups else 0.0
    return {"requests": len(groups), "failed_requests": failed, "failed_rate": round(rate, 4),
            "max_failed_rate": ERROR_GATE_MAX_FAILED, "passed": rate <= ERROR_GATE_MAX_FAILED}


def compute_stats(rows: list[dict], extra: dict) -> dict:
    """Counts per generator over the whole candidates file, plus run-level counters."""
    by_gen: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_gen[str(r.get("generator"))].append(r)
    gens = {}
    for gen, grp in sorted(by_gen.items()):
        errors = Counter(str(r["error"]).split(":")[0] for r in grp if r.get("error"))
        gens[gen] = {
            "questions": len({r["question_id"] for r in grp}),
            "candidates": len(grp),
            "errors": sum(errors.values()),
            "error_kinds": dict(sorted(errors.items())),
            "truncated": sum(1 for r in grp if r.get("truncated")),
            "empty_content": sum(1 for r in grp if r.get("empty")),
            "no_sql": sum(1 for r in grp if not r.get("sql")),
            "completion_tokens_total": _request_tokens(grp),
        }
    return {
        "candidates": len(rows),
        "errors": sum(g["errors"] for g in gens.values()),
        "truncated": sum(g["truncated"] for g in gens.values()),
        "empty_content": sum(g["empty_content"] for g in gens.values()),
        "no_sql": sum(g["no_sql"] for g in gens.values()),
        "by_generator": gens,
        "error_gate": error_gate(rows),
        "note": "completion_tokens on each row is the request total shared by its n samples",
        **extra,
    }
