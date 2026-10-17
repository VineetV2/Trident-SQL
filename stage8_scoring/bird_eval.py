"""BIRD execution-accuracy (EX) evaluator.

Faithful port of the official script
(DAMO-ConvAI/bird/llm/src/evaluation.py, sha256 2f591e55...):
  - correct iff set(rows of predicted SQL) == set(rows of gold SQL) on the question's database
  - 30 s budget per question; timeout or any exception (pred OR gold) counts wrong

Deliberate differences:
  - Fixed denominator = every question in the gold file. Missing predictions count wrong.
  - Results keyed by question_id, never by list position.
  - Databases opened read-only, so a predicted DELETE/UPDATE cannot corrupt later scoring.
  - No DDL, ATTACH or writes (common.sql.execute.harden_connection): a predicted CREATE TEMP VIEW could
    otherwise shadow a table the gold SQL reads on the same connection. Such a prediction counts wrong.
  - Gold execution failures are reported explicitly instead of silently scored 0.

Usage:
  python -m stage8_scoring.bird_eval --pred preds.json --gold dev.json \
      --db-root data/raw/dev_20240627/dev_databases --out score.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from typing import Optional

from common.sql.execute import harden_connection

OFFICIAL_SEPARATOR = "\t----- bird -----\t"
DEFAULT_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class CompareResult:
    correct: bool
    pred_error: Optional[str] = None
    gold_error: Optional[str] = None


class _Timeout(Exception):
    pass


def _run(conn: sqlite3.Connection, sql: str, deadline: float) -> list:
    def guard() -> int:
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(guard, 10_000)
    try:
        return conn.execute(sql).fetchall()
    except sqlite3.OperationalError as exc:
        if time.monotonic() > deadline and "interrupted" in str(exc):
            raise _Timeout() from exc
        raise


def compare_one(
    pred_sql: str, gold_sql: str, db_path: str, timeout_s: float = DEFAULT_TIMEOUT_S
) -> CompareResult:
    """Execute pred then gold on one read-only connection, sharing one time budget."""
    deadline = time.monotonic() + timeout_s
    conn = harden_connection(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True))
    try:
        try:
            pred_rows = _run(conn, pred_sql, deadline)
        except _Timeout:
            return CompareResult(False, pred_error="timeout")
        except Exception as exc:  # any failure of the prediction = wrong, as official
            return CompareResult(False, pred_error=f"{type(exc).__name__}: {exc}")
        try:
            gold_rows = _run(conn, gold_sql, deadline)
        except _Timeout:
            return CompareResult(False, gold_error="timeout")
        except Exception as exc:
            return CompareResult(False, gold_error=f"{type(exc).__name__}: {exc}")
        return CompareResult(set(pred_rows) == set(gold_rows))
    finally:
        conn.close()


def load_predictions(path: str) -> dict[int, str]:
    """Load {question_id: sql}. Accepts the official 'SQL\\t----- bird -----\\tdb_id' values."""
    with open(path) as f:
        raw = json.load(f)
    preds = {}
    for qid, value in raw.items():
        sql = value.split(OFFICIAL_SEPARATOR)[0] if isinstance(value, str) else ""
        preds[int(qid)] = sql
    return preds


def _db_path(db_root: str, db_id: str) -> str:
    return f"{db_root.rstrip('/')}/{db_id}/{db_id}.sqlite"


def _worker(args: tuple) -> tuple:
    qid, pred_sql, gold_sql, db_path, timeout_s = args
    return qid, compare_one(pred_sql, gold_sql, db_path, timeout_s)


def score(
    preds: dict[int, str],
    gold: list[dict],
    db_root: str,
    num_workers: int = 8,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict:
    gold_ids = [g["question_id"] for g in gold]
    if len(set(gold_ids)) != len(gold_ids):
        raise ValueError("gold file has duplicate question_ids")
    unknown = sorted(set(preds) - set(gold_ids))
    if unknown:
        raise ValueError(f"predictions for question_ids not in gold: {unknown[:10]}")

    missing = sorted(q for q in gold_ids if not (preds.get(q) or "").strip())
    jobs = [
        (g["question_id"], preds[g["question_id"]], g["SQL"], _db_path(db_root, g["db_id"]), timeout_s)
        for g in gold
        if g["question_id"] not in missing
    ]
    results: dict[int, CompareResult] = {q: CompareResult(False, pred_error="missing") for q in missing}
    if num_workers <= 1:
        results.update(dict(map(_worker, jobs)))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            results.update(dict(pool.map(_worker, jobs, chunksize=4)))

    by_diff: dict[str, dict] = {}
    for g in gold:
        bucket = by_diff.setdefault(g.get("difficulty", "unknown"), {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += int(results[g["question_id"]].correct)
    for bucket in by_diff.values():
        bucket["ex"] = 100.0 * bucket["correct"] / bucket["n"]

    correct = sum(int(r.correct) for r in results.values())
    return {
        "n": len(gold),
        "correct": correct,
        "ex": 100.0 * correct / len(gold),
        "by_difficulty": by_diff,
        "missing": missing,
        "gold_errors": sorted(q for q, r in results.items() if r.gold_error),
        "per_question": {str(q): asdict(results[q]) for q in gold_ids},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="JSON {question_id: sql}")
    ap.add_argument("--gold", required=True, help="BIRD dev json (list of records with SQL)")
    ap.add_argument("--db-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-n", type=int, default=None, help="abort unless gold has exactly this many questions")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    args = ap.parse_args()

    with open(args.gold) as f:
        gold = json.load(f)
    if args.expect_n is not None and len(gold) != args.expect_n:
        raise SystemExit(f"gold has {len(gold)} questions, expected {args.expect_n}")
    report = score(load_predictions(args.pred), gold, args.db_root, args.workers, args.timeout)
    report["inputs"] = {"pred": args.pred, "gold": args.gold, "db_root": args.db_root, "timeout_s": args.timeout}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)

    print(f"EX = {report['correct']}/{report['n']} = {report['ex']:.2f}%")
    for name in ("simple", "moderate", "challenging"):
        b = report["by_difficulty"].get(name)
        if b:
            print(f"  {name:12s} {b['correct']}/{b['n']} = {b['ex']:.2f}%")
    print(f"missing predictions: {len(report['missing'])}   gold execution errors: {len(report['gold_errors'])}")


if __name__ == "__main__":
    main()
