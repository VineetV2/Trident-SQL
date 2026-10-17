"""Post-selection DISTINCT fix on a selection snapshot, and a BIRD EX scorer for any stage snapshot.

Fix: reads a selection snapshot P (a path, because selection variants write to different files), applies
stage6_query_tree/qtree/distinct_fix.decide to pipeline_artifacts.sql_selection.final_selected_sql only, and writes a NEW snapshot
<dir>/<name>.post.snapshot (+ .post.snapshot.data/items.jsonl; P is never modified), with every edit in
<dir>/<name>.post.changes.jsonl and counts in <dir>/<name>.post.summary.json. Per-item timeout and isolation as in
stage6_query_tree/run_query_tree.py (an item that fails keeps its selected SQL). A complete output is skipped unless --force.
The question side comes from the snapshot item itself (stage snapshots carry every earlier stage's artifacts).

Score: stage8_scoring/bird_eval (port of the official BIRD script: set(pred rows) == set(gold rows), 30 s per question,
fixed denominator = every item in the snapshot, empty prediction = wrong). Gold SQL, db id and difficulty come from
the snapshot items (the base stages copy BIRD's dev.json into each item's input); this is the only place gold is read.
Writes {n, correct, ex, by_difficulty, per_question{qid: {correct, pred_error, gold_error}}, ...}.
Questions whose gold SQL fails stay in the denominator.

  python -m stage8_scoring.run_scoring --snapshot P [--db-root ROOT] [--force] [--score]
  python -m stage8_scoring.run_scoring --score-only --snapshot P [--db-root ROOT] [--out score.json]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import List, Optional

from common.snapshot import artifact, atomic_json, atomic_lines, iter_items, read_manifest, with_artifact, write_snapshot
from common.snapshot import qid as question_id
from stage6_query_tree.run_query_tree import DEFAULT_ITEM_TIMEOUT_S, call_with_timeout, db_file, side_of, sidecar, snapshot_complete
from stage8_scoring.bird_eval import DEFAULT_TIMEOUT_S, score
from stage6_query_tree.qtree import distinct_fix

SEL_STAGE, SEL_FIELD = "sql_selection", "final_selected_sql"


def post_path(snapshot_path: str) -> str:
    return sidecar(snapshot_path, ".post.snapshot")


def fix_one(item: dict, db_root: Optional[str], pk_cache: dict, timeout_s: float) -> dict:
    qid, sql = question_id(item), artifact(item, SEL_STAGE, SEL_FIELD)
    side = side_of(item)
    if not (sql or "").strip():
        return {"question_id": qid, "sql": sql, "reason": "empty_sql"}
    if side is None:
        return {"question_id": qid, "sql": sql, "reason": "no_question_side"}
    try:
        new, why = call_with_timeout(lambda: distinct_fix.decide(sql, side, db_file(item, db_root), pk_cache),
                                     timeout_s)
    except TimeoutError:
        return {"question_id": qid, "sql": sql, "reason": "timeout"}
    except Exception as exc:
        return {"question_id": qid, "sql": sql, "reason": "error", "error": f"{type(exc).__name__}: {exc}"}
    return {"question_id": qid, "sql": new if new is not None else sql, "reason": why, "before": sql}


def apply_post(snapshot_path: str, out_path: Optional[str] = None, db_root: Optional[str] = None,
               force: bool = False, item_timeout_s: float = DEFAULT_ITEM_TIMEOUT_S) -> dict:
    out_path = out_path or post_path(snapshot_path)
    if os.path.abspath(out_path) == os.path.abspath(snapshot_path):
        raise ValueError("output must differ from the selection snapshot")
    manifest, items = read_manifest(snapshot_path), list(iter_items(snapshot_path))
    summary_path = sidecar(out_path, ".summary.json")
    if not force and snapshot_complete(out_path, len(items)) and os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            return {**json.load(f), "skipped": "output snapshot already complete (use --force to redo)"}
    pk_cache: dict = {}
    results = [fix_one(it, db_root, pk_cache, item_timeout_s) for it in items]
    changes = [{"question_id": r["question_id"], "fix": "distinct", "before": r["before"], "after": r["sql"]}
               for r in results if r["reason"] == "edited"]
    new_items = [with_artifact(it, SEL_STAGE, {SEL_FIELD: r["sql"]}) for it, r in zip(items, results)]
    atomic_lines(sidecar(out_path, ".changes.jsonl"), changes)
    summary = {"input": os.path.abspath(snapshot_path), "output": os.path.abspath(out_path), "db_root": db_root,
               "n_items": len(items), "changed": len(changes), "reasons": dict(Counter(r["reason"] for r in results)),
               "failures": [{k: r.get(k) for k in ("question_id", "reason", "error")} for r in results
                            if r["reason"] in ("timeout", "error")]}
    atomic_json(summary_path, summary)
    write_snapshot(out_path, manifest, new_items)
    return summary


# ---------------------------------------------------------------- scoring
def _db_root_of(item: dict, db_root: Optional[str]) -> str:
    if db_root:
        return db_root
    path = item["input"]["database_path"]  # <root>/<db_id>/<db_id>.sqlite
    return os.path.dirname(os.path.dirname(path))


def score_snapshot(snapshot_path: str, db_root: Optional[str] = None, workers: int = 8,
                   timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
    """BIRD EX of final_selected_sql over every item of a stage snapshot (gold from the items)."""
    items = list(iter_items(snapshot_path))
    roots = {_db_root_of(it, db_root) for it in items}
    if len(roots) > 1:
        raise ValueError(f"items point at {len(roots)} database roots; pass --db-root")
    gold = [{"question_id": question_id(it), "SQL": it["input"]["gold_sql"], "db_id": it["input"]["database_id"],
             "difficulty": it["input"].get("difficulty") or "unknown"} for it in items]
    preds = {question_id(it): artifact(it, SEL_STAGE, SEL_FIELD) or "" for it in items}
    report = score(preds, gold, roots.pop() if roots else (db_root or ""), workers, timeout_s)
    return {**report, "snapshot": os.path.abspath(snapshot_path)}


def compare(before: dict, after: dict) -> dict:
    a, b = before["per_question"], after["per_question"]
    fixed = sorted(int(q) for q in a if not a[q]["correct"] and b.get(q, {}).get("correct"))
    broken = sorted(int(q) for q in a if a[q]["correct"] and not b.get(q, {}).get("correct"))
    return {"before": before["correct"], "after": after["correct"], "n": before["n"],
            "fixed": fixed, "broken": broken, "net": len(fixed) - len(broken)}


def _print_score(report: dict) -> None:
    print(f"EX = {report['correct']}/{report['n']} = {report['ex']:.2f}%  ({report['snapshot']})")
    for name in ("simple", "moderate", "challenging"):
        b = report["by_difficulty"].get(name)
        if b:
            print(f"  {name:12s} {b['correct']}/{b['n']} = {b['ex']:.2f}%")
    print(f"missing predictions: {len(report['missing'])}   gold execution errors: {len(report['gold_errors'])}")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True, help="a selection snapshot (.snapshot manifest)")
    ap.add_argument("--output", default=None, help="default <dir>/<name>.post.snapshot")
    ap.add_argument("--db-root", default=None, help="remap database paths to <db-root>/<db_id>/<db_id>.sqlite")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--item-timeout", type=float, default=DEFAULT_ITEM_TIMEOUT_S)
    ap.add_argument("--score", action="store_true", help="also score the input and the post snapshot (reads gold)")
    ap.add_argument("--score-only", action="store_true", help="only score --snapshot (reads gold)")
    ap.add_argument("--out", default=None, help="score JSON path (default next to the snapshot)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--expect-n", type=int, default=None, help="abort unless the snapshot has exactly this many items")
    a = ap.parse_args(argv)
    if a.expect_n is not None:
        n = read_manifest(a.snapshot).get("num_items")
        if n != a.expect_n:
            raise SystemExit(f"snapshot has {n} items, expected {a.expect_n}")
    if a.score_only:
        report = score_snapshot(a.snapshot, a.db_root, a.workers)
        atomic_json(a.out or sidecar(a.snapshot, ".score.json"), report)
        _print_score(report)
        return
    summary = apply_post(a.snapshot, a.output, a.db_root, a.force, a.item_timeout)
    print(json.dumps({k: summary.get(k) for k in ("output", "n_items", "changed", "reasons", "skipped")
                      if k in summary}, indent=1))
    if a.score:
        before = score_snapshot(a.snapshot, a.db_root, a.workers)
        after = score_snapshot(summary["output"], a.db_root, a.workers)
        diff = compare(before, after)
        atomic_json(a.out or sidecar(summary["output"], ".score.json"),
                     {"before": before, "after": after, "compare": diff})
        _print_score(before)
        _print_score(after)
        print(f"fixed {len(diff['fixed'])} / broken {len(diff['broken'])} (net {diff['net']:+d})")


if __name__ == "__main__":
    main()
