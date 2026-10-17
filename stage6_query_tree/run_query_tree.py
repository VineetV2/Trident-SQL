"""Query-tree stage between the base stages' sql_revision (stage 5) and sql_selection (stage 7).

Reads <run_dir>/sql_revision.snapshot and writes a NEW snapshot (default <run_dir>/sql_revision_tree.snapshot, same
structured layout: <name>.snapshot manifest + <name>.snapshot.data/items.jsonl of {input, pipeline_artifacts}) in which
pipeline_artifacts.sql_revision.sql_candidates_after_revision is rewritten candidate by candidate. LENGTH and ORDER
never change, so the per-candidate writer labels (`sql_candidate_generators`) stay aligned. The revision
snapshot is never modified. Point selection at the new file with THESIS_SELECTION_INPUT.

Fixes run in this fixed order, each only when named in --fixes:
  1. distinct  qtree/distinct_fix.decide per candidate (COUNT(DISTINCT x), no distinct cue, x not a key, and the
               edit changes the result; learned on train, frozen). Decisions are memoised per SQL string per item.
  2. rules     a hook for the targeted tree rules (stage6_query_tree.qtree.rule_fix:apply_rules). --rules-module
               pkg.mod:func (or env THESIS_TREE_RULES_MODULE) names a function

                   apply_rules(item_view: ItemView, candidates: list[str], labels: list[str]) -> list[str]

               called once per item with the candidates AFTER the distinct fix. It must return a list of str of the
               same length (index i = the new SQL for candidate i; return it unchanged to keep it). Anything else is an
               item error. ItemView (below) gives question, evidence, linked schema (database_schema_after_schema_
               linking: {tables: {t: {columns: {c: {column_type, primary_key, ...}}}}}), retrieved values ({t: {c:
               [{value, distance}]}}), the full schema (database_schema, columns[...]["primary_key"]), database_path,
               database_id and execute(sql) -> common.sql.execute.ExecResult (sqlite, URI mode=ro, wall-clock timeout,
               10,000-row cap). labels[i] is the writer of candidate i ("?" when the snapshot has no labels).

Failure policy: any exception or invalid hook output on an item -> logged, the item keeps its ORIGINAL candidates
(no partial fix), the run continues. Per-item timeout (--item-timeout, default 600 s; 3600 s with the rules fix): the item keeps its original
candidates; its worker thread cannot be killed and finishes in the background (each SQL execution stops after
distinct_fix.EXEC_TIMEOUT_S = 30 s; a rules hook should bound its own LLM calls).
Resume: finished items are appended to <out>.progress.jsonl and reused on rerun (same input/fixes/hook); a complete
output snapshot is skipped. --force starts from scratch.
Outputs next to the snapshot: <out>.changes.jsonl (question_id, candidate_index, label, fix, before, after) and
<out>.summary.json (counts per fix, per label, per reason; item errors/timeouts).

Snapshot I/O: common/snapshot.py (raw JSON, no import of the base stages), so every field, including ones added by
earlier stages, survives verbatim.

  python -m stage6_query_tree.run_query_tree --run-dir D [--input sql_revision.snapshot] [--output ...]
      --fixes distinct[,rules] [--rules-module pkg.mod:func] [--db-root ROOT] [--workers N] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from common.snapshot import artifact, atomic_json, atomic_lines, items_file, iter_items, read_manifest, write_snapshot
from common.snapshot import qid as question_id
from common.snapshot import with_artifact
from stage6_query_tree.qtree import distinct_fix
from stage6_query_tree.qtree.match import QuestionSide
from common.sql.execute import ExecResult, execute

CANDIDATES_STAGE, CANDIDATES_FIELD = "sql_revision", "sql_candidates_after_revision"
LABELS_FIELD = "sql_candidate_generators"
FIX_ORDER = ("distinct", "rules")
RULES_ENV = "THESIS_TREE_RULES_MODULE"
DEFAULT_ITEM_TIMEOUT_S = 600.0
# With the rules hook an item makes one or more LLM calls of up to 600 s each (and waits for a free LLM slot), so the
# distinct-only budget would revert slow, heavily flagged items wholesale. Rules runs get a
# larger per-item budget and as many workers as the hook's LLM slots unless --item-timeout / --workers are given.
RULES_ITEM_TIMEOUT_S = 3600.0
RULES_WORKERS = 3
NO_LABEL = "?"


# ---------------------------------------------------------------- snapshot helpers (I/O lives in snapshot.py)
def snapshot_complete(path: str, expect_n: int) -> bool:
    """Manifest present and its num_items == items.jsonl lines == expect_n."""
    if not os.path.isfile(path):
        return False
    try:
        manifest = read_manifest(path)
        with open(items_file(path, manifest), encoding="utf-8") as f:
            n = sum(1 for line in f if line.strip())
    except (OSError, ValueError):
        return False
    return manifest.get("num_items") == n == expect_n


def sidecar(snapshot_path: str, suffix: str) -> str:
    """<dir>/<name without .snapshot><suffix>, e.g. sql_revision_tree.changes.jsonl."""
    base = snapshot_path[: -len(".snapshot")] if snapshot_path.endswith(".snapshot") else snapshot_path
    return base + suffix


def db_file(item: dict, db_root: Optional[str] = None) -> str:
    """The question's sqlite file; with db_root, <db_root>/<db_id>/<db_id>.sqlite (snapshot paths are cluster paths)."""
    inp = item["input"]
    if db_root:
        return os.path.join(db_root, inp["database_id"], inp["database_id"] + ".sqlite")
    return inp["database_path"]


def side_of(item: dict) -> Optional[QuestionSide]:
    """Question side for stage6_query_tree.qtree.match: question, evidence, linked schema and retrieved values."""
    inp = item.get("input") or {}
    if not inp.get("question"):
        return None
    return QuestionSide(inp["question"], inp.get("evidence") or "",
                        artifact(item, "schema_linking", "database_schema_after_schema_linking") or {},
                        artifact(item, "value_retrieval", "retrieved_values"))


def labels_of(item: dict, n: int) -> Tuple[List[str], bool]:
    """Writer label per candidate (`sql_candidate_generators`; looked up in sql_revision, sql_generation,
    then the input record). (labels, found); "?" for every candidate when absent or misaligned."""
    for value in (artifact(item, "sql_revision", LABELS_FIELD), artifact(item, "sql_generation", LABELS_FIELD),
                  (item.get("input") or {}).get(LABELS_FIELD), item.get(LABELS_FIELD)):
        if isinstance(value, list) and len(value) == n:
            return [NO_LABEL if v is None else str(v) for v in value], True
    return [NO_LABEL] * n, False


# ---------------------------------------------------------------- rules hook interface
@dataclass(frozen=True)
class ItemView:
    """What a rules hook sees for one question. execute() is read-only (sqlite URI mode=ro) with a timeout."""
    question_id: int
    question: str
    evidence: str
    linked_schema: Dict[str, Any]
    retrieved_values: Optional[Dict[str, Any]]
    database_schema: Dict[str, Any]
    database_path: str
    database_id: str
    exec_timeout_s: float = distinct_fix.EXEC_TIMEOUT_S

    def execute(self, sql: str) -> ExecResult:
        return execute(self.database_path, sql, timeout_s=self.exec_timeout_s)


def item_view(item: dict, db_root: Optional[str] = None) -> ItemView:
    inp = item["input"]
    return ItemView(question_id(item), inp.get("question") or "", inp.get("evidence") or "",
                    artifact(item, "schema_linking", "database_schema_after_schema_linking") or {},
                    artifact(item, "value_retrieval", "retrieved_values"),
                    inp.get("database_schema") or {}, db_file(item, db_root), inp.get("database_id") or "")


RulesFn = Callable[[ItemView, List[str], List[str]], List[str]]


def load_rules(spec: str) -> RulesFn:
    """'pkg.module:function' -> the function (fails fast, before any item runs)."""
    mod, sep, fn = spec.partition(":")
    if not sep or not mod or not fn:
        raise ValueError(f"rules module must look like 'pkg.module:function', got {spec!r}")
    func = getattr(importlib.import_module(mod), fn)
    if not callable(func):
        raise ValueError(f"{spec} is not callable")
    return func


# ---------------------------------------------------------------- fixes
def fix_distinct(view: ItemView, side: Optional[QuestionSide], candidates: List[str],
                 pk_cache: dict) -> Tuple[List[str], Counter]:
    memo: Dict[str, Tuple[Optional[str], str]] = {}
    out, reasons = [], Counter()
    for sql in candidates:
        if sql not in memo:
            if side is None:
                memo[sql] = (None, "no_question_side")
            elif not (sql or "").strip():
                memo[sql] = (None, "empty_sql")
            else:
                memo[sql] = distinct_fix.decide(sql, side, view.database_path, pk_cache)
        new, why = memo[sql]
        reasons[why] += 1
        out.append(new if new is not None else sql)
    return out, reasons


def fix_rules(rules: RulesFn, view: ItemView, candidates: List[str],
              labels: List[str]) -> Tuple[List[str], Counter]:
    """The hook's edits, except that an edit which fails to execute while the original executed is undone (a broken
    edit must not replace a working candidate and drop its writer from the vote)."""
    out = rules(view, list(candidates), list(labels))
    if not isinstance(out, list) or len(out) != len(candidates) or not all(isinstance(s, str) for s in out):
        raise ValueError(f"rules hook must return list[str] of length {len(candidates)}, got "
                         f"{type(out).__name__} of length {len(out) if hasattr(out, '__len__') else '?'}")
    runs: Dict[str, bool] = {}

    def runs_ok(sql: str) -> bool:
        if sql not in runs:
            runs[sql] = view.execute(sql).ok
        return runs[sql]

    kept, reasons = [], Counter()
    for before, after in zip(candidates, out):
        if after == before:
            kept.append(after)
        elif not runs_ok(after) and runs_ok(before):
            reasons["edit_fails_to_execute_reverted"] += 1
            kept.append(before)
        else:
            reasons["edit_kept"] += 1
            kept.append(after)
    return kept, reasons


@dataclass
class Config:
    fixes: Tuple[str, ...]
    rules: Optional[RulesFn] = None
    db_root: Optional[str] = None
    item_timeout_s: float = DEFAULT_ITEM_TIMEOUT_S
    pk_cache: dict = field(default_factory=dict)


def process_item(item: dict, cfg: Config) -> dict:
    """One item -> {question_id, candidates, changes, reasons, labels_found}; raises on any failure."""
    qid = question_id(item)
    original = list(artifact(item, CANDIDATES_STAGE, CANDIDATES_FIELD) or [])
    labels, found = labels_of(item, len(original))
    view = item_view(item, cfg.db_root)
    current, changes, reasons = list(original), [], {}
    for fix in FIX_ORDER:
        if fix not in cfg.fixes or not current:
            continue
        if fix == "distinct":
            new, why = fix_distinct(view, side_of(item), current, cfg.pk_cache)
            reasons[fix] = dict(why)
        else:
            new, why = fix_rules(cfg.rules, view, current, labels)
            reasons[fix] = dict(why)
        changes += [{"question_id": qid, "candidate_index": i, "label": labels[i], "fix": fix,
                     "before": before, "after": after}
                    for i, (before, after) in enumerate(zip(current, new)) if after != before]
        current = new
    return {"question_id": qid, "candidates": current, "changes": changes, "reasons": reasons,
            "labels_found": found}


def call_with_timeout(fn: Callable[[], Any], timeout_s: float) -> Any:
    """fn() in a daemon thread; TimeoutError after timeout_s (the thread is abandoned, not killed)."""
    box: Dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # handed to the caller
            box["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise TimeoutError(f"item exceeded {timeout_s:.0f} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def safe_process(item: dict, cfg: Config) -> dict:
    """process_item with isolation: on error/timeout the item keeps its original candidates."""
    qid = question_id(item)
    try:
        out = call_with_timeout(lambda: process_item(item, cfg), cfg.item_timeout_s)
        return {**out, "status": "ok"}
    except TimeoutError as exc:
        status, err = "timeout", str(exc)
    except Exception as exc:
        status, err = "error", f"{type(exc).__name__}: {exc}"
    original = list(artifact(item, CANDIDATES_STAGE, CANDIDATES_FIELD) or [])
    return {"question_id": qid, "candidates": original, "changes": [], "reasons": {}, "labels_found": False,
            "status": status, "error": err}


# ---------------------------------------------------------------- run
# Environment the rules hook reads that changes its OUTPUT (cache/log/run-id paths do not): switch values, and the
# CONTENT of the rules and model-settings files, so a resume never mixes items made under different settings.
HOOK_ENV_VALUES = ("THESIS_TREE_RULES_FACTS", "THESIS_TREE_RULES_LOOP", "THESIS_TREE_RULES_SAMPLES")
HOOK_ENV_FILES = ("THESIS_TREE_RULES_FILE", "THESIS_MODEL_SETTINGS")


def _file_md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def hook_settings(fixes: Tuple[str, ...]) -> Dict[str, str]:
    if "rules" not in fixes:
        return {}
    out = {k: os.environ.get(k, "") for k in HOOK_ENV_VALUES}
    for k in HOOK_ENV_FILES:
        path = os.environ.get(k, "")
        out[k] = _file_md5(path) if path and os.path.isfile(path) else path
    return out


def signature(input_path: str, fixes: Tuple[str, ...], rules_spec: Optional[str], db_root: Optional[str]) -> str:
    key = json.dumps([os.path.abspath(input_path), list(fixes), rules_spec or "", db_root or "",
                      hook_settings(fixes)], sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()


def load_progress(path: str, sig: str) -> Dict[int, dict]:
    done: Dict[int, dict] = {}
    if not os.path.isfile(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:  # a line cut by a crash
                continue
            if row.get("sig") == sig and row.get("status") == "ok":
                done[int(row["question_id"])] = row
    return done


def summarize(outcomes: List[dict], fixes: Tuple[str, ...], meta: dict) -> dict:
    changed_cands, changed_items, by_label = Counter(), Counter(), {}
    reasons: Dict[str, Counter] = {}
    for o in outcomes:
        for c in o["changes"]:
            changed_cands[c["fix"]] += 1
            by_label.setdefault(c["fix"], Counter())[c["label"]] += 1
        for fix in {c["fix"] for c in o["changes"]}:
            changed_items[fix] += 1
        for fix, why in (o.get("reasons") or {}).items():
            reasons.setdefault(fix, Counter()).update(why)
    status = Counter(o["status"] for o in outcomes)
    failures = [{"question_id": o["question_id"], "status": o["status"], "error": o.get("error")}
                for o in outcomes if o["status"] != "ok"]
    return {**meta, "fixes": list(fixes), "n_items": len(outcomes),
            "n_candidates": sum(len(o["candidates"]) for o in outcomes),
            "items_with_labels": sum(1 for o in outcomes if o.get("labels_found")),
            "status": dict(status), "changed_candidates": dict(changed_cands), "changed_items": dict(changed_items),
            "changed_by_label": {k: dict(v) for k, v in by_label.items()},
            "reasons": {k: dict(v) for k, v in reasons.items()}, "failures": failures}


def run(input_path: str, output_path: str, fixes: Tuple[str, ...], rules_spec: Optional[str] = None,
        db_root: Optional[str] = None, workers: int = 4, item_timeout_s: float = DEFAULT_ITEM_TIMEOUT_S,
        force: bool = False) -> dict:
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("output must differ from the input snapshot (the revision snapshot is never modified)")
    unknown = set(fixes) - set(FIX_ORDER)
    if unknown or not fixes:
        raise ValueError(f"--fixes must be a non-empty subset of {FIX_ORDER}, got {fixes}")
    rules = None
    if "rules" in fixes:
        if not rules_spec:
            raise ValueError(f"fix 'rules' needs --rules-module or env {RULES_ENV}")
        rules = load_rules(rules_spec)
    fixes = tuple(f for f in FIX_ORDER if f in fixes)
    manifest, items = read_manifest(input_path), list(iter_items(input_path))
    summary_path, changes_path = sidecar(output_path, ".summary.json"), sidecar(output_path, ".changes.jsonl")
    progress_path = sidecar(output_path, ".progress.jsonl")
    sig = signature(input_path, fixes, rules_spec, db_root)
    if not force and snapshot_complete(output_path, len(items)) and os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            previous = json.load(f)
        if previous.get("signature") not in (None, sig):
            raise ValueError(f"{output_path} was made with other inputs/settings (signature {previous.get('signature')}"
                             f" != {sig}); use another --output or --force")
        return {**previous, "skipped": "output snapshot already complete (use --force to redo)"}
    if force and os.path.exists(progress_path):
        os.remove(progress_path)
    done = load_progress(progress_path, sig)
    cfg = Config(fixes, rules, db_root, item_timeout_s)
    todo = [it for it in items if question_id(it) not in done]
    lock = threading.Lock()
    with open(progress_path, "a", encoding="utf-8") as prog:
        def work(item: dict) -> dict:
            out = safe_process(item, cfg)
            with lock:
                prog.write(json.dumps({**out, "sig": sig}, ensure_ascii=False) + "\n")
                prog.flush()
            return out
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for out in pool.map(work, todo):
                done[out["question_id"]] = out
    outcomes = [done[question_id(it)] for it in items]
    new_items = [with_artifact(it, CANDIDATES_STAGE, {CANDIDATES_FIELD: o["candidates"]})
                 for it, o in zip(items, outcomes)]
    atomic_lines(changes_path, [c for o in outcomes for c in o["changes"]])
    summary = summarize(outcomes, fixes, {"input": os.path.abspath(input_path), "output": os.path.abspath(output_path),
                                          "rules_module": rules_spec, "db_root": db_root,
                                          "signature": sig, "hook_settings": hook_settings(fixes)})
    atomic_json(summary_path, summary)
    write_snapshot(output_path, manifest, new_items)
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--input", default="sql_revision.snapshot", help="relative to --run-dir unless absolute")
    ap.add_argument("--output", default="sql_revision_tree.snapshot", help="relative to --run-dir unless absolute")
    ap.add_argument("--fixes", required=True, help="comma list of: " + ",".join(FIX_ORDER))
    ap.add_argument("--rules-module", default=os.environ.get(RULES_ENV), help="pkg.module:function")
    ap.add_argument("--db-root", default=None, help="remap database_path to <db-root>/<db_id>/<db_id>.sqlite")
    ap.add_argument("--workers", type=int, default=None, help=f"default 4, or {RULES_WORKERS} with the rules fix")
    ap.add_argument("--item-timeout", type=float, default=None,
                    help=f"seconds per item; default {DEFAULT_ITEM_TIMEOUT_S:.0f}, or {RULES_ITEM_TIMEOUT_S:.0f} with rules")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    fixes = tuple(f.strip() for f in a.fixes.split(",") if f.strip())
    with_rules = "rules" in fixes
    workers = a.workers if a.workers is not None else (RULES_WORKERS if with_rules else 4)
    item_timeout = a.item_timeout if a.item_timeout is not None else (
        RULES_ITEM_TIMEOUT_S if with_rules else DEFAULT_ITEM_TIMEOUT_S)
    summary = run(os.path.join(a.run_dir, a.input), os.path.join(a.run_dir, a.output), fixes, a.rules_module,
                  a.db_root, workers, item_timeout, a.force)
    print(json.dumps({k: summary.get(k) for k in ("output", "n_items", "status", "changed_candidates",
                                                    "changed_items", "skipped") if k in summary}, indent=1))


if __name__ == "__main__":
    main()
