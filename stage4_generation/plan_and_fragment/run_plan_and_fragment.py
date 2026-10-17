"""Our extra SQL writer (the literal head) on a run of the base stages, between their step 6 (sql_generation) and
step 7 (sql_revision); its candidates are appended to each item so the base stages' revision and selection treat
them like their own.

Flow (one run dir = the run folder):
  1. guards: <run>/sql_generation.snapshot exists and its stage left no checkpoint; sql_revision, sql_revision_tree
     (stage6_query_tree) and sql_selection have NOT started (snapshot or <stage>.artifacts checkpoint) -
     the base stages would resume them from the OLD candidate lists (a revision checkpoint holds lists of the old length,
     misaligned with the merged ones). --force moves them aside (renamed <name>.stale_<UTC time>, nothing deleted) so
     they rerun on the merged pool. With nothing left to append the run is a no-op and the guards are skipped.
  2. every item must carry sql_candidate_generators (THESIS_LABELS=1 patch, one label per sql_candidates entry:
     "dc"/"skeleton"/"icl"); missing or misaligned labels raise.
  3. adapter (snapshot item -> what stage4_generation.plan_and_fragment reads): Question(question_id, database_id, question, evidence,
     gold blanked); linking row {question_id, columns: ["table.column", ...]} = every column of
     schema_linking.database_schema_after_schema_linking EXCEPT is_unuseful ones (the base stages drop those from
     their own prompts, so our writers see what the base stages' writers saw; describe.py re-adds keys of shown
     tables);
     values row {question_id, hits} = value_retrieval.retrieved_values through stage4_generation.writer_support.literal_values (distance <= 0.3,
     5 per column) - passed to the literal writer only when THESIS_PROMPT_FIXES names literal_values.
  4. writer: stage4_generation.plan_and_fragment.stage.GenerationStage, out dir <run>/thesis_extra/<writer>/generation/
     (resumable per (question, writer); all-error questions are retried on the next run). literal = the literal head
     (no demos: messages.py literal_messages), 4 samples + its COUNT(DISTINCT) re-ask. Only items not
     yet labelled with the writer are run, minus "exhausted" ones (request finished, not retried by the stage, but no
     usable SQL).
  5. merge: usable rows (SQL, no error, not truncated) appended to sql_candidates, the writer's name appended to
     sql_candidate_generators once per candidate; an item already carrying the writer's label is skipped, so a rerun
     never double-appends. The generation snapshot is rewritten in place (atomic) after a one-time backup
     sql_generation.snapshot.data/items.jsonl.pre_extra.
  6. <run>/thesis_extra/extra_generation.json: per-writer counts, failures, error gate, labels now in the snapshot.

LLM: common.llm_client.LLMClient with the --settings file (model, temperature, top_p, max_tokens, reasoning effort,
request timeout; retries on connection/5xx only), base URL from --base-url or TP_BASE_URL, --workers requests in
flight; disk cache key = md5(model, temperature, top_p, effort, max_tokens, n, run_id, messages). A question is
bounded by its requests: literal <= 2 sequential calls.
A failed call becomes error rows: logged, nothing appended for that writer on that item, the run continues; exit 3
when more than 5 % of a writer's questions failed (merge still done; a rerun retries them).

Usage:
  python -m stage4_generation.plan_and_fragment.run_plan_and_fragment --run-dir <run folder> --prep-root <prep>/<split> \
      --settings <model settings yaml> [--writers literal] [--n 4] [--workers 8] [--db-root <dir>] [--split dev|train] \
      [--force] [--dry-run | --mock] [--prewrite]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import time
from collections import Counter, defaultdict
from typing import Callable, Optional

from common.data.bird import Question, db_path
from common import snapshot as S

log = logging.getLogger("stage4_generation.plan_and_fragment.run_plan_and_fragment")

WRITERS = ("literal",)
DEFAULT_N = 4  # user decision 2026-10-07: "4 each"
DEFAULT_WORKERS = 8  # LLM requests in flight against the 120B server
GEN_STAGE = "sql_generation"
DOWNSTREAM = ("sql_revision", "sql_revision_tree", "sql_selection")  # sql_revision_tree: tree_stage.py
LABELS = "sql_candidate_generators"
BACKUP_SUFFIX = ".pre_extra"
OUT_DIR = "thesis_extra"
SUMMARY = "extra_generation.json"
EXIT_REFUSED = 2
EXIT_ERROR_GATE = 3
VALUE_WRITERS = ("literal",)  # B2: only the literal head reads retrieved values (prompt fix literal_values)


class Refused(RuntimeError):
    """The run dir is not in a state where appending candidates is safe."""


@dataclasses.dataclass(frozen=True)
class Config:
    run_dir: str
    prep_root: str
    writers: tuple[str, ...] = WRITERS
    n: int = DEFAULT_N
    workers: int = DEFAULT_WORKERS
    db_root: Optional[str] = None
    split: Optional[str] = None
    force: bool = False
    dry_run: bool = False
    chunk_size: int = 30


# ---------------------------------------------------------------- adapter
def item_question(item: dict) -> Question:
    inp = item["input"]
    return Question(question_id=int(inp["question_id"]), db_id=inp["database_id"], question=inp["question"],
                    evidence=inp.get("evidence") or "", gold_sql="", difficulty=inp.get("difficulty") or None)


def linked_columns(item: dict) -> list[str]:
    """"table.column" for every linked column the base stages' writers saw (is_unuseful columns dropped)."""
    schema = S.artifact(item, "schema_linking", "database_schema_after_schema_linking") or {}
    out = []
    for table, tinfo in (schema.get("tables") or {}).items():
        for col, cinfo in ((tinfo or {}).get("columns") or {}).items():
            if not (cinfo or {}).get("is_unuseful", False):
                out.append(f"{table}.{col}")
    return out


def linking_row(item: dict) -> dict:
    cols = linked_columns(item)
    return {"question_id": S.qid(item), "columns": cols, "n_columns": len(cols)}


def values_row(item: dict) -> dict:
    from stage4_generation.writer_support.literal_values import hits_from_retrieved
    return {"question_id": S.qid(item),
            "hits": hits_from_retrieved(S.artifact(item, "value_retrieval", "retrieved_values"))}


# ---------------------------------------------------------------- guards
def check_labels(items: list[dict]) -> None:
    """Every item with candidates carries one generator label per candidate (THESIS_LABELS=1)."""
    for it in items:
        sqls, labels = S.artifact(it, GEN_STAGE, "sql_candidates"), S.artifact(it, GEN_STAGE, LABELS)
        if sqls is None:
            continue
        if labels is None:
            raise ValueError(f"question {S.qid(it)}: {LABELS} is missing; rerun the base stages' step 6 with THESIS_LABELS=1")
        if len(labels) != len(sqls):
            raise ValueError(f"question {S.qid(it)}: {len(sqls)} candidates but {len(labels)} labels")


def guard_run_dir(run_dir: str, force: bool, dry_run: bool) -> list[str]:
    """Refuse unsafe states; with force, move downstream stages aside. Returns the moved paths."""
    gen = S.stage_snapshot(run_dir, GEN_STAGE)
    if not os.path.exists(gen):
        raise Refused(f"{gen} not found: run the base stages through step 6 (sql_generation) first")
    if S.has_checkpoint(gen):
        raise Refused(f"{S.artifacts_dir(gen)} holds an unfinished sql_generation checkpoint: let step 6 finish first")
    started = [st for st in DOWNSTREAM if S.stage_started(run_dir, st)]
    if not started:
        return []
    if not force:
        raise Refused(f"{', '.join(started)} already started in {run_dir}: the base stages would resume them from the old "
                      "candidate lists and never revise the appended ones. Rerun with --force to move them aside.")
    if dry_run:
        log.warning("dry run: would move aside %s", started)
        return []
    tag = "stale_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    moved = [p for st in started for p in S.move_aside(run_dir, st, tag)]
    log.warning("--force: moved aside %s", moved)
    return moved


def resolve_db_root(items: list[dict], db_root: Optional[str]) -> str:
    """--db-root, else derived from the items' database_path (<root>/<db>/<db>.sqlite); every database must exist."""
    if not db_root:
        paths = {os.path.dirname(os.path.dirname(it["input"]["database_path"])) for it in items}
        if len(paths) != 1 or not os.path.isabs(next(iter(paths))):
            raise Refused(f"cannot derive the database root from database_path ({sorted(paths)[:3]}): pass --db-root")
        db_root = next(iter(paths))
    missing = sorted({it["input"]["database_id"] for it in items
                      if not os.path.exists(db_path(db_root, it["input"]["database_id"]))})
    if missing:
        raise Refused(f"databases missing under {db_root}: {missing[:10]}")
    return db_root


def prep_problems(prep_root: str, db_root: str, db_ids: set[str]) -> list[str]:
    """Missing prep outputs, each with the command that builds it."""
    no_cols = sorted(d for d in db_ids if not os.path.exists(os.path.join(prep_root, d, "columns.json")))
    out = []
    if no_cols:
        out.append(f"columns.json missing for {no_cols}: python -m stage0_dataset.prep.build --db-root {db_root} "
                   f"--out-root {prep_root} --db-ids {' '.join(no_cols)}")
    return out


def resolve_split(run_dir: str, split: Optional[str]) -> str:
    """dev or train: from --split, else the snapshot manifest's dataset_config.split."""
    if not split:
        cfg = S.read_manifest(S.stage_snapshot(run_dir, GEN_STAGE)).get("dataset_config") or {}
        split = cfg.get("split")
    if split not in ("dev", "train"):
        raise Refused(f"split must be dev or train (got {split!r}); pass --split")
    return split


# ---------------------------------------------------------------- merge
def usable_sqls(rows: list[dict]) -> dict[int, list[str]]:
    """question_id -> SQLs in sample order (rows with SQL, no error, not truncated; as src/b2/hooks.py)."""
    out: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for r in rows:
        if (r.get("sql") or "").strip() and not r.get("error") and not r.get("truncated"):
            out[int(r["question_id"])].append((int(r.get("sample") or 0), r["sql"]))
    return {q: [s for _, s in sorted(v)] for q, v in out.items()}


def merge_item(item: dict, sqls_by_writer: dict[str, list[str]]) -> tuple[dict, dict[str, int]]:
    """(new item, {writer: candidates added}); writers already in the labels are skipped."""
    g = S.artifact(item, GEN_STAGE)
    sqls, labels = list(g.get("sql_candidates") or []), list(g.get(LABELS) or [])
    added = {}
    for writer, extra in sqls_by_writer.items():
        if writer in labels or not extra:
            continue
        sqls, labels = sqls + list(extra), labels + [writer] * len(extra)
        added[writer] = len(extra)
    if not added:
        return item, {}
    return S.with_artifact(item, GEN_STAGE, {"sql_candidates": sqls, LABELS: labels}), added


def merge_all(items: list[dict], rows_by_writer: dict[str, list[dict]]) -> tuple[list[dict], dict]:
    """(new items, per-writer merge stats). Items without base candidates are passed through unchanged."""
    usable = {w: usable_sqls(rows) for w, rows in rows_by_writer.items()}
    out = []
    stats = {w: {"candidates_added": 0, "items_added": 0, "items_skipped_already_merged": 0,
                 "qids_without_candidates": []} for w in rows_by_writer}
    for it in items:
        if S.artifact(it, GEN_STAGE, "sql_candidates") is None:
            out.append(it)
            continue
        q, labels = S.qid(it), S.artifact(it, GEN_STAGE, LABELS) or []
        new, got = merge_item(it, {w: usable[w].get(q, []) for w in rows_by_writer})
        out.append(new)
        for w in rows_by_writer:
            if w in labels:
                stats[w]["items_skipped_already_merged"] += 1
            elif w in got:
                stats[w]["candidates_added"] += got[w]
                stats[w]["items_added"] += 1
            else:
                stats[w]["qids_without_candidates"].append(q)
    return out, stats


def label_totals(items: list[dict]) -> dict:
    """Candidates per label over the snapshot, and items carrying each label."""
    cands, carrying = Counter(), Counter()
    for it in items:
        labels = S.artifact(it, GEN_STAGE, LABELS) or []
        cands.update(labels)
        carrying.update(set(labels))
    return {lab: {"candidates": cands[lab], "items": carrying[lab]} for lab in sorted(cands)}


# ---------------------------------------------------------------- writers
def candidates_path(run_dir: str, writer: str) -> str:
    return os.path.join(run_dir, OUT_DIR, writer, "generation", "candidates.jsonl")


def exhausted_qids(run_dir: str, writer: str, n: int) -> set[int]:
    """Questions whose writer request finished (n samples, not all errors, so the stage will not retry it) without
    one usable SQL: nothing can be appended, and they must not keep the item pending forever."""
    from stage4_generation.plan_and_fragment.store import read_jsonl
    groups: dict[int, list[dict]] = defaultdict(list)
    for r in read_jsonl(candidates_path(run_dir, writer)):
        if r.get("generator") == writer and "question_id" in r:
            groups[int(r["question_id"])].append(r)
    usable = usable_sqls([r for grp in groups.values() for r in grp])
    return {q for q, grp in groups.items() if sorted(r.get("sample") for r in grp) == list(range(n))
            and not all(r.get("error") for r in grp) and not usable.get(q)}


def pending_items(items: list[dict], writer: str, skip: frozenset[int] = frozenset()) -> list[dict]:
    """Items with base candidates that do not carry the writer's label yet (minus `skip` question ids)."""
    return [it for it in items if S.artifact(it, GEN_STAGE, "sql_candidates") is not None
            and writer not in (S.artifact(it, GEN_STAGE, LABELS) or []) and S.qid(it) not in skip]


def run_writer(writer: str, items: list[dict], cfg: Config, client, db_root: str, split: str,
               render_schema: Optional[Callable] = None) -> tuple[list[dict], dict]:
    """Run one writer over `items` into <run>/thesis_extra/<writer>/generation; (rows of these items, stage stats)."""
    from stage4_generation.writer_support.prompt_fixes import fix_on
    from stage4_generation.plan_and_fragment.stage import GenerationStage
    from stage4_generation.plan_and_fragment.store import read_jsonl

    root = os.path.join(cfg.run_dir, OUT_DIR, writer)
    values = ({S.qid(it): values_row(it) for it in items}
              if writer in VALUE_WRITERS and fix_on("literal_values") else None)
    stage = GenerationStage(
        client, os.path.join(root, "generation"), db_root, cfg.prep_root, {S.qid(it): linking_row(it) for it in items},
        values=values, generators=(writer,), n=cfg.n, chunk_size=cfg.chunk_size, render_schema=render_schema,
        extra_stats={"source": "extra_generation", "split": split, "values": values is not None})
    stats = stage.run([item_question(it) for it in items])
    wanted = {S.qid(it) for it in items}
    rows = [r for r in read_jsonl(stage.cand_path) if int(r.get("question_id", -1)) in wanted]
    return rows, stats


def _stage_summary(stats: dict, writer: str) -> dict:
    g = (stats.get("by_generator") or {}).get(writer, {})
    keep = ("questions", "candidates", "errors", "error_kinds", "truncated", "no_sql")
    return {"error_gate": stats.get("error_gate"), **{k: g[k] for k in keep if k in g}}


# ---------------------------------------------------------------- run
def run(cfg: Config, client=None, render_schema: Optional[Callable] = None) -> dict:
    """Guards, writers, merge, summary; returns the summary (also <run>/thesis_extra/extra_generation.json)."""
    bad = [w for w in cfg.writers if w not in WRITERS]
    if bad or not cfg.writers or cfg.n < 1:
        raise ValueError(f"writers must be a subset of {WRITERS} and n >= 1 (got {cfg.writers}, n={cfg.n})")
    gen = S.stage_snapshot(cfg.run_dir, GEN_STAGE)
    if not os.path.exists(gen):
        raise Refused(f"{gen} not found: run the base stages through step 6 (sql_generation) first")
    items = S.read_items(gen)
    check_labels(items)
    split = resolve_split(cfg.run_dir, cfg.split)
    exhausted = {w: frozenset(exhausted_qids(cfg.run_dir, w, cfg.n)) for w in cfg.writers}
    pending = {w: pending_items(items, w, exhausted[w]) for w in cfg.writers}
    plan = {w: len(pending[w]) for w in cfg.writers}
    todo = [it for w in cfg.writers for it in pending[w]]
    log.info("extra generation (%s): %d items; pending per writer %s; n=%d", split, len(items), plan, cfg.n)
    if not todo:  # everything merged already: nothing to append, so a started revision is still consistent
        log.info("every item already carries %s; nothing to do", list(cfg.writers))
        return {"dry_run": cfg.dry_run, "split": split, "items": len(items), "pending": plan, "nothing_to_do": True,
                "exhausted": {w: sorted(q)[:200] for w, q in exhausted.items()},
                "labels_in_snapshot": label_totals(items)}
    db_root = resolve_db_root(todo, cfg.db_root)
    problems = prep_problems(cfg.prep_root, db_root, {it["input"]["database_id"] for it in todo})
    if problems:
        raise Refused("prep missing (build it on CPU first):\n  " + "\n  ".join(problems))
    moved = guard_run_dir(cfg.run_dir, cfg.force, cfg.dry_run)
    if cfg.dry_run:
        return {"dry_run": True, "split": split, "items": len(items), "pending": plan, "db_root": db_root}
    rows, stage_stats = {}, {}
    for w in cfg.writers:
        if pending[w]:
            rows[w], st = run_writer(w, pending[w], cfg, client, db_root, split, render_schema)
            stage_stats[w] = _stage_summary(st, w)
    backup = None
    merged, merge_stats = merge_all(items, rows) if rows else (items, {})
    if any(s["candidates_added"] for s in merge_stats.values()):
        backup = S.backup_items(gen, BACKUP_SUFFIX)
        S.rewrite_items(gen, merged)
    summary = {
        "run_dir": os.path.abspath(cfg.run_dir), "split": split, "writers": list(cfg.writers), "n": cfg.n,
        "items": len(items), "pending_at_start": plan, "merge": merge_stats, "writer_stages": stage_stats,
        "exhausted_before_run": {w: sorted(q)[:200] for w, q in exhausted.items()},
        "labels_in_snapshot": label_totals(merged), "backup": backup, "moved_aside": moved,
        "error_gate_passed": all((s.get("error_gate") or {}).get("passed", True) for s in stage_stats.values()),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    S.atomic_json(os.path.join(cfg.run_dir, OUT_DIR, SUMMARY), summary)
    log.info("extra generation done: %s", {w: s["candidates_added"] for w, s in merge_stats.items()})
    return summary


def prewrite(cfg: Config, client=None) -> dict:
    """Our writers on the linked questions (<run>/schema_linking.snapshot) WHILE the base stages' writers run, so
    all of a question's writers work in the same generation step. Candidates only, no merge: the normal run after
    the base stages' writers finish resumes from the same per-question files and LLM cache (no new calls for
    finished questions) and merges as before."""
    src = S.stage_snapshot(cfg.run_dir, "schema_linking")
    if not os.path.exists(src):
        raise Refused(f"{src} not found: run the column finding first")
    items = S.read_items(src)
    split = cfg.split or "dev"
    db_root = resolve_db_root(items, cfg.db_root)
    problems = prep_problems(cfg.prep_root, db_root, {it["input"]["database_id"] for it in items})
    if problems:
        raise Refused("prep missing (build it on CPU first):\n  " + "\n  ".join(problems))
    if cfg.dry_run:
        return {"dry_run": True, "prewrite": True, "split": split, "items": len(items), "db_root": db_root}
    stats = {}
    for w in cfg.writers:
        _, st = run_writer(w, items, cfg, client, db_root, split)
        stats[w] = _stage_summary(st, w)
    log.info("prewrite done: %s", {w: s.get("candidates") for w, s in stats.items()})
    return {"prewrite": True, "split": split, "items": len(items), "writer_stages": stats}


# ---------------------------------------------------------------- clients
class MockClient:
    """No server: every request answers `SELECT 1` (fenced) for each of its n samples. Smoke tests only."""

    def __init__(self, sql: str = "SELECT 1") -> None:
        self.sql = sql
        self.requests: list[dict] = []

    def chat_many(self, reqs: list[dict]) -> list[list]:
        from common.llm_client.client import Completion
        self.requests = self.requests + list(reqs)
        body = f"```sql\n{self.sql}\n```"
        return [[Completion(body, "", "stop", 0, 0, 0.0, False, False, False) for _ in range(r.get("n", 1))]
                for r in reqs]


def normalize_base_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def make_client(args: argparse.Namespace):
    if args.mock:
        return MockClient()
    from common.llm_client.client import LLMClient, load_settings
    base = normalize_base_url(args.base_url or "")
    if not base:
        raise Refused("TP_BASE_URL is not set (and no --base-url): no LLM server")
    cache_dir = args.cache_dir or os.path.join(args.run_dir, OUT_DIR, "llm_cache")
    run_id = args.run_id or os.path.basename(os.path.abspath(args.run_dir))
    return LLMClient(load_settings(args.settings), base, cache_dir, run_id, max_concurrency=args.workers)


# ---------------------------------------------------------------- CLI
def _writers(text: str) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(t.strip() for t in text.split(",") if t.strip()))
    if not names or any(n not in WRITERS for n in names):
        raise argparse.ArgumentTypeError(f"--writers must be a subset of {','.join(WRITERS)}, got {text!r}")
    return names


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="the run folder of the base stages (holds sql_generation.snapshot)")
    ap.add_argument("--prep-root", required=True,
                    help="stage0_dataset.prep output for the split (<db>/columns.json)")
    ap.add_argument("--writers", type=_writers, default=WRITERS)
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="samples per writer per question")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="LLM requests in flight")
    ap.add_argument("--db-root", default=None, help="default: derived from the items' database_path")
    ap.add_argument("--split", choices=("dev", "train"), default=None, help="default: the snapshot's dataset_config")
    ap.add_argument("--force", action="store_true", help="move started sql_revision/sql_selection aside and proceed")
    ap.add_argument("--chunk-size", type=int, default=30, help="questions per checkpointed batch")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="checks and plan only: no LLM call, nothing written")
    mode.add_argument("--mock", action="store_true", help="fake LLM answering SELECT 1 (smoke test, no server)")
    ap.add_argument("--prewrite", action="store_true",
                    help="write candidates from schema_linking.snapshot while the base stages' writers run; no merge")
    ap.add_argument("--settings", required=True, help="model settings YAML for common.llm_client")
    ap.add_argument("--base-url", default=os.environ.get("TP_BASE_URL", ""))
    ap.add_argument("--cache-dir", default=os.environ.get("TP_CACHE_DIR"), help="default <run>/thesis_extra/llm_cache")
    ap.add_argument("--run-id", default=None, help="LLM cache namespace (default: the run dir's name)")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    cfg = Config(run_dir=args.run_dir, prep_root=args.prep_root, writers=args.writers, n=args.n, workers=args.workers,
                 db_root=args.db_root, split=args.split, force=args.force, dry_run=args.dry_run,
                 chunk_size=args.chunk_size)
    try:
        client = None if args.dry_run else make_client(args)
        summary = prewrite(cfg, client=client) if args.prewrite else run(cfg, client=client)
    except (Refused, ValueError) as exc:
        log.error("refused: %s", exc)
        return EXIT_REFUSED
    print(json.dumps(summary, indent=1))
    if not summary.get("error_gate_passed", True):
        log.error("error gate failed for a writer (> 5 %% of its questions); rerun to retry them")
        return EXIT_ERROR_GATE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
