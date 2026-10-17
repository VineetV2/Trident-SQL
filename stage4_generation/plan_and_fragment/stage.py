"""B1 generation stage with the literal generator; n samples per question.

literal: one request (n), then one n=1 re-ask per sample that uses COUNT(DISTINCT) without a distinct-values cue
(see literal.py).

Outputs in <out_dir> (= <run_dir>/generation):
  candidates.jsonl     one line per candidate: {question_id, db_id, generator, sample, sql,
                       finish_reason, truncated, empty, error, completion_tokens, prompt_tokens,
                       content, literal_reasked, literal_original_sql, literal_reask_error}
  stats.json           counts of errors, truncations, empties (recomputed each run)

Prompt sources are documented in prompts.py.
Gold SQL of the questions being answered is blanked on entry (dataclasses.replace) so no prompt
can read it.
"""
from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional

from common.data.bird import Question, db_path
from stage4_generation.plan_and_fragment import literal as LIT
from stage4_generation.plan_and_fragment import messages as M
from stage4_generation.plan_and_fragment import store
from common.sql.extract import extract_sql

log = logging.getLogger(__name__)

GENERATORS = ("literal",)
CANDIDATES = "candidates.jsonl"
STATS = "stats.json"


def _default_render_schema(prep_dir: str, db_file: str, columns=None, value_hits=None) -> str:
    from stage4_generation.plan_and_fragment.describe import render_described_schema  # lazy: sibling module
    return render_described_schema(prep_dir, db_file, columns=columns, value_hits=value_hits)


@dataclass(frozen=True)
class Context:
    question: Question
    schema: str
    schema_error: Optional[str]
    columns: Optional[list[str]]
    hits: Optional[list[dict]]


def blank_gold(questions: list[Question]) -> list[Question]:
    """The generator never sees the target's gold SQL."""
    return [dataclasses.replace(q, gold_sql="") for q in questions]


class GenerationStage:
    def __init__(
        self,
        client,
        out_dir: str,
        db_root: str,
        prep_root: str,
        linking: dict[int, dict],
        values: Optional[dict[int, dict]] = None,
        generators: tuple[str, ...] = GENERATORS,
        n: int = 5,
        chunk_size: int = 30,
        render_schema: Optional[Callable[..., str]] = None,
        extra_stats: Optional[dict] = None,
    ) -> None:
        unknown = set(generators) - set(GENERATORS)
        if unknown or not generators:
            raise ValueError(f"generators must be a non-empty subset of {GENERATORS}, got {generators}")
        if n < 1 or chunk_size < 1:
            raise ValueError("n and chunk_size must be >= 1")
        self.client = client
        self.out_dir = out_dir
        self.db_root = db_root
        self.prep_root = prep_root
        self.linking = linking
        self.values = values
        self.generators = tuple(generators)
        self.n = n
        self.chunk_size = chunk_size
        self.render_schema = render_schema or _default_render_schema
        self.extra_stats = dict(extra_stats or {})
        self.cand_path = os.path.join(out_dir, CANDIDATES)

    # ------------------------------------------------------------ public
    def run(self, questions: list[Question]) -> dict:
        os.makedirs(self.out_dir, exist_ok=True)
        questions = blank_gold(questions)
        done, bad_at_start = store.prepare_candidates(self.cand_path, self.n)
        todo = [(q, [g for g in self.generators if (q.question_id, g) not in done]) for q in questions]
        todo = [(q, gens) for q, gens in todo if gens]
        log.info("generation: %d questions, %d with pending generators", len(questions), len(todo))
        for start in range(0, len(todo), self.chunk_size):
            self._run_chunk(todo[start:start + self.chunk_size])
            log.info("generation: %d/%d questions written", min(start + self.chunk_size, len(todo)), len(todo))
        rows, bad_cand = store.read_jsonl_counted(self.cand_path)
        # qrg_checklists stays in stats.json with its old value 0: only writers not in this repository wrote that file
        malformed = {"candidates_dropped_at_start": bad_at_start, "candidates": bad_cand, "qrg_checklists": 0}
        extra = {**self._input_counts(questions), **self.extra_stats, "malformed_lines": malformed}
        stats = store.compute_stats(rows, extra)
        store.write_json_atomic(os.path.join(self.out_dir, STATS), stats)
        gate = stats["error_gate"]
        if not gate["passed"]:
            log.error("error gate FAILED: %d/%d requests failed (%.1f%% > %.0f%%)", gate["failed_requests"],
                      gate["requests"], 100 * gate["failed_rate"], 100 * gate["max_failed_rate"])
        return stats

    # ------------------------------------------------------------ context
    def _input_counts(self, questions: list[Question]) -> dict:
        ids = [q.question_id for q in questions]
        return {
            "questions_selected": len(ids),
            "generators": list(self.generators),
            "demo_db_policy": None,  # kept with its old value for the literal writer (it shows no demos)
            "n": self.n,
            "linking_missing_or_empty": sum(1 for i in ids if not (self.linking.get(i) or {}).get("columns")),
            "values_missing": None if self.values is None else sum(1 for i in ids if i not in self.values),
            # kept with its old value for the literal writer (it reads no look-alike pairs)
            "altgen_pair_sources": {"precomputed": 0, "on_demand": 0},
        }

    def _context(self, q: Question) -> Context:
        link = self.linking.get(q.question_id) or {}
        columns = list(link.get("columns") or []) or None  # missing/empty linking -> full schema
        if columns is None:
            log.warning("q%s: no linked columns; rendering the full schema", q.question_id)
        vrow = (self.values or {}).get(q.question_id)
        hits = list(vrow.get("hits") or []) if vrow else None
        prep_dir = os.path.join(self.prep_root, q.db_id)
        try:
            schema = self.render_schema(prep_dir, db_path(self.db_root, q.db_id), columns=columns, value_hits=hits)
            error = None if schema and schema.strip() else "schema_error: empty schema text"
        except Exception as exc:  # logged and counted as error rows; the run continues
            log.exception("q%s: schema rendering failed", q.question_id)
            schema, error = "", f"schema_error: {type(exc).__name__}: {exc}"
        return Context(q, schema or "", error, columns, hits)

    # ------------------------------------------------------------ literal
    def _literal_reask(self, items: list[tuple[Question, list[dict], list[dict]]]) -> None:
        """items: (question, request messages, its n rows). Rows are patched in place: a flagged sample whose
        re-ask returns SQL is replaced (original kept in literal_original_sql); a failed or SQL-less re-ask
        keeps the original with error None, because merge_extra_candidates drops rows that carry an error."""
        reqs, targets = [], []
        for q, msgs, rows in items:
            for row in rows:
                row.update({"literal_reasked": False, "literal_original_sql": None, "literal_reask_error": None})
                if row["error"] or row["truncated"] or not LIT.count_distinct_flag(q.question, q.evidence, row["sql"]):
                    continue
                reqs.append({"messages": LIT.reask_messages(msgs, row["content"]), "n": 1,
                             "tag": f"gen:literal_reask:{q.question_id}:{row['sample']}"})
                targets.append(row)
        results = self.client.chat_many(reqs) if reqs else []
        for row, comps in zip(targets, results):
            c = comps[0] if comps else None
            sql = "" if (c is None or c.error or c.truncated) else extract_sql(c.content)
            if not sql:
                row["literal_reask_error"] = (c.error if c else None) or ("truncated" if c and c.truncated else "no_sql")
                continue
            row.update({"literal_original_sql": row["sql"], "sql": sql, "content": c.content,
                        "literal_reasked": True})

    def _rows(self, gen: str, q: Question, comps: Optional[list], extra: dict, error: Optional[str]) -> list[dict]:
        rows = []
        for i in range(self.n):
            c = comps[i] if comps else None
            base = {"question_id": q.question_id, "db_id": q.db_id, "generator": gen, "sample": i}
            if c is None:
                body = {"sql": "", "finish_reason": None, "truncated": False, "empty": True, "error": error,
                        "completion_tokens": 0, "prompt_tokens": 0, "content": ""}
            else:
                body = {"sql": "" if c.error else extract_sql(c.content), "finish_reason": c.finish_reason,
                        "truncated": c.truncated, "empty": c.empty, "error": c.error,
                        "completion_tokens": c.completion_tokens, "prompt_tokens": c.prompt_tokens,
                        "content": c.content}
            rows.append({**base, **body, **extra})
        return rows

    def _run_chunk(self, chunk: list[tuple[Question, list[str]]]) -> None:
        ctxs = {q.question_id: self._context(q) for q, _ in chunk}
        rows, pending = [], []
        for q, gens in chunk:
            ctx = ctxs[q.question_id]
            for gen in gens:
                if ctx.schema_error:
                    rows.extend(self._rows(gen, q, None, {}, ctx.schema_error))
                    continue
                msgs = M.literal_messages(q.question, q.evidence, ctx.schema)
                pending.append((gen, q, {"messages": msgs, "n": self.n, "tag": f"gen:{gen}:{q.question_id}"}))
        results = self.client.chat_many([p[2] for p in pending]) if pending else []
        lit_items = []
        for (gen, q, req), comps in zip(pending, results):
            if len(comps) != self.n:
                rows.extend(self._rows(gen, q, None, {}, f"bad_response: {len(comps)} of {self.n} samples"))
            else:
                new = self._rows(gen, q, comps, {}, None)
                lit_items.append((q, req["messages"], new))
                rows.extend(new)
        self._literal_reask(lit_items)
        rows.sort(key=lambda r: (r["question_id"], GENERATORS.index(r["generator"]), r["sample"]))
        store.append_rows(self.cand_path, rows)
