"""Targeted query-tree rules as the stage 6 rules hook.

Hook contract (stage6_query_tree/run_query_tree.py): `apply_rules(item_view, candidates, labels) -> list[str]`, same
length and order as `candidates`; a candidate is returned unchanged unless a routed rule produced a different SQL.
Loaded with `--rules-module stage6_query_tree.qtree.rule_fix:apply_rules`.

What it does per candidate:
  1. compare the question side (question, evidence, linked schema, retrieved values) with the candidate's query
     tree (stage6_query_tree.qtree.match) -> every finding kind, all logged;
  2. keep only the rules whose `tree_findings` the candidate carries (stage6_query_tree.qtree.cards.tree_route_with_kinds);
     a rule without `tree_findings` is unrouted and shown to every candidate (unrouted mode);
     no rule -> no model call;
  3. measure facts on the database through item_view.execute (rows returned; with every DISTINCT removed; whether a
     COUNT(DISTINCT) column is a primary key) and render the rule-correction prompt (stage6_query_tree/prompts.py)
     with the routed rules and the facts section;
  4. one gpt-oss-120b call (common.llm_client.client.LLMClient, project settings, disk cache, 600 s), parse the
     <result>{applied_rules, corrected_sql}</result> answer; applied rules are filtered to the ones shown; an edit
     that cites none of the shown rules is rejected (cited-rule guard) and the candidate kept.
Failure policy: any exception, parse failure or empty answer keeps that candidate unchanged; the function never raises
for one candidate and never changes the list length.

Environment (read once, on first call):
  THESIS_TREE_RULES_FILE   rules JSON: a list of {rule_name, gist, bad_pattern, correct_pattern, fix, note?,
                           tree_findings?}, e.g. stage6_query_tree/rules/tree_rules_r8_supnest.json
  TP_BASE_URL              vLLM server (serving/serve_and_run_local.sh sets it)
  THESIS_TREE_RULES_CACHE  LLM cache dir (default <rules file dir>/llm_cache)
  THESIS_TREE_RULES_RUN_ID cache run id (default "tree_rules")
  THESIS_TREE_RULES_LOG    per-candidate JSONL log (default <rules file dir>/tree_rules_log.jsonl)
  THESIS_MODEL_SETTINGS    settings yaml (default configs/model_gpt_oss_120b.yaml)
  THESIS_TREE_RULES_LOOP   off (default) | on. on = stage6_query_tree.qtree.rule_loop: ONE rule per model call, the query tree is
                           re-checked on the edited SQL, the next finding's rule follows; an edit is kept only if it
                           cites its rule, removes its finding, adds none, and passes the data gate (runs, not empty,
                           not all NULL, result changed).
  THESIS_TREE_RULES_SAMPLES 1 (default) .. 9. k > 1 (loop mode only): each rule step asks k answers and keeps the edit
                           only when a majority propose accepted edits with the same result (agreement voting).
  THESIS_TREE_RULES_FACTS  on (default) | off. off = no measured-facts section and no fact queries: the fixer gets
                           the plain rule-correction prompt with the routed rules; on = step 3 above.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from stage6_query_tree import prompts
from stage6_query_tree.qtree.cards import tree_route_with_kinds
from stage6_query_tree.qtree.rule_loop import same_text
from stage6_query_tree.qtree.match import QuestionSide
from stage6_query_tree.qtree.sql_tree import parse_sql_tree

REPO = Path(__file__).resolve().parents[2]
RULES_ENV = "THESIS_TREE_RULES_FILE"
FACTS_ENV = "THESIS_TREE_RULES_FACTS"
LOOP_ENV = "THESIS_TREE_RULES_LOOP"
SAMPLES_ENV = "THESIS_TREE_RULES_SAMPLES"
RULE_FIELDS = ("rule_name", "gist", "bad_pattern", "correct_pattern", "fix")
PREVIEW_ROWS = 3
_DISTINCT_RE = re.compile(r"\bDISTINCT\b\s*", re.I)
_RESULT_RE = re.compile(r"<result>(.*?)</result>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)```", re.I)
_lock = threading.Lock()
_LOG = logging.getLogger(__name__)
_state: Dict[str, Any] = {}


# ---------------------------------------------------------------- configuration (lazy, once)

def load_rules_file(path: str) -> List[dict]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"rules file must hold a non-empty JSON list: {path}")
    rules, seen = [], set()
    for i, r in enumerate(data):
        bad = [k for k in RULE_FIELDS if not isinstance(r.get(k), str) or not r[k].strip()]
        # a rule WITHOUT the tree_findings key is unrouted (unrouted mode: shown to every candidate); a
        # present key must be a non-empty list (an empty list is ambiguous)
        routed = "tree_findings" in r
        if bad or r["rule_name"] in seen or (routed and (not isinstance(r["tree_findings"], list) or not r["tree_findings"])):
            raise ValueError(f"rule #{i} malformed (missing {bad}, duplicate name or empty tree_findings): {path}")
        seen.add(r["rule_name"])
        rule = {**{k: r[k] for k in RULE_FIELDS}, "note": r.get("note", "")}
        if routed:
            rule["tree_findings"] = list(r["tree_findings"])
        rules.append(rule)
    return rules


def _config() -> Dict[str, Any]:
    with _lock:
        if _state:
            return _state
        rules_path = os.environ.get(RULES_ENV, "").strip()
        if not rules_path:
            raise RuntimeError(f"{RULES_ENV} is not set")
        rules = load_rules_file(rules_path)
        base = os.path.dirname(os.path.abspath(rules_path))
        from common.llm_client.client import LLMClient, load_settings
        settings = load_settings(os.environ.get("THESIS_MODEL_SETTINGS") or str(REPO / "configs" / "model_gpt_oss_120b.yaml"))
        client = LLMClient(settings, base_url=os.environ["TP_BASE_URL"],
                           cache_dir=os.environ.get("THESIS_TREE_RULES_CACHE") or os.path.join(base, "llm_cache"),
                           run_id=os.environ.get("THESIS_TREE_RULES_RUN_ID") or "tree_rules", max_concurrency=3)
        facts = (os.environ.get(FACTS_ENV) or "on").strip().lower()
        if facts not in ("on", "off"):
            raise ValueError(f"{FACTS_ENV} must be 'on' or 'off', got {facts!r}")
        loop = (os.environ.get(LOOP_ENV) or "off").strip().lower()
        if loop not in ("on", "off"):
            raise ValueError(f"{LOOP_ENV} must be 'on' or 'off', got {loop!r}")
        raw = (os.environ.get(SAMPLES_ENV) or "1").strip()
        if not raw.isdigit() or not 1 <= int(raw) <= 9:
            raise ValueError(f"{SAMPLES_ENV} must be an integer 1..9, got {raw!r}")
        samples = int(raw)
        if samples > 1 and loop != "on":
            raise ValueError(f"{SAMPLES_ENV} > 1 needs {LOOP_ENV}=on")
        if loop == "on":
            missing = [r["rule_name"] for r in rules if not r.get("tree_findings")]
            if missing:
                raise ValueError(f"{LOOP_ENV}=on needs tree_findings on every rule; missing on {missing}")
        _state.update(rules=rules, client=client, prompts=prompts, facts=facts == "on", loop=loop == "on",
                      samples=samples,
                      log_path=os.environ.get("THESIS_TREE_RULES_LOG") or os.path.join(base, "tree_rules_log.jsonl"),
                      rules_file=rules_path)
        return _state


def _reset() -> None:   # tests
    with _lock:
        _state.clear()


# ---------------------------------------------------------------- schema text and facts

def render_schema(schema: Dict[str, Any]) -> str:
    """Compact text of a snapshot schema dict ({tables: {t: {columns: {c: {column_type, primary_key, foreign_keys,
    value_examples}}}}}): one line per table, columns with type, PK, FK and up to 3 example values."""
    lines = []
    for t, info in sorted(((schema or {}).get("tables") or {}).items()):
        cols = []
        for c, meta in ((info or {}).get("columns") or {}).items():
            meta = meta or {}
            bits = [str(meta.get("column_type") or "").strip()]
            if meta.get("primary_key"):
                bits.append("PK")
            for fk in meta.get("foreign_keys") or []:
                bits.append(f"FK->{fk}" if isinstance(fk, str) else f"FK->{fk}")
            ex = meta.get("value_examples") or []
            if ex:
                bits.append("e.g. " + ", ".join(str(v) for v in list(ex)[:3]))
            cols.append(f"{c} ({' '.join(b for b in bits if b)})")
        lines.append(f"{t}: " + "; ".join(cols))
    return "\n".join(lines) or "(no schema)"


def _preview(rows: Sequence[Any]) -> str:
    return json.dumps([list(r) for r in list(rows)[:PREVIEW_ROWS]], default=str, ensure_ascii=False)


def _rows_fact(lead: str, res: Any) -> str:
    rows = list(getattr(res, "rows", ()) or ())
    if not rows:
        return f"{lead} 0 rows (empty result)"
    first = "all" if len(rows) <= PREVIEW_ROWS else f"first {PREVIEW_ROWS}"
    null = " (every value is NULL)" if getattr(res, "null_only", None) else ""
    return f"{lead} {len(rows)} row(s){null}; {first}: {_preview(rows)}"


def _pk_facts(sql: str, schema: Dict[str, Any]) -> List[str]:
    tree = parse_sql_tree(sql)
    if not tree.ok:
        return []
    counted = [(tab, col) for o in tree.outputs if o.fn == "COUNT" and o.distinct for tab, col in o.columns]
    if not counted:
        return ["COUNT(DISTINCT ...) is not a top-level output, so its column's key status is unknown"] if tree.count_distinct else []
    tables = {str(t).lower(): (info or {}).get("columns") or {} for t, info in ((schema or {}).get("tables") or {}).items()}
    out = []
    for tab, col in counted:
        cols = tables.get((tab or "").lower())
        meta = next((v for k, v in cols.items() if str(k).lower() == col), None) if cols else None
        if meta is None or "primary_key" not in meta:
            out.append(f"COUNT(DISTINCT {tab or '?'}.{col}): whether {col} is a primary key is unknown")
        elif meta["primary_key"]:
            out.append(f"COUNT(DISTINCT {tab}.{col}): {col} is a primary key of {tab} (each row has its own value, so "
                       f"DISTINCT changes the count only when the joins repeat rows)")
        else:
            out.append(f"COUNT(DISTINCT {tab}.{col}): {col} is NOT a primary key of {tab} (the same value can occur in several rows)")
    return out


def candidate_facts(view: Any, sql: str, kinds: frozenset) -> str:
    """'- fact' lines measured read-only on the item's database; '' when nothing could be measured."""
    try:
        lines: List[str] = []
        original = view.execute(sql)
        if not original.ok:
            lines.append(f"the candidate could not be executed ({original.error})")
        else:
            lines.append(_rows_fact("the candidate returns", original))
        if "distinct_no_cue" in kinds and _DISTINCT_RE.search(sql):
            without = view.execute(_DISTINCT_RE.sub("", sql))
            if not without.ok:
                lines.append(f"with every DISTINCT removed the candidate could not be executed ({without.error})")
            elif not original.ok:
                lines.append(_rows_fact("with every DISTINCT removed the candidate returns", without))
            else:
                same = frozenset(map(tuple, original.rows)) == frozenset(map(tuple, without.rows))
                lines.append(f"with every DISTINCT removed the candidate returns {len(without.rows)} row(s) instead of "
                             f"{len(original.rows)}; " + ("the result is identical" if same else
                                                          f"the result differs (without DISTINCT: {_preview(without.rows)})"))
            lines.extend(_pk_facts(sql, getattr(view, "database_schema", None) or {}))
        return "\n".join(f"- {l}" for l in lines)
    except Exception:
        return ""


# ---------------------------------------------------------------- model call and parsing

def parse_answer(text: str, shown_names: set) -> Optional[Dict[str, Any]]:
    """{applied_rules (subset of shown), corrected_sql} from <result>...</result>, else None. Trailing text after the
    JSON object inside <result> is tolerated (raw_decode)."""
    m = _RESULT_RE.search(text or "")
    if not m:
        return None
    content = m.group(1).strip()
    fence = _FENCE_RE.search(content)
    if fence:
        content = fence.group(1).strip()
    start = content.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder(strict=False).raw_decode(content[start:])
    except json.JSONDecodeError:
        return None
    applied = obj.get("applied_rules")
    sql = obj.get("corrected_sql")
    if not isinstance(applied, list) or not isinstance(sql, str) or not sql.strip():
        return None
    return {"applied_rules": [a for a in applied if a in shown_names], "corrected_sql": sql.strip()}


_COUNTS: "collections.Counter[str]" = collections.Counter()


def _count(name: str) -> None:
    with _lock:
        _COUNTS[name] += 1


def rule_stats() -> Dict[str, int]:
    """Outcome counts of non-loop corrections in this process (llm_error, parse_failed, guard_reject,
    edit_does_not_run, changed, no_edit), so an outage can never look like 'no effect'."""
    with _lock:
        return dict(_COUNTS)


def _breaks_execution(view: Any, old_sql: str, new_sql: str) -> bool:
    """True when the edit fails to run although the candidate runs; False when it cannot be checked."""
    run = getattr(view, "execute", None)
    if run is None:
        return False
    try:
        return bool(run(old_sql).ok) and not bool(run(new_sql).ok)
    except Exception:  # the check itself failed: keep the earlier behaviour
        return False


def _norm(sql: str) -> str:
    return " ".join((sql or "").split()).lower()


def _log(cfg: Dict[str, Any], record: Dict[str, Any]) -> None:
    try:
        with _lock:
            os.makedirs(os.path.dirname(os.path.abspath(cfg["log_path"])), exist_ok=True)
            with open(cfg["log_path"], "a") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def correct_one(cfg: Dict[str, Any], view: Any, side: QuestionSide, schema_text: str, sql: str) -> str:
    """The new SQL for one candidate (the same SQL when nothing applies or anything fails)."""
    record: Dict[str, Any] = {"question_id": getattr(view, "question_id", None), "sql": sql, "findings": [],
                              "rules_shown": [], "llm_call": False, "rules_applied": [], "changed": False}
    if cfg.get("loop"):
        return correct_one_loop(cfg, view, side, schema_text, sql, record)
    try:
        shown, kinds = tree_route_with_kinds(cfg["rules"], sql, side)
        record["findings"] = sorted(kinds)
        record["rules_shown"] = [r["rule_name"] for r in shown]
        if not shown:
            _log(cfg, record)
            return sql
        if cfg.get("facts", True):
            facts = candidate_facts(view, sql, kinds)
            record["facts"] = facts
            prompt = cfg["prompts"].format_rule_correction_prompt_with_facts(
                database_schema=schema_text, question=side.question, evidence=side.evidence or "", sql=sql, rules=shown,
                facts_text=facts)
        else:   # facts off: the plain rule-correction prompt with the routed rules, no fact queries
            record["facts"] = None
            prompt = cfg["prompts"].format_rule_correction_prompt(
                database_schema=schema_text, question=side.question, evidence=side.evidence or "", sql=sql, rules=shown)
        record["llm_call"] = True
        comps = cfg["client"].chat([{"role": "user", "content": prompt}], n=1, tag="tree_rules")
        comp = comps[0] if comps else None
        if comp is None or comp.error:
            _count("llm_error")
            record["error"] = getattr(comp, "error", "no completion")
            _log(cfg, record)
            return sql
        parsed = parse_answer(comp.content or "", set(record["rules_shown"]))
        if parsed is None:
            _count("parse_failed")
            record["error"] = "parse_failed"
            _log(cfg, record)
            return sql
        record["rules_applied"] = parsed["applied_rules"]
        new_sql = parsed["corrected_sql"]
        edited = not same_text(new_sql, sql)   # case-sensitive: a literal-case edit is an edit
        if edited and not record["rules_applied"]:   # cited-rule guard: an uncited edit is rejected
            _count("guard_reject")
            record.update(changed=False, rejected="edit_without_cited_rule", proposed_sql=new_sql)
            _log(cfg, record)
            return sql
        if edited and _breaks_execution(view, sql, new_sql):   # never trade a working candidate
            _count("edit_does_not_run")                        # for one that does not run
            record.update(changed=False, rejected="edit_does_not_run", proposed_sql=new_sql)
            _log(cfg, record)
            return sql
        record["changed"] = edited
        _count("changed" if edited else "no_edit")
        _log(cfg, record)
        return new_sql if edited else sql
    except Exception as e:  # never let one candidate break the item
        record["error"] = f"exception:{type(e).__name__}: {e}"[:300]
        _log(cfg, record)
        return sql


def correct_one_loop(cfg: Dict[str, Any], view: Any, side: QuestionSide, schema_text: str, sql: str,
                     record: Dict[str, Any]) -> str:
    """THESIS_TREE_RULES_LOOP=on: stage6_query_tree.qtree.rule_loop with this hook's prompt, model and execution."""
    from stage6_query_tree.qtree.rule_loop import ROWS_CAPPED, default_finder, run_loop
    try:
        finder = default_finder(side)

        def ask(rule: dict, current: str):
            if cfg.get("facts", True):
                prompt = cfg["prompts"].format_rule_correction_prompt_with_facts(
                    database_schema=schema_text, question=side.question, evidence=side.evidence or "", sql=current,
                    rules=[rule], facts_text=candidate_facts(view, current, finder(current)))
            else:
                prompt = cfg["prompts"].format_rule_correction_prompt(
                    database_schema=schema_text, question=side.question, evidence=side.evidence or "", sql=current,
                    rules=[rule])
            k = int(cfg.get("samples", 1))
            comps = cfg["client"].chat([{"role": "user", "content": prompt}], n=k, tag="tree_rules")
            parsed = [None if (c is None or c.error) else parse_answer(c.content or "", {rule["rule_name"]}) for c in comps or []]
            if k > 1:
                return parsed
            return parsed[0] if parsed else None

        def execute(q: str):
            r = view.execute(q)
            if getattr(r, "rows_capped", False):   # only a prefix was seen: the result cannot be compared
                return (False, None, ROWS_CAPPED)
            return (bool(r.ok), list(r.rows) if r.ok else None, None if r.ok else r.error)

        def on_exception(where: str, exc: BaseException) -> None:
            _LOG.warning("tree rules loop %s failed for item %s: %s", where, getattr(view, "question_id", "?"), exc,
                         exc_info=exc)

        res = run_loop(sql, cfg["rules"], ask, execute, finder, on_exception=on_exception,
                       samples=int(cfg.get("samples", 1)))
        record.update(findings=sorted(res.findings_before), findings_after=sorted(res.findings_after),
                      llm_call=res.calls > 0, calls=res.calls, loop=True, rules_applied=list(res.applied),
                      rules_shown=[st.rule for st in res.steps], changed=res.changed,
                      steps=[st.as_dict() for st in res.steps], facts_mode="on" if cfg.get("facts", True) else "off",
                      samples=int(cfg.get("samples", 1)))
        _log(cfg, record)
        return res.sql if res.changed else sql
    except Exception as e:  # never let one candidate break the item
        record["error"] = f"exception:{type(e).__name__}: {e}"[:300]
        _log(cfg, record)
        return sql


def apply_rules(item_view: Any, candidates: List[str], labels: List[str]) -> List[str]:
    """The stage 6 rules hook: targeted tree rules on every candidate; same length and order; identical SQL text is
    corrected once (memoised per item)."""
    cfg = _config()
    side = QuestionSide(getattr(item_view, "question", "") or "", getattr(item_view, "evidence", "") or "",
                        getattr(item_view, "linked_schema", None) or {}, getattr(item_view, "retrieved_values", None))
    schema_text = render_schema(getattr(item_view, "linked_schema", None) or getattr(item_view, "database_schema", None) or {})
    memo: Dict[str, str] = {}
    out = []
    for sql in candidates:
        key = _norm(sql)
        if key not in memo:
            memo[key] = correct_one(cfg, item_view, side, schema_text, sql) if sql and sql.strip() else sql
        out.append(memo[key])
    return out
