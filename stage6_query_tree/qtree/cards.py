"""Query-tree routing for rule cards: a card may carry `tree_findings`, a list of mismatch
kinds from stage6_query_tree.qtree.match. Such a card is shown to the model for a candidate ONLY when the candidate's own query
tree, compared with the question side, has one of those kinds. Cards without `tree_findings` are kept for every
candidate (unrouted mode).

Why: with all rules shown to every candidate, the model may apply a rule where nothing is wrong. Here the code
decides which rule a candidate may even see; the model still decides whether it applies and writes the edit; the
post-edit execution check (rule_fix._breaks_execution, run_query_tree.fix_rules) still gates the edit. The tree only
triggers; it never decides.

Entry points (rule_fix.correct_one calls tree_route_with_kinds(rules, sql, side)):
    side = side_from_item(item)
    shown = tree_route(cards, sql, side)
Failure policy: an unparseable candidate or a missing question side shows NO tree-routed card (never a wrong one).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from stage6_query_tree.qtree.match import QuestionSide, match
from stage6_query_tree.qtree.sql_tree import parse_sql_tree

TREE_KEY = "tree_findings"
KNOWN_KINDS = frozenset({"unlinked_table", "unlinked_column", "unsupported_literal", "missing_literal",
                         "distinct_no_cue", "missing_superlative", "missing_aggregate", "extra_nesting",
                         "extra_nesting_cued", "dangling_table", "unparseable", "count_distinct_no_cue"})


def side_from_item(item: Any) -> Optional[QuestionSide]:
    """QuestionSide from a snapshot item object (duck-typed: question, evidence, linked schema, retrieved values)."""
    question = getattr(item, "question", None)
    if not question:
        return None
    return QuestionSide(question, getattr(item, "evidence", "") or "",
                        getattr(item, "database_schema_after_schema_linking", None) or {},
                        getattr(item, "retrieved_values", None))


def finding_kinds(sql: str, side: Optional[QuestionSide]) -> frozenset:
    """Mismatch kinds of one candidate; empty when the side is unknown or the SQL does not parse."""
    if side is None:
        return frozenset()
    try:
        return frozenset(m.kind for m in match(parse_sql_tree(sql), side))
    except Exception:
        return frozenset()


def tree_route(cards: Iterable[dict], sql: str, side: Optional[QuestionSide]) -> List[dict]:
    """Keep every card without `tree_findings`; keep a tree-routed card only when the candidate shows one of its
    kinds. Computes the tree once per candidate."""
    cards = list(cards)
    routed = [c for c in cards if c.get(TREE_KEY)]
    if not routed:
        return cards
    kinds = finding_kinds(sql, side)
    return [c for c in cards if not c.get(TREE_KEY) or kinds & frozenset(c[TREE_KEY])]


def tree_route_with_kinds(cards: Iterable[dict], sql: str, side: Optional[QuestionSide]) -> tuple:
    """(cards to show, every finding kind of the candidate). The tree is checked for ALL kinds (detect and log every
    error; only the routed rules repair), so the kinds are returned for logging."""
    cards = list(cards)
    kinds = finding_kinds(sql, side)
    return [c for c in cards if not c.get(TREE_KEY) or kinds & frozenset(c[TREE_KEY])], kinds


TREE_LOG_NAME = "tree_findings_log.jsonl"

# What the rules hook shows the fixer as "Measured facts about the candidate SQL" in routed mode (THESIS-PATCH,
# written by rule_fix.candidate_facts), described per finding kind.
# Every kind gets the first line; the rest only where listed.
FACT_LINE_ROWS = ("the candidate returns N row(s), columns [...]; first 3 rows shown (or: the candidate could not be "
                  "executed (<error kind>))")
FACT_LINES_DISTINCT = (
    "with every DISTINCT removed the candidate returns M row(s) instead of N; the result is identical (= nothing to "
    "fix) or the result differs (with a preview of the rows without DISTINCT); note: EVERY DISTINCT in the query is "
    "removed for this measurement, including ones inside subqueries",
    "for each top-level COUNT(DISTINCT t.c): c is a primary key of t, or c is NOT a primary key of t, or unknown; a "
    "COUNT(DISTINCT) that sits only inside a subquery is reported as not a top-level output",
)
FACTS_SHOWN: Dict[str, List[str]] = {k: [FACT_LINE_ROWS] for k in sorted(KNOWN_KINDS - {"unparseable"})}
FACTS_SHOWN["distinct_no_cue"] = [FACT_LINE_ROWS, *FACT_LINES_DISTINCT]


def summarize_findings_log(path: str) -> dict:
    """Counts over a tree_findings_log.jsonl (one JSON line per candidate): candidates, candidates with any
    finding, per-kind counts, per-rule shown / applied / changed, candidates with no LLM call."""
    import json
    from collections import Counter
    n = with_finding = no_call = 0
    kinds, shown, applied, changed = Counter(), Counter(), Counter(), Counter()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            if r.get("findings"):
                with_finding += 1
            kinds.update(r.get("findings") or [])
            shown.update(r.get("rules_shown") or [])
            applied.update(r.get("rules_applied") or [])
            if r.get("changed"):
                changed.update(r.get("rules_applied") or ["(no rule cited)"])
            if not r.get("llm_call"):
                no_call += 1
    return {"candidates": n, "with_any_finding": with_finding, "no_llm_call": no_call, "findings": dict(kinds),
            "rules_shown": dict(shown), "rules_applied": dict(applied), "changed_by_rule": dict(changed)}


def tree_key_errors(card: dict, where: str) -> List[str]:
    """Validation of the optional key: error messages for this card, empty when the key is absent or valid."""
    if TREE_KEY not in card:
        return []
    v = card[TREE_KEY]
    if not isinstance(v, list) or not v or not all(isinstance(k, str) for k in v):
        return [f"{where}: {TREE_KEY} must be a non-empty list of strings"]
    bad = sorted(set(v) - KNOWN_KINDS)
    return [f"{where}: {TREE_KEY} has unknown kind(s) {bad}"] if bad else []
