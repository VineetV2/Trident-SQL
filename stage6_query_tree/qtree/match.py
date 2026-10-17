"""Two-way matcher between the question side and the SQL tree.

Question -> SQL: does the SQL realise what the question asks (required literals, aggregation, superlative)?
SQL -> question: is every table, column, literal and DISTINCT in the SQL supported by the question, its hint
(evidence), the linked schema or the retrieved values?
Each mismatch is typed and carries a plain-English explanation (the "back to English" step); typed
mismatches are the candidates for rules, which are kept only if they raise accuracy on TRAIN.
The question side is the snapshot item's input for that question (question, evidence, linked schema, retrieved
values); nothing here reads gold SQL.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from stage6_query_tree.qtree.sql_tree import SQLTree

DISTINCT_CUE = re.compile(r"\b(distinct|different|unique|kinds?|types?|varieties|various)\b", re.I)
SUPERLATIVE_CUE = re.compile(r"\b(highest|lowest|most|least|maximum|minimum|max|min|top|largest|smallest|biggest|"
                             r"oldest|youngest|best|worst|latest|earliest|longest|shortest|greatest|fewest|"
                             r"newest|cheapest|heaviest|lightest|tallest)\b", re.I)
AT_MOST = re.compile(r"\bat (most|least)\b", re.I)
AVG_CUE = re.compile(r"\b(average|mean|avg)\b", re.I)
COUNT_CUE = re.compile(r"\b(how many|number of|count)\b", re.I)
EVIDENCE_VALUE = re.compile(r"(?:=|\bin\s*\(|\blike)\s*(?:'([^']{1,80})'|\"([^\"]{1,80})\")", re.I)
NEST_CUE = re.compile(r"\b(than|average|percentage|percent|ratio|difference|each|per|compared|rate|proportion|"
                      r"times|more|less|over)\b", re.I)
QUESTION_QUOTED = re.compile(r"(?<![\w'])'([^']{2,60})'(?![\w'])|\"([^\"]{2,60})\"")
NOTES_MARKER = re.compile(r"\n\s*\[")


def clean_evidence(evidence: str) -> str:
    """The original BIRD hint only: drop notes our pipeline appends after a blank line + '[' header."""
    return NOTES_MARKER.split(evidence or "", maxsplit=1)[0]


@dataclass
class QuestionSide:
    question: str
    evidence: str = ""
    linked_schema: Dict[str, Any] = field(default_factory=dict)
    retrieved_values: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class Mismatch:
    kind: str
    detail: str
    english: str


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _linked(schema: Dict[str, Any]) -> Dict[str, Set[str]]:
    tables = (schema or {}).get("tables") or {}
    return {t.lower(): {c.lower() for c in ((info or {}).get("columns") or {})} for t, info in tables.items()}


def _retrieved(rv: Optional[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for cols in (rv or {}).values():
        for vals in (cols or {}).values():
            for v in vals or []:
                out.add(_norm(v.get("value", "") if isinstance(v, dict) else v))
    return out


def _required_literals(side: QuestionSide) -> List[str]:
    """Values the question or its hint pins down explicitly (quoted strings)."""
    found = [a or b for a, b in EVIDENCE_VALUE.findall(clean_evidence(side.evidence))]
    found += [a or b for a, b in QUESTION_QUOTED.findall(side.question or "")]
    return [v for v in found if v.strip() and "?" not in v]


def match(t: SQLTree, side: QuestionSide) -> List[Mismatch]:
    if not t.ok:
        return [Mismatch("unparseable", t.error, "The SQL could not be parsed, so its meaning cannot be checked.")]
    out: List[Mismatch] = []
    linked = _linked(side.linked_schema)
    text = _norm(f"{side.question} {clean_evidence(side.evidence)}")
    # SQL -> question: tables and columns must come from the linked schema
    if linked:
        for tab in sorted(t.tables - set(linked)):
            out.append(Mismatch("unlinked_table", tab, f"The SQL reads table '{tab}', which the question was never "
                                                        f"linked to."))
        all_cols = set().union(*linked.values()) if linked else set()
        for tab, col in sorted(t.columns, key=lambda c: (c[0] or "", c[1])):
            if col == "*" or col in t.output_aliases or (tab and tab not in linked):
                continue
            if (tab and col not in linked.get(tab, set())) or (not tab and col not in all_cols):
                out.append(Mismatch("unlinked_column", f"{tab or '?'}.{col}",
                                    f"The SQL uses column '{col}', which is not among the columns linked to the "
                                    f"question."))
    # SQL -> question: string literals must be supported by the question, hint or retrieved values
    support = text + " " + " ".join(_retrieved(side.retrieved_values))
    for f in t.filters:
        for v in f.value.split("|"):
            vv = _norm(v).strip("%")
            if not vv or re.fullmatch(r"-?[\d.]+", vv):
                continue
            if not re.search(r"(?<!\w)" + re.escape(vv) + r"(?!\w)", support):
                out.append(Mismatch("unsupported_literal", f"{f.column[1]} {f.op} '{v}'",
                                    f"The SQL filters {f.column[1]} on '{v}', a value the question and hint never "
                                    f"mention."))
    # question -> SQL: explicitly quoted values must be used
    sql_lits = {_norm(x).strip("%") for x in t.literals}
    for lit in _required_literals(side):
        if not any(_norm(lit) in s or s in _norm(lit) for s in sql_lits if s):
            out.append(Mismatch("missing_literal", lit, f"The question or hint names the value '{lit}', but the SQL "
                                                        f"never uses it."))
    # SQL -> question: DISTINCT needs a cue
    if (t.count_distinct or t.select_distinct) and not DISTINCT_CUE.search(text):
        out.append(Mismatch("distinct_no_cue", "COUNT(DISTINCT)" if t.count_distinct else "SELECT DISTINCT",
                            "The SQL removes duplicates, but the question does not ask for distinct values."))
    # round 8 routing kind: the same missing cue, but only for COUNT(DISTINCT ...) in the outermost select (the
    # DISTINCT rule's scope); distinct_no_cue above is unchanged so earlier runs stay reproducible
    if t.top_count_distinct and not DISTINCT_CUE.search(text):
        out.append(Mismatch("count_distinct_no_cue", "COUNT(DISTINCT) in the outer SELECT",
                            "The SQL counts distinct values in its answer, but the question does not ask for distinct "
                            "values."))
    # question -> SQL: superlative and aggregation
    if SUPERLATIVE_CUE.search(AT_MOST.sub("", side.question or "")) and not t.order_desc and not ({"MAX", "MIN"} & t.agg_fns):
        out.append(Mismatch("missing_superlative", "no ORDER BY / MAX / MIN",
                            "The question asks for a highest/lowest kind of answer, but the SQL neither sorts nor "
                            "takes a maximum or minimum."))
    if AVG_CUE.search(side.question or "") and not ({"AVG", "SUM"} & t.agg_fns):
        out.append(Mismatch("missing_aggregate", "AVG", "The question asks for an average, but the SQL computes none."))
    if COUNT_CUE.search(side.question or "") and not ({"COUNT", "SUM"} & t.agg_fns):
        out.append(Mismatch("missing_aggregate", "COUNT", "The question asks how many, but the SQL counts nothing."))
    # tree shape: the SQL's plan is much deeper than the question needs (CTEs or 3+ query levels)
    if t.ctes or t.scopes >= 3:
        cue = bool(NEST_CUE.search(side.question or ""))
        out.append(Mismatch("extra_nesting" if not cue else "extra_nesting_cued",
                            f"{t.scopes} query levels, {t.ctes} CTEs",
                            f"The SQL builds {t.scopes} nested query levels"
                            f"{' and temporary tables' if t.ctes else ''}, but the question asks for a simpler, "
                            f"flatter answer."))
    # SQL -> question: a table used only as a join leaf adds rows but no meaning
    if len(t.tables) >= 2:
        degree: Dict[str, int] = {}
        for edge in t.joins:
            for tab in {c[0] for c in edge if c[0]}:
                degree[tab] = degree.get(tab, 0) + 1
        for tab in sorted(t.tables):
            if degree.get(tab, 0) == 1 and not t.table_columns_outside_joins.get(tab):
                out.append(Mismatch("dangling_table", tab, f"The SQL joins table '{tab}' but uses nothing from it; "
                                                           f"the join can only add or drop rows."))
    return out
