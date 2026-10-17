"""Tree repair recipes (query-tree mechanism 3): plain code edits of a flagged SQL, one recipe per mismatch kind.

The matcher (match.py) says WHAT looks wrong in a candidate's tree; a recipe makes the one edit that removes
that mismatch and returns the edited SQL. No model decides anything here. The only caller is distinct_fix.decide: it
asks repairs() for the distinct_no_cue kind and keeps only the drop_distinct output, which it uses only for a
COUNT(DISTINCT x) whose x is not a key and when the edited SQL runs and returns a different result; the edited SQL then
REPLACES that SQL in place (stage 6 `--fixes distinct`, per candidate; stage 8's post-selection fix, the selected SQL).
No new candidate is added and nothing goes back into a vote. No other recipe's output is used anywhere (drop_not_null,
having no kind, also runs in that call and is dropped), and pipeline.py runs neither of those two fixes. Recipes are
deliberately narrow; anything outside the pattern is skipped.
  drop_distinct        distinct_no_cue      SELECT DISTINCT -> SELECT; COUNT(DISTINCT x) -> COUNT(x)
  add_top1             missing_superlative  the hint names MAX(col)/MIN(col): add ORDER BY col DESC/ASC LIMIT 1
  swap_unlinked_column unlinked_column      a column outside the linked schema -> the most name-similar linked
                                            column of the same table (one new SQL per replaced column)
  drop_not_null        (no kind)            remove `col IS NOT NULL` conjuncts the question/hint never asks for
  flatten_in_subquery  extra_nesting        WHERE a.c IN (SELECT t.k FROM t WHERE p) -> JOIN t ON a.c = t.k ... p
Contract: repairs() never raises, never returns the input unchanged, every output re-parses (sqlite), order is fixed.
Nothing here reads gold SQL.
"""
from __future__ import annotations

import difflib
import re
from typing import Callable, Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp

from stage4_generation.plan_and_fragment.pairs import name_tokens
from stage6_query_tree.qtree.match import QuestionSide, clean_evidence

DIALECT = "sqlite"
HINT_EXTREME = re.compile(r"\b(MAX|MIN)\s*\(\s*([`\"\[]?)([A-Za-z_][\w .]*?)\2\]?\s*\)", re.I)
NULL_CUE = re.compile(r"\b(null|empty|missing|blank|unknown|not available|no value)\b", re.I)
SIMILAR_MIN = 0.6


def _parse(sql: str) -> Optional[exp.Expression]:
    try:
        return sqlglot.parse_one(sql, read=DIALECT)
    except Exception:  # unparseable input: no recipe applies
        return None


def _out(node: exp.Expression) -> str:
    return node.sql(dialect=DIALECT)


def _outer_select(root: exp.Expression) -> Optional[exp.Select]:
    return root if isinstance(root, exp.Select) else None


def _own_tables(select: exp.Select) -> Dict[str, str]:
    """alias-or-name (lower) -> table name (lower) for the tables of this SELECT's own FROM/JOIN."""
    out: Dict[str, str] = {}
    for t in select.find_all(exp.Table):
        if t.find_ancestor(exp.Select) is select:
            out[(t.alias_or_name or t.name).lower()] = t.name.lower()
    return out


def _conjuncts(node: Optional[exp.Expression]) -> List[exp.Expression]:
    if node is None:
        return []
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    if isinstance(node, exp.Paren) and isinstance(node.this, exp.And):
        return _conjuncts(node.this)
    return [node]


def _set_where(select: exp.Select, conds: List[exp.Expression]) -> None:
    if not conds:
        select.set("where", None)
        return
    cond = conds[0]
    for c in conds[1:]:
        cond = exp.and_(cond, c)
    select.set("where", exp.Where(this=cond))


def _has_agg(nodes: List[exp.Expression]) -> bool:
    return any(n.find(exp.AggFunc) is not None for n in nodes)


# ------------------------------------------------------------------ recipes
def drop_distinct(sql: str, side: QuestionSide) -> List[str]:
    root = _parse(sql)
    if root is None:
        return []
    changed = False
    for s in root.find_all(exp.Select):
        if s.args.get("distinct") is not None:
            s.set("distinct", None)
            changed = True
    for c in list(root.find_all(exp.Count)):
        d = c.this
        if isinstance(d, exp.Distinct) and len(d.expressions) == 1:
            c.set("this", d.expressions[0].copy())
            changed = True
    return [_out(root)] if changed else []


def add_top1(sql: str, side: QuestionSide) -> List[str]:
    hits = {(fn.upper(), col.strip().split(".")[-1].strip("`\"[] ")) for fn, _, col in
            HINT_EXTREME.findall(clean_evidence(side.evidence))}
    hits = {(fn, col) for fn, col in hits if col}
    if len(hits) != 1:
        return []                                   # no hint, or several candidates: ambiguous
    fn, col = next(iter(hits))
    root = _parse(sql)
    sel = _outer_select(root) if root is not None else None
    if sel is None or sel.args.get("order") or sel.args.get("limit") or _has_agg(sel.expressions):
        return []
    tables = _own_tables(sel)
    quals = {c.table or "" for c in sel.find_all(exp.Column)
             if c.name.lower() == col.lower() and c.find_ancestor(exp.Select) is sel}
    if len(tables) == 1:
        qual = next(iter(quals)) if len(quals) == 1 else ""
    else:
        quals.discard("")
        if len(quals) != 1:
            return []                               # unqualified in a multi-table query, or several: ambiguous
        qual = next(iter(quals))
    column = exp.column(col, table=qual or None, quoted=not re.fullmatch(r"[A-Za-z_]\w*", col))
    desc = fn == "MAX"                              # SQLite default NULL order: first for ASC, last for DESC
    sel = sel.order_by(exp.Ordered(this=column, desc=desc, nulls_first=not desc), copy=False).limit(1, copy=False)
    return [_out(sel)]


def _linked_columns(side: QuestionSide) -> Dict[str, List[str]]:
    tables = (side.linked_schema or {}).get("tables") or {}
    return {t.lower(): sorted(((info or {}).get("columns") or {})) for t, info in tables.items()}


def _closest(col: str, choices: List[str]) -> Optional[str]:
    scored = []
    for c in choices:
        ratio = difflib.SequenceMatcher(None, col.lower(), c.lower()).ratio()
        if ratio >= SIMILAR_MIN or name_tokens(col) & name_tokens(c):
            scored.append((-ratio, c.lower(), c))
    return min(scored)[2] if scored else None


def swap_unlinked_column(sql: str, side: QuestionSide) -> List[str]:
    root = _parse(sql)
    linked = _linked_columns(side)
    if root is None or not linked:
        return []
    alias = {}
    for t in root.find_all(exp.Table):
        alias[(t.alias_or_name or t.name).lower()] = t.name.lower()
    real_tables = set(alias.values())
    out_aliases = {e.alias.lower() for s in root.find_all(exp.Select) for e in s.expressions if e.alias}
    targets: List[Tuple[str, str]] = []             # (table, column) to replace, first-seen order
    for c in root.find_all(exp.Column):
        name = c.name
        if not name or name == "*" or name.lower() in out_aliases:
            continue
        tab = alias.get((c.table or "").lower()) if c.table else (next(iter(real_tables)) if len(real_tables) == 1 else None)
        if tab is None or tab not in linked or name.lower() in {x.lower() for x in linked[tab]}:
            continue
        if (tab, name.lower()) not in targets:
            targets.append((tab, name.lower()))
    results = []
    for tab, name in targets:
        new = _closest(name, linked[tab])
        if new is None:
            continue
        edited = root.copy()
        for c in edited.find_all(exp.Column):
            ctab = alias.get((c.table or "").lower()) if c.table else (next(iter(real_tables)) if len(real_tables) == 1 else None)
            if ctab == tab and c.name.lower() == name:
                c.set("this", exp.to_identifier(new, quoted=not re.fullmatch(r"[A-Za-z_]\w*", new)))
        results.append(_out(edited))
    return results


def drop_not_null(sql: str, side: QuestionSide) -> List[str]:
    if NULL_CUE.search(f"{side.question} {clean_evidence(side.evidence)}") or re.search(r"\bnot null\b", clean_evidence(side.evidence), re.I):
        return []
    root = _parse(sql)
    if root is None:
        return []
    changed = False
    for s in root.find_all(exp.Select):
        where = s.args.get("where")
        if where is None:
            continue
        conds = _conjuncts(where.this)
        keep = [c for c in conds if not (isinstance(c, exp.Not) and isinstance(c.this, exp.Is)
                                         and isinstance(c.this.expression, exp.Null))]
        if len(keep) != len(conds):
            _set_where(s, [k.copy() for k in keep])
            changed = True
    return [_out(root)] if changed else []


def flatten_in_subquery(sql: str, side: QuestionSide) -> List[str]:
    root = _parse(sql)
    sel = _outer_select(root) if root is not None else None
    if sel is None or sel.args.get("where") is None:
        return []
    outer_tables = _own_tables(sel)
    if any(not c.table for c in sel.find_all(exp.Column)
           if c.find_ancestor(exp.Select) is sel) and len(outer_tables) != 1:
        return []                                   # unqualified outer columns would become ambiguous
    conds = _conjuncts(sel.args["where"].this)
    for i, cond in enumerate(conds):
        if not isinstance(cond, exp.In) or cond.args.get("query") is None:
            continue
        sub = cond.args["query"].this if isinstance(cond.args["query"], exp.Subquery) else cond.args["query"]
        if not isinstance(sub, exp.Select) or len(sub.expressions) != 1 or not isinstance(sub.expressions[0], exp.Column):
            continue
        if sub.args.get("joins") or sub.args.get("group") or sub.args.get("limit") or sub.args.get("order") \
                or sub.args.get("distinct") is not None or _has_agg(sub.expressions):
            continue
        if len(list(sub.find_all(exp.Select))) != 1 or not isinstance(cond.this, exp.Column):
            continue
        sub_tables = list(sub.find_all(exp.Table))
        if len(sub_tables) != 1:
            continue
        t = sub_tables[0]
        name = (t.alias_or_name or t.name).lower()
        if name in outer_tables or t.name.lower() in outer_tables.values():
            continue                                # self-join or alias clash: skip
        if any(c.table and c.table.lower() != name for c in sub.find_all(exp.Column)):
            continue                                # correlated or foreign qualifier
        new_sub = sub.copy()
        for c in new_sub.find_all(exp.Column):
            if not c.table:
                c.set("table", exp.to_identifier(t.alias_or_name or t.name))
        outer = sel.copy()
        if len(outer_tables) == 1:
            oname = next(iter(outer_tables))
            for c in outer.find_all(exp.Column):
                if not c.table and c.find_ancestor(exp.Select) is outer:
                    c.set("table", exp.to_identifier(oname))
        left = cond.this.copy()
        if not left.table and len(outer_tables) == 1:
            left.set("table", exp.to_identifier(next(iter(outer_tables))))
        on = exp.EQ(this=left, expression=new_sub.expressions[0].copy())
        outer = outer.join(t.copy(), on=on, copy=False)
        rest = [c.copy() for j, c in enumerate(_conjuncts(outer.args["where"].this)) if j != i]
        sub_where = new_sub.args.get("where")
        _set_where(outer, rest + ([sub_where.this.copy()] if sub_where is not None else []))
        return [_out(outer)]
    return []


RECIPES: List[Tuple[str, Optional[str], Callable[[str, QuestionSide], List[str]]]] = [
    ("drop_distinct", "distinct_no_cue", drop_distinct),
    ("add_top1", "missing_superlative", add_top1),
    ("swap_unlinked_column", "unlinked_column", swap_unlinked_column),
    ("drop_not_null", None, drop_not_null),
    ("flatten_in_subquery", "extra_nesting", flatten_in_subquery),
]


def _canon(sql: str) -> Optional[str]:
    node = _parse(sql)
    return _out(node) if node is not None else None


def repairs(sql: str, side: QuestionSide, kinds: Set[str]) -> List[Tuple[str, str]]:
    """(recipe, new_sql) for every recipe whose mismatch kind is present; fixed order, no duplicates."""
    original = _canon(sql)
    if original is None:
        return []
    seen, out = {original}, []
    for name, kind, fn in RECIPES:
        if kind is not None and kind not in kinds:
            continue
        try:
            produced = fn(sql, side)
        except Exception:  # a recipe that cannot handle this SQL produces nothing
            produced = []
        for new in produced:
            canon = _canon(new)
            if canon is None or canon in seen:
                continue
            seen.add(canon)
            out.append((name, canon))
    return out
