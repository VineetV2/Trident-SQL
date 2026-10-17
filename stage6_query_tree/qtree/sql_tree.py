"""SQL side of the query-tree comparison: a logical view of one SQL query (no execution, no schema needed).

The generated SQL becomes a query-plan tree. We keep the logical operators that carry meaning (outputs with their
aggregate, filters with their literal, join edges, grouping, ordering, limit, DISTINCT, number of query scopes), in
the same terms the question side uses, so the two can be matched by meaning rather than by words. Built on sqlglot's
parse tree; aliases are resolved to table names per query scope.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp

Col = Tuple[Optional[str], str]  # (table or None when it cannot be resolved without a schema, column), lowercased
AGG_TYPES = {exp.Count: "COUNT", exp.Sum: "SUM", exp.Avg: "AVG", exp.Min: "MIN", exp.Max: "MAX"}
CMP_TYPES = {exp.EQ: "=", exp.NEQ: "!=", exp.GT: ">", exp.GTE: ">=", exp.LT: "<", exp.LTE: "<=", exp.Like: "LIKE"}


@dataclass(frozen=True)
class Output:
    fn: Optional[str]          # COUNT/SUM/AVG/MIN/MAX or None for a plain value
    distinct: bool
    columns: FrozenSet[Col]


@dataclass(frozen=True)
class Filter:
    column: Col
    op: str
    value: str                 # literal as text; IN lists are joined with "|"


@dataclass
class SQLTree:
    ok: bool
    error: str = ""
    tables: Set[str] = field(default_factory=set)
    columns: Set[Col] = field(default_factory=set)
    outputs: List[Output] = field(default_factory=list)
    filters: List[Filter] = field(default_factory=list)
    # comparisons inside expressions (CASE WHEN / IIF / SUM(col = v) / ORDER BY ...), outside WHERE/HAVING/JOIN ON
    expr_filters: List[Filter] = field(default_factory=list)
    # IS NULL / IS NOT NULL conditions in WHERE / HAVING (kept apart from `filters` so the frozen checks that read
    # `filters` are unchanged); used by the fork finder only
    null_filters: List[Filter] = field(default_factory=list)
    joins: Set[FrozenSet[Col]] = field(default_factory=set)
    group_by: Set[Col] = field(default_factory=set)
    order_desc: List[bool] = field(default_factory=list)
    limit: Optional[int] = None
    agg_fns: Set[str] = field(default_factory=set)
    count_distinct: bool = False
    select_distinct: bool = False
    literals: List[str] = field(default_factory=list)
    scopes: int = 0
    table_columns_outside_joins: Dict[str, Set[str]] = field(default_factory=dict)
    output_aliases: Set[str] = field(default_factory=set)
    top_count_distinct: bool = False   # COUNT(DISTINCT ...) in the OUTERMOST select list (round 8 routing)
    ctes: int = 0


def _scope_tables(select: exp.Select) -> Dict[str, str]:
    """alias-or-name (lower) -> table name (lower) for the tables in this SELECT's own FROM/JOIN."""
    out: Dict[str, str] = {}
    for t in select.find_all(exp.Table):
        if t.find_ancestor(exp.Select) is select:
            out[(t.alias_or_name or t.name).lower()] = t.name.lower()
    return out


def _resolve(col: exp.Column, global_alias: Dict[str, str]) -> Col:
    name = col.name.lower()
    qual = (col.table or "").lower()
    scope = col.find_ancestor(exp.Select)
    local = _scope_tables(scope) if scope is not None else {}
    if qual:
        return (local.get(qual) or global_alias.get(qual) or qual, name)
    if len(set(local.values())) == 1:
        return (next(iter(local.values())), name)
    return (None, name)


def _literal_text(node: exp.Expression) -> Optional[str]:
    if isinstance(node, exp.Literal):
        return str(node.this)
    if isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal):
        return "-" + str(node.this.this)
    return None


def _agg_of(node: exp.Expression) -> Tuple[Optional[str], bool]:
    for typ, fn in AGG_TYPES.items():
        found = node if isinstance(node, typ) else node.find(typ)
        if found is not None:
            return fn, isinstance(found.this, exp.Distinct)
    return None, False


def parse_sql_tree(sql: str) -> SQLTree:
    try:
        root = sqlglot.parse_one(sql, read="sqlite")
    except Exception as e:  # unparseable SQL is reported, never raised
        return SQLTree(ok=False, error=str(e)[:200])
    if root is None or not isinstance(root, (exp.Select, exp.Union, exp.Subquery)) and root.find(exp.Select) is None:
        return SQLTree(ok=False, error="not a SELECT")
    t = SQLTree(ok=True)
    cte_names = {c.alias_or_name.lower() for c in root.find_all(exp.CTE)}
    t.ctes = len(cte_names)
    global_alias: Dict[str, str] = {}
    for tab in root.find_all(exp.Table):
        name = tab.name.lower()
        if name and name not in cte_names:
            t.tables.add(name)
            global_alias[(tab.alias_or_name or tab.name).lower()] = name
    selects = list(root.find_all(exp.Select))
    t.scopes = len(selects)
    t.select_distinct = any(s.args.get("distinct") is not None for s in selects)
    join_cols: Set[Col] = set()
    for j in root.find_all(exp.Join):
        on = j.args.get("on")
        for eq in (on.find_all(exp.EQ) if on is not None else []):
            if isinstance(eq.this, exp.Column) and isinstance(eq.expression, exp.Column):
                a, b = _resolve(eq.this, global_alias), _resolve(eq.expression, global_alias)
                t.joins.add(frozenset({a, b})); join_cols |= {a, b}
    for col in root.find_all(exp.Column):
        c = _resolve(col, global_alias)
        t.columns.add(c)
        if c not in join_cols and c[0]:
            t.table_columns_outside_joins.setdefault(c[0], set()).add(c[1])
    for node in root.find_all(*AGG_TYPES):
        t.agg_fns.add(AGG_TYPES[type(node)])
        if isinstance(node, exp.Count) and isinstance(node.this, exp.Distinct):
            t.count_distinct = True
    for lit in root.find_all(exp.Literal):
        t.literals.append(str(lit.this))
    # filters in WHERE / HAVING of every scope
    for clause in list(root.find_all(exp.Where)) + list(root.find_all(exp.Having)):
        for typ, op in CMP_TYPES.items():
            for cmp_node in clause.find_all(typ):
                left, right = cmp_node.this, cmp_node.expression
                if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                    a, b = _resolve(left, global_alias), _resolve(right, global_alias)
                    if a[0] != b[0]:
                        t.joins.add(frozenset({a, b}))
                    continue
                if not isinstance(left, exp.Column):
                    left, right = right, left
                val = _literal_text(right) if right is not None else None
                if isinstance(left, exp.Column) and val is not None:
                    t.filters.append(Filter(_resolve(left, global_alias), op, val))
        for node in clause.find_all(exp.In):
            vals = [_literal_text(e) for e in node.expressions]
            if isinstance(node.this, exp.Column) and vals and all(v is not None for v in vals):
                t.filters.append(Filter(_resolve(node.this, global_alias), "IN", "|".join(vals)))
        for node in clause.find_all(exp.Is):
            if isinstance(node.this, exp.Column) and isinstance(node.expression, exp.Null):
                op = "IS NOT NULL" if isinstance(node.parent, exp.Not) else "IS NULL"
                t.null_filters.append(Filter(_resolve(node.this, global_alias), op, "NULL"))
        for node in clause.find_all(exp.Between):
            lo, hi = _literal_text(node.args.get("low")), _literal_text(node.args.get("high"))
            if isinstance(node.this, exp.Column) and lo is not None and hi is not None:
                t.filters.append(Filter(_resolve(node.this, global_alias), "BETWEEN", f"{lo}|{hi}"))
    # comparisons inside expressions: same extraction, for nodes with no WHERE / HAVING / JOIN ancestor
    for typ, op in CMP_TYPES.items():
        for cmp_node in root.find_all(typ):
            if cmp_node.find_ancestor(exp.Where, exp.Having, exp.Join) is not None:
                continue
            left, right = cmp_node.this, cmp_node.expression
            if not isinstance(left, exp.Column):
                left, right = right, left
            val = _literal_text(right) if right is not None else None
            if isinstance(left, exp.Column) and val is not None:
                t.expr_filters.append(Filter(_resolve(left, global_alias), op, val))
    for node in root.find_all(exp.In):
        if node.find_ancestor(exp.Where, exp.Having, exp.Join) is not None:
            continue
        vals = [_literal_text(e) for e in node.expressions]
        if isinstance(node.this, exp.Column) and vals and all(v is not None for v in vals):
            t.expr_filters.append(Filter(_resolve(node.this, global_alias), "IN", "|".join(vals)))
    top = root if isinstance(root, exp.Select) else root.find(exp.Select)
    t.top_count_distinct = any(isinstance(cnt.this, exp.Distinct) and cnt.find_ancestor(exp.Select) is top
                               for proj in top.expressions for cnt in proj.find_all(exp.Count))
    for proj in top.expressions:
        if isinstance(proj, exp.Alias):
            t.output_aliases.add(proj.alias.lower())
        inner = proj.this if isinstance(proj, exp.Alias) else proj
        fn, dist = _agg_of(inner)
        t.outputs.append(Output(fn, dist, frozenset(_resolve(c, global_alias) for c in inner.find_all(exp.Column))))
    group = top.args.get("group")
    if group is not None:
        t.group_by = {_resolve(c, global_alias) for e in group.expressions for c in
                      ([e] if isinstance(e, exp.Column) else e.find_all(exp.Column))}
    order = top.args.get("order")
    if order is not None:
        t.order_desc = [bool(o.args.get("desc")) for o in order.expressions]
    limit = top.args.get("limit")
    if limit is not None:
        try:
            t.limit = int(_literal_text(limit.expression) or "")
        except (TypeError, ValueError):
            t.limit = None
    return t
