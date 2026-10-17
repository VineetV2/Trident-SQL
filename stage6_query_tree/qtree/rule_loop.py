"""One-rule-at-a-time tree repair loop with a re-check and a data gate: after each edit the query tree comparison runs
again, and only the rule that matches a remaining finding is applied next.

Per candidate SQL:
  1. compare the question side with the candidate's query tree -> finding kinds (stage6_query_tree.qtree.match);
  2. take the FIRST rule (file order) whose tree_findings the candidate carries and that was not tried yet;
     no such rule -> stop;
  3. ask the model with that ONE rule (the caller builds the prompt: plain or with measured facts);
  4. accept the edit only if ALL hold, else keep the current SQL:
       - the answer parses and the SQL actually changed;
       - the answer cites the rule it was shown (cited-rule guard);
       - tree re-check: the edited SQL no longer carries the rule's own finding, and carries no finding the
         current SQL did not have;
       - data gate: the edited SQL runs, returns rows when the current one did, is not all NULL when the current one
         was not, and returns a DIFFERENT result (an edit that changes nothing is not taken); a row-capped result on
         either side cannot be compared and the edit is not taken; when the CURRENT SQL does not run, any edit that
         runs is taken (it repairs an execution failure);
  5. go back to 1 with the (possibly edited) SQL. Each rule is asked at most once per candidate; at most
     `max_calls` model calls per candidate.
The gate was chosen on measured evidence before this module existed (pre-registered, round 5): on earlier runs it
kept every fixing edit and caught some breaking ones; a "never drop DISTINCT on a key column" gate was measured too
and rejected (it lost many of the fixes).

Pure logic: no model client, no database. The caller passes `ask(rule, sql) -> {"applied_rules", "corrected_sql"} | None
| PROMPT_TOO_LONG` and `execute(sql) -> (ok, rows | None, error | None)` (error "rows_capped" for a capped result);
`finder(sql) -> frozenset of kinds` defaults to the real tree check; `on_exception(where, exc)` is told about every
exception the loop absorbs (the caller logs it; a systematic failure must not look like "no effect").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

DEFAULT_MAX_CALLS = 3
PROMPT_TOO_LONG = "__prompt_too_long__"   # ask() sentinel: no model call was made
ROWS_CAPPED = "rows_capped"

Ask = Callable[[dict, str], Optional[Dict[str, Any]]]
Execute = Callable[[str], Tuple[bool, Optional[Sequence[Sequence[Any]]], Optional[str]]]
Finder = Callable[[str], FrozenSet[str]]


@dataclass
class Step:
    rule: str
    status: str                 # accepted | rejected | no_edit | no_answer
    reason: str = ""
    proposed_sql: Optional[str] = None
    votes: Optional[Dict[str, int]] = None   # agreement mode only

    def as_dict(self) -> Dict[str, Any]:
        d = {"rule": self.rule, "status": self.status}
        if self.reason:
            d["reason"] = self.reason
        if self.proposed_sql is not None and self.status != "accepted":
            d["proposed_sql"] = self.proposed_sql
        if self.votes is not None:
            d["votes"] = self.votes
        return d


@dataclass
class LoopResult:
    sql: str
    applied: List[str] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    findings_before: FrozenSet[str] = frozenset()
    findings_after: FrozenSet[str] = frozenset()
    calls: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.applied)


def norm_sql(sql: str) -> str:
    """Loose key (whitespace collapsed, lower case): used only by callers for logging / memo, never for decisions."""
    return " ".join((sql or "").split()).strip().rstrip(";").strip().lower()


def same_text(a: str, b: str) -> bool:
    """Case-SENSITIVE comparison with whitespace collapsed: a change of a literal's case is an edit."""
    return " ".join((a or "").split()).strip().rstrip(";").strip() == " ".join((b or "").split()).strip().rstrip(";").strip()


def _rows_set(rows: Optional[Sequence[Sequence[Any]]]) -> frozenset:
    return frozenset(tuple(r) for r in (rows or ()))


def _all_null(rows: Optional[Sequence[Sequence[Any]]]) -> bool:
    rows = list(rows or ())
    return bool(rows) and all(v is None for r in rows for v in r)


def data_gate(before: Tuple[bool, Any, Any], after: Tuple[bool, Any, Any]) -> Tuple[bool, str]:
    """(accept?, reason) for an edit, from the execution outcome of the current SQL and of the edited SQL."""
    b_ok, b_rows, b_err = before
    a_ok, a_rows, a_err = after
    if a_err == ROWS_CAPPED or b_err == ROWS_CAPPED:
        return False, "cannot_compare:" + ROWS_CAPPED
    if not a_ok:
        return False, f"edit_does_not_run:{a_err}"
    if not b_ok:
        return True, "ok_current_did_not_run"
    if b_rows and not a_rows:
        return False, "empty_result"
    if _all_null(a_rows) and not _all_null(b_rows):
        return False, "all_null_result"
    if _rows_set(a_rows) == _rows_set(b_rows):
        return False, "result_unchanged"
    return True, "ok"


def default_finder(side) -> Finder:
    from stage6_query_tree.qtree.cards import finding_kinds
    return lambda sql: finding_kinds(sql, side)


def run_loop(sql: str, rules: Sequence[dict], ask: Ask, execute: Execute, finder: Finder,
             max_calls: int = DEFAULT_MAX_CALLS,
             on_exception: Optional[Callable[[str, BaseException], None]] = None,
             samples: int = 1, min_agree: Optional[int] = None) -> LoopResult:
    """The loop described in the module docstring. Never raises for a model or database failure (they reject the step).

    Agreement (round 6): `ask` may return a LIST of `samples` answers for one rule. Each answer is judged alone
    (cited rule, tree re-check, data gate); the edit is taken only when at least `min_agree` answers (default: a
    majority of `samples`) propose accepted edits with the SAME result rows. Missing or failed answers count as
    "keep". With samples = 1 this is exactly the single-answer behaviour."""
    if samples < 1:
        raise ValueError("samples must be >= 1")
    need = min_agree if min_agree is not None else samples // 2 + 1

    def report(where: str, exc: BaseException) -> None:
        if on_exception is not None:
            try:
                on_exception(where, exc)
            except Exception:
                pass

    current = sql
    first = finder(current)
    res = LoopResult(sql=current, findings_before=first)
    tried = set()
    exec_cache: Dict[str, Tuple[bool, Any, Any]] = {}

    def run(q: str):
        key = (q or "").strip().rstrip(";").strip()   # exact text: literals keep their case and spacing
        if key not in exec_cache:
            try:
                exec_cache[key] = execute(q)
            except Exception as e:  # an execution adapter must not break the loop
                report("execute", e)
                exec_cache[key] = (False, None, f"exception:{type(e).__name__}")
        return exec_cache[key]

    def judge(rule: dict, answer: Any, kinds: FrozenSet[str]) -> Tuple[str, str, Optional[str], Any]:
        """One answer -> (decision: edit | no_answer | no_edit | rejected, reason, proposed SQL, after-kinds)."""
        name = rule["rule_name"]
        if not isinstance(answer, dict) or not isinstance(answer.get("corrected_sql"), str) or not answer["corrected_sql"].strip():
            return "no_answer", "parse_failed_or_empty", None, None
        proposed = answer["corrected_sql"].strip()
        if same_text(proposed, current):
            return "no_edit", "", None, None
        cited = answer.get("applied_rules")
        if not isinstance(cited, list) or name not in cited:
            return "rejected", "edit_without_cited_rule", proposed, None
        try:
            after_kinds = finder(proposed)
        except Exception as e:
            report("finder", e)
            return "rejected", f"tree_recheck:exception:{type(e).__name__}", proposed, None
        if set(rule.get("tree_findings") or ()) & after_kinds:
            return "rejected", "tree_recheck:own_finding_remains", proposed, None
        new = sorted(after_kinds - kinds)
        if new:
            return "rejected", "tree_recheck:new_finding:" + ",".join(new), proposed, None
        try:
            ok, reason = data_gate(run(current), run(proposed))
        except Exception as e:
            report("data_gate", e)
            ok, reason = False, f"exception:{type(e).__name__}"
        if not ok:
            return "rejected", "data_gate:" + reason, proposed, None
        return "edit", reason, proposed, after_kinds

    kinds = first
    attempts = 0
    while attempts < max_calls:
        rule = next((r for r in rules if r["rule_name"] not in tried and set(r.get("tree_findings") or ()) & kinds), None)
        if rule is None:
            break
        name = rule["rule_name"]
        tried.add(name)
        attempts += 1
        try:
            answer = ask(rule, current)
        except Exception as e:
            report("ask", e)
            res.calls += 1
            res.steps.append(Step(name, "no_answer", f"exception:{type(e).__name__}: {e}"[:200]))
            continue
        if answer == PROMPT_TOO_LONG:
            res.steps.append(Step(name, "no_answer", "prompt_too_long"))
            continue
        res.calls += 1
        answers = list(answer) if isinstance(answer, list) else [answer]
        if samples == 1:
            decision, reason, proposed, after_kinds = judge(rule, answers[0] if answers else None, kinds)
            if decision != "edit":
                res.steps.append(Step(name, decision, reason, proposed))
                continue
            current, kinds = proposed, after_kinds
            res.applied.append(name)
            res.steps.append(Step(name, "accepted", reason))
            continue
        # agreement over several answers
        judged = [judge(rule, a, kinds) for a in answers[:samples]]
        groups: Dict[frozenset, List[Tuple[str, Any, str]]] = {}
        for decision, reason, proposed, after_kinds in judged:
            if decision == "edit":
                groups.setdefault(_rows_set(run(proposed)[1]), []).append((proposed, after_kinds, reason))
        tally = collections_counter(d for d, *_ in judged)
        votes = {"samples": samples, "answers": len(answers), "edit": tally.get("edit", 0),
                 "no_edit": tally.get("no_edit", 0), "rejected": tally.get("rejected", 0),
                 "no_answer": tally.get("no_answer", 0) + max(0, samples - len(answers)),
                 "largest_group": max((len(g) for g in groups.values()), default=0), "need": need}
        best = max(groups.values(), key=len, default=[])
        if len(best) >= need:
            proposed, after_kinds, reason = best[0]
            current, kinds = proposed, after_kinds
            res.applied.append(name)
            res.steps.append(Step(name, "accepted", f"agree:{len(best)}/{samples}", votes=votes))
            continue
        if not groups:
            first_proposal = next((p for d, _, p, _ in judged if p), None)
            res.steps.append(Step(name, "no_edit" if tally.get("no_edit", 0) >= need else "rejected",
                                  "" if tally.get("no_edit", 0) >= need else "no_agreement:no_valid_edit",
                                  first_proposal, votes=votes))
        else:
            res.steps.append(Step(name, "rejected", f"no_agreement:{len(best)}/{samples}", best[0][0], votes=votes))
    res.sql = current
    res.findings_after = kinds
    return res


def collections_counter(items):
    from collections import Counter
    return Counter(items)
