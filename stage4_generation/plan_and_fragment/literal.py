"""Literal head (generator 'literal'): the COUNT(DISTINCT) check and its one re-ask per flagged sample.

Designed and measured on train only. A sample is flagged when its
SQL uses COUNT(DISTINCT ...) and neither the question nor the evidence gives a distinct-values cue. On train
winners this construct is about twice as common in unanimous-wrong picks as in correct ones (21% vs 10%, slice);
the other constructs the prompt discourages (SELECT DISTINCT, IS NOT NULL, CTE) are not enriched, so they are left
to the prompt alone. The re-ask continues the sample's own conversation and lets the model keep DISTINCT when the
wording needs it. A failed or SQL-less re-ask keeps the original sample.
"""
from __future__ import annotations

import re

from stage4_generation.plan_and_fragment import prompts as P

_COUNT_DISTINCT = re.compile(r"\bcount\s*\(\s*distinct\b", re.IGNORECASE)
_CUE = re.compile(r"\b(distinct|different|unique|kinds?|types?)\b", re.IGNORECASE)


def count_distinct_flag(question: str, evidence: str, sql: str) -> bool:
    """True when `sql` uses COUNT(DISTINCT ...) and the question/evidence carries no distinct-values cue."""
    if not sql or not _COUNT_DISTINCT.search(sql):
        return False
    return not _CUE.search(f"{question or ''} {evidence or ''}")


def reask_messages(messages: list[dict], assistant_content: str) -> list[dict]:
    """The sample's conversation plus its own answer and the re-ask."""
    return list(messages) + [{"role": "assistant", "content": assistant_content},
                             {"role": "user", "content": P.LITERAL_REASK}]
