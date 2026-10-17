"""Chat messages for the literal generator.

Templates live in prompts.py (source and changes documented there). Nothing here
touches gold SQL of the target question: callers pass question text, evidence and schema only.
"""
from __future__ import annotations

from stage4_generation.plan_and_fragment import prompts as P


def _system(head: str) -> str:
    return "\n\n".join([head, P.SQLITE_RULES, P.OUTPUT_FORMAT])


def _input(question: str, evidence: str, schema: str) -> str:
    ev = (evidence or "").strip() or "(none)"
    body = P.INPUT_TEMPLATE.format(schema=schema, evidence=ev, question=question.strip())
    return body + "\n\n" + P.EVIDENCE_NOTE


def literal_messages(question: str, evidence: str, schema: str) -> list[dict]:
    return [
        {"role": "system", "content": _system(P.LITERAL_HEAD)},
        {"role": "user", "content": _input(question, evidence, schema)},
    ]
