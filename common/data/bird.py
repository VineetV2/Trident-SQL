"""BIRD question record and database paths (dev and train share one layout)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Question:
    question_id: int
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    difficulty: Optional[str]


def db_path(db_root: str, db_id: str) -> str:
    return f"{db_root.rstrip('/')}/{db_id}/{db_id}.sqlite"
