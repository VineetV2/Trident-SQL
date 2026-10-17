"""The value-finding stage's retrieved values -> the value-hit rows stage4_generation.plan_and_fragment reads (`--values`), for prompt fix literal_values.

Source: the finalized schema-linking snapshot's items, pipeline_artifacts.value_retrieval.retrieved_values, shaped
{table: {column: [{"value": str, "distance": float}, ...]}} (checked on runs/2026-10-05_final_train_learn2000:
"1." for keyword "1" has distance 0.066, unrelated image URLs ~0.60, so lower = closer). Kept: distance <= 0.3, at
most 5 distinct values per column, closest first. Output row: {"question_id", "hits": [{"table", "column", "value",
"score", "distance"}]} with score = 1 - distance, because plan_and_fragment/describe.py `_group_hits` orders by score
(highest first) and only uses hits on the columns it renders.
"""
from __future__ import annotations

from typing import Iterable, Optional

MAX_DISTANCE = 0.3
MAX_PER_COLUMN = 5


def _distance(hit) -> Optional[float]:
    """The hit's distance; a similarity-style field ('similarity' / 'score') is turned into 1 - similarity."""
    if not isinstance(hit, dict):
        return None
    for key, flip in (("distance", False), ("similarity", True), ("score", True)):
        if hit.get(key) is not None:
            try:
                value = float(hit[key])
            except (TypeError, ValueError):
                return None
            return 1.0 - value if flip else value
    return None


def column_hits(table: str, column: str, raw: Iterable) -> list[dict]:
    ranked = sorted((d, str(h.get("value"))) for h in raw or [] if (d := _distance(h)) is not None
                    and h.get("value") is not None and d <= MAX_DISTANCE)
    out, seen = [], set()
    for dist, value in ranked:
        if value in seen:
            continue
        seen.add(value)
        out.append({"table": table, "column": column, "value": value,
                    "score": round(1.0 - dist, 6), "distance": round(dist, 6)})
        if len(out) == MAX_PER_COLUMN:
            break
    return out


def hits_from_retrieved(retrieved: Optional[dict]) -> list[dict]:
    hits: list[dict] = []
    for table, cols in (retrieved or {}).items():
        for column, raw in (cols or {}).items():
            hits += column_hits(table, column, raw)
    return hits
