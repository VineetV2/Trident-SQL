"""Pull one SQL query out of a model's final answer text."""
from __future__ import annotations

import re

_FENCE = re.compile(r"```(?:sqlite|sql)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
_BARE = re.compile(r"\b(WITH|SELECT)\b.*", re.IGNORECASE | re.DOTALL)


def extract_sql(text: str) -> str:
    """Return the last fenced SQL block; else text from the first WITH/SELECT; else ''.

    The last block is used because models sometimes show a draft before the final query.
    Trailing semicolons and whitespace are stripped. Never raises.
    """
    if not text:
        return ""
    blocks = [b.strip() for b in _FENCE.findall(text) if b.strip()]
    if blocks:
        sql = blocks[-1]
    else:
        m = _BARE.search(text)
        sql = m.group(0).strip() if m else ""
    return sql.rstrip().rstrip(";").strip()
