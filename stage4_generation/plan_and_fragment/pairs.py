"""Column-name tokens, the one part kept of the look-alike column pair finder (used by writers this repository
does not include). stage6_query_tree.qtree.recipes imports name_tokens.
"""
from __future__ import annotations

import re

NAME_STOP = {"id", "name", "no", "num", "code", "type", "date", "the", "of", "a", "key", "desc", "flag", "text",
             "home", "away", "player", "class", "team", "api"}


def name_tokens(name: str) -> set[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return {t for t in re.split(r"[^a-zA-Z0-9]+", s.lower()) if len(t) >= 3} - NAME_STOP
