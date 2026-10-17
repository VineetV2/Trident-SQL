"""THESIS-PATCH (llm-cache): common-random-numbers cache for the base stages' LLM.ask, gated by THESIS_LLM_CACHE_DIR.

Purpose: the selection-variant matrix re-runs the selection stage several times on the same candidates. Without a
cache every variant re-samples the pairwise judge, so two variants differ by judge sampling noise even where their
candidates agree. With the cache, the k-th time a given request is made in a process it gets the k-th stored answer
(sampled once, by the first process that needed it), so variants differ only where their inputs differ, and
the base stages' own retries (extractor re-asks after a parse failure) still get fresh samples (occurrence k+1).

Key = md5 over every request field that changes the answer: model, temperature, max_tokens, reasoning_effort,
extra_body, messages, n and any extra kwargs (timeout excluded). Unset = no cache = the default behaviour.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import Counter
from typing import Any, Dict, List, Optional

ENV = "THESIS_LLM_CACHE_DIR"
_LOCK = threading.Lock()
_SEEN: Counter = Counter()


def cache_dir() -> Optional[str]:
    path = (os.environ.get(ENV) or "").strip()
    return path or None


def request_key(params: Dict[str, Any]) -> str:
    body = {k: v for k, v in params.items() if k != "timeout"}
    return hashlib.md5(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def next_occurrence(key: str) -> int:
    with _LOCK:
        k = _SEEN[key]
        _SEEN[key] += 1
        return k


def _path(directory: str, key: str, occurrence: int) -> str:
    return os.path.join(directory, key[:2], f"{key}.{occurrence}.json")


def load(directory: str, key: str, occurrence: int) -> Optional[List[str]]:
    path = _path(directory, key, occurrence)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)["contents"]


def store(directory: str, key: str, occurrence: int, contents: List[str]) -> None:
    path = _path(directory, key, occurrence)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"contents": contents}, f, ensure_ascii=False)
    os.replace(tmp, path)
