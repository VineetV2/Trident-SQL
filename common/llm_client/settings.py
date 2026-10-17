"""Model settings for gpt-oss-120b, loaded from the YAML file pipeline.py writes into the run folder.

The YAML is the single source of truth. Only the fields the client needs are
required; everything else in the file (hf_revision, vllm_version, ...) is ignored here and may
still be TBD. A required field that is missing, null, "TBD" or out of range raises ValueError
naming the field, so a half-filled config can never silently reach the server.
"""
from __future__ import annotations

from dataclasses import dataclass

import yaml

REASONING_EFFORTS = frozenset({"low", "medium", "high"})
REQUIRED_FIELDS = ("model", "reasoning_effort", "temperature", "top_p", "max_output_tokens", "request_timeout_s")


@dataclass(frozen=True)
class ModelSettings:
    model: str
    reasoning_effort: str
    temperature: float
    top_p: float
    max_output_tokens: int
    request_timeout_s: float


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_required(raw: dict) -> None:
    for name in REQUIRED_FIELDS:
        value = raw.get(name)
        if value is None or (isinstance(value, str) and value.strip().upper() in {"", "TBD"}):
            raise ValueError(f"model settings: required field '{name}' is missing or TBD (got {value!r})")


def _validate(raw: dict) -> ModelSettings:
    _check_required(raw)
    if not isinstance(raw["model"], str):
        raise ValueError(f"model settings: 'model' must be a string, got {raw['model']!r}")
    if raw["reasoning_effort"] not in REASONING_EFFORTS:
        raise ValueError(
            f"model settings: 'reasoning_effort' must be one of {sorted(REASONING_EFFORTS)}, "
            f"got {raw['reasoning_effort']!r}"
        )
    for name in ("temperature", "top_p"):
        if not _is_number(raw[name]):
            raise ValueError(f"model settings: '{name}' must be a number, got {raw[name]!r}")
    tokens = raw["max_output_tokens"]
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        raise ValueError(f"model settings: 'max_output_tokens' must be an int > 0, got {tokens!r}")
    timeout = raw["request_timeout_s"]
    if not _is_number(timeout) or timeout <= 0:
        raise ValueError(f"model settings: 'request_timeout_s' must be a number > 0, got {timeout!r}")
    return ModelSettings(
        model=raw["model"],
        reasoning_effort=raw["reasoning_effort"],
        temperature=float(raw["temperature"]),
        top_p=float(raw["top_p"]),
        max_output_tokens=tokens,
        request_timeout_s=float(timeout),
    )


def load_settings(path: str) -> ModelSettings:
    """Read and validate the model settings YAML."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"model settings: {path} is not a YAML mapping")
    return _validate(raw)
