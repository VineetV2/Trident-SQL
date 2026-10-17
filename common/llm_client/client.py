"""Client for openai/gpt-oss-120b behind vLLM's OpenAI-compatible server (/v1/chat/completions).

Every call uses the model settings file that pipeline.py writes into the run folder:
  - payload always carries model, messages, n, temperature, top_p, max_tokens and
    reasoning_effort (if reasoning_effort is omitted vLLM silently runs at medium); stream=false
  - answer from message.content, chain of thought from message.reasoning (or reasoning_content)
  - finish_reason == "length" -> truncated; blank content -> empty; raw harmony markers in
    content -> error="harmony_leak" (server started without a reasoning parser); never repaired

Reliability (project pipeline rules):
  - at most 3 requests in flight against the 120B server (semaphore + thread pool)
  - per-request timeout = request_timeout_s; retry connection errors and 5xx with exponential
    backoff; never retry 4xx or a read timeout (a 600 s timeout means runaway generation)
  - after the final failure return Completions with error set; never raise to the caller
  - disk cache keyed by md5 over {model, temperature, top_p, reasoning_effort, max_tokens, n,
    run_id, messages}; only error-free responses are cached; files written atomically
  - one JSONL usage line per call in <cache_dir>/usage.jsonl for cost measurement
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import requests

from common.llm_client.settings import ModelSettings, load_settings

__all__ = ["Completion", "LLMClient", "ModelSettings", "cache_key", "health_check", "load_settings"]

log = logging.getLogger(__name__)

CHAT_PATH = "/v1/chat/completions"
MODELS_PATH = "/v1/models"
HARMONY_MARKERS = ("<|channel|>", "<|message|>", "<|start|>")
MAX_CONCURRENCY_LIMIT = 64  # was 3 (user rule); raised by the user for B2 on 2026-09-25 (D35: 32, D38: 64)


@dataclass(frozen=True)
class Completion:
    content: str
    reasoning: str
    finish_reason: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool
    truncated: bool
    empty: bool
    error: Optional[str] = None


class _RequestFailed(Exception):
    """Final failure: becomes Completion.error."""


class _Retryable(Exception):
    """Transient failure (connection error, 5xx): retried with backoff."""


def cache_key(settings: ModelSettings, messages: list[dict], n: int, run_id: str, salt: str = "") -> str:
    """md5 over every parameter that changes the output. Never drop temperature or run_id.

    `salt` separates repeated identical requests that must get fresh samples (e.g. a parse-retry
    loop re-sending the same prompt); empty salt leaves the key unchanged (old caches stay valid).
    """
    material = {
        "model": settings.model,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "reasoning_effort": settings.reasoning_effort,
        "max_tokens": settings.max_output_tokens,
        "n": n,
        "run_id": run_id,
        "messages": messages,
    }
    if salt:
        material["salt"] = salt
    blob = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def build_payload(settings: ModelSettings, messages: list[dict], n: int) -> dict:
    return {
        "model": settings.model,
        "messages": messages,
        "n": n,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_output_tokens,
        "reasoning_effort": settings.reasoning_effort,
        "stream": False,
    }


def _parse_choice(choice: dict, usage: dict, latency_s: float, cached: bool) -> Completion:
    message = choice.get("message") or {}
    content = message.get("content") or ""  # pre-PR#30608 truncation returned null
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    finish = choice.get("finish_reason")
    leaked = any(marker in content for marker in HARMONY_MARKERS)
    return Completion(
        content=content,
        reasoning=reasoning,
        finish_reason=finish,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        latency_s=latency_s,
        cached=cached,
        truncated=finish == "length",
        empty=not content.strip(),
        error="harmony_leak" if leaked else None,
    )


def parse_response(body: object, n: int, latency_s: float, cached: bool) -> list[Completion]:
    """Turn a chat-completions JSON body into n Completions. Usage is response-level (shared)."""
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or len(choices) != n:
        return error_completions(n, "bad_response", latency_s)
    usage = body.get("usage") or {}
    ordered = sorted(choices, key=lambda c: c.get("index", 0))
    return [_parse_choice(c, usage, latency_s, cached) for c in ordered]


def error_completions(n: int, error: str, latency_s: float = 0.0) -> list[Completion]:
    return [Completion("", "", None, 0, 0, latency_s, False, False, True, error) for _ in range(n)]


class LLMClient:
    """Thread-safe chat client with disk cache, retries and bounded concurrency."""

    def __init__(
        self,
        settings: ModelSettings,
        base_url: str,
        cache_dir: str,
        run_id: str,
        max_concurrency: int = 3,
        max_retries: int = 2,
        backoff_s: float = 2.0,
    ) -> None:
        if not 1 <= max_concurrency <= MAX_CONCURRENCY_LIMIT:
            raise ValueError(f"max_concurrency must be in 1..{MAX_CONCURRENCY_LIMIT}, got {max_concurrency}")
        if max_retries < 0 or backoff_s < 0:
            raise ValueError("max_retries and backoff_s must be >= 0")
        if not run_id:
            raise ValueError("run_id is required (it separates independent runs in the cache)")
        self.settings = settings
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.run_id = run_id
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._log_lock = threading.Lock()
        os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------ public API

    def with_temperature(self, temperature: float) -> "LLMClient":
        """A client identical to this one except the temperature (D61: sql_revision at 0.6/0.7/0.8). It shares
        this client's in-flight slots and usage-log lock, so the concurrency cap still covers both; the cache key
        includes the temperature, so the answers never mix."""
        clone = copy.copy(self)
        clone.settings = dataclasses.replace(self.settings, temperature=float(temperature))
        return clone

    def chat(self, messages: list[dict], n: int = 1, tag: Optional[str] = None, salt: str = "") -> list[Completion]:
        """One request -> n Completions. Failures come back as Completions with error set."""
        if not messages or n < 1:
            raise ValueError("chat needs non-empty messages and n >= 1")
        key = cache_key(self.settings, messages, n, self.run_id, salt)
        try:
            completions = self._chat_uncaught(key, messages, n)
        except Exception as exc:  # graceful degradation: the pipeline must keep going
            log.exception("chat failed unexpectedly for key %s", key)
            completions = error_completions(n, f"internal: {type(exc).__name__}: {exc}")
        self._log_usage(tag, key, n, completions)
        return completions

    def chat_many(self, requests_: list[dict]) -> list[list[Completion]]:
        """Run many {"messages", "n"?, "tag"?} requests, <= max_concurrency at a time, in input order."""
        def one(req: dict) -> list[Completion]:
            return self.chat(req["messages"], n=req.get("n", 1), tag=req.get("tag"))

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            return list(pool.map(one, requests_))

    # ------------------------------------------------------------ internals

    def _chat_uncaught(self, key: str, messages: list[dict], n: int) -> list[Completion]:
        cached = self._cache_read(key)
        if cached is not None:
            return parse_response(cached, n, 0.0, cached=True)
        payload = build_payload(self.settings, messages, n)
        start = time.monotonic()
        try:
            body = self._post_with_retries(payload)
        except _RequestFailed as exc:
            log.warning("request %s failed: %s", key, exc)
            return error_completions(n, str(exc), time.monotonic() - start)
        completions = parse_response(body, n, time.monotonic() - start, cached=False)
        if all(c.error is None for c in completions):
            self._cache_write(key, body)
        return completions

    def _post_with_retries(self, payload: dict) -> dict:
        last_error = "unknown"
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(self.backoff_s * 2 ** (attempt - 1))
            try:
                return self._post_once(payload)
            except _Retryable as exc:
                last_error = str(exc)
                log.warning("attempt %d/%d failed: %s", attempt + 1, self.max_retries + 1, exc)
        raise _RequestFailed(last_error)

    def _post_once(self, payload: dict) -> dict:
        url = self.base_url + CHAT_PATH
        with self._slots:
            try:
                resp = requests.post(url, json=payload, timeout=self.settings.request_timeout_s)
            except requests.ConnectTimeout as exc:
                raise _Retryable(f"connection_error: {exc}") from exc
            except requests.Timeout as exc:
                raise _RequestFailed(f"timeout after {self.settings.request_timeout_s}s") from exc
            except requests.ConnectionError as exc:
                raise _Retryable(f"connection_error: {exc}") from exc
        if resp.status_code >= 500:
            raise _Retryable(f"http_{resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise _RequestFailed(f"http_{resp.status_code}: {resp.text[:200]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise _RequestFailed("bad_response") from exc

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, key[:2], f"{key}.json")

    def _cache_read(self, key: str) -> Optional[dict]:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return json.load(f)["response"]
        except (OSError, ValueError, KeyError) as exc:
            log.warning("ignoring unreadable cache file %s: %s", path, exc)
            return None

    def _cache_write(self, key: str, body: dict) -> None:
        path = self._cache_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{uuid.uuid4().hex}.tmp"
        record = {"key": key, "run_id": self.run_id, "response": body}
        try:
            with open(tmp, "w") as f:
                json.dump(record, f, ensure_ascii=False)
            os.replace(tmp, path)
        except OSError as exc:
            log.warning("cache write failed for %s: %s", key, exc)
            if os.path.exists(tmp):
                os.remove(tmp)

    def _log_usage(self, tag: Optional[str], key: str, n: int, completions: list[Completion]) -> None:
        first = completions[0]
        errors = [c.error for c in completions if c.error]
        entry = {
            "ts": time.time(),
            "run_id": self.run_id,
            "tag": tag,
            "key": key,
            "n": n,
            "cached": first.cached,
            "prompt_tokens": first.prompt_tokens,
            "completion_tokens": first.completion_tokens,
            "latency_s": round(first.latency_s, 3),
            "finish_reason": [c.finish_reason for c in completions],
            "error": errors[0] if errors else None,
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._log_lock, open(os.path.join(self.cache_dir, "usage.jsonl"), "a") as f:
            f.write(line)


def health_check(base_url: str, model: str, timeout: float = 10.0) -> bool:
    """True when GET /v1/models answers and lists `model`. Never raises."""
    try:
        resp = requests.get(base_url.rstrip("/") + MODELS_PATH, timeout=timeout)
        resp.raise_for_status()
        return any(m.get("id") == model for m in resp.json().get("data", []))
    except (requests.RequestException, ValueError, AttributeError) as exc:
        log.info("health check failed for %s: %s", base_url, exc)
        return False
