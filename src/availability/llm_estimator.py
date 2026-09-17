"""
LLM estimator for P(plays), same interface as the tabular baselines.

- Prompt: src/availability/prompt.py (anonymized by default).
- Model: Gemini via google-genai, JSON output, thinking off, temperature 0.
- Cache: every (model, prompt) pair is stored in availability.sqlite's
  llm_cache, so re-running an eval costs nothing and results are reproducible.
- Failures after retries return NaN; the eval script fills those from the
  status prior and reports the count.

No label ever enters the prompt: render_prompt reads only context columns.
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.availability.db import get_conn
from src.availability.prompt import SYSTEM_INSTRUCTIONS, render_prompt
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 5


class _RateLimiter:
    def __init__(self, calls_per_minute: int):
        self._interval = 60.0 / max(calls_per_minute, 1)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            gap = self._interval - (time.monotonic() - self._last)
            if gap > 0:
                time.sleep(gap)
            self._last = time.monotonic()


def prompt_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\n{SYSTEM_INSTRUCTIONS}\n{prompt}".encode()).hexdigest()


def _parse(text: str) -> dict:
    raw = json.loads(text)
    p = float(raw["p_play"])
    share = float(raw.get("expected_minutes_share", np.nan))
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p_play out of range: {p}")
    return {
        "p_play": p,
        "expected_minutes_share": min(max(share, 0.0), 1.0) if not np.isnan(share) else np.nan,
        "rationale": str(raw.get("rationale", ""))[:500],
    }


class GeminiClient:
    """Thin wrapper so tests can substitute a fake with the same .complete(prompt) -> text."""

    def __init__(self, model: str, calls_per_minute: int):
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))
        self.model = model
        self._limiter = _RateLimiter(calls_per_minute)

    def complete(self, prompt: str) -> str:
        self._limiter.wait()
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTIONS,
                response_mime_type="application/json",
                temperature=0.0,
                thinking_config=self._types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text


class LLMEstimator:
    """predict(df) -> P(plays) per row. `fit` is a no-op (no training)."""

    def __init__(
        self,
        variant: str = "anonymized",
        client=None,
        db_path: str | None = None,
        parallel_workers: int | None = None,
        model: str | None = None,
    ):
        cfg = load_config().availability_agent
        self.variant = variant
        self.name = "llm" if variant == "anonymized" else "llm_named"
        self.model = model or cfg.llm_model
        self.db_path = db_path or cfg.db_path
        self.workers = parallel_workers or cfg.parallel_workers
        self._client = client or GeminiClient(self.model, cfg.api_calls_per_minute)
        self.last_details: pd.DataFrame | None = None
        self.n_failed = 0

    def fit(self, df: pd.DataFrame):
        return self

    # -- cache -------------------------------------------------------------
    def _cached(self, conn: sqlite3.Connection, keys: list[str]) -> dict[str, dict]:
        out = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            q = f"SELECT prompt_hash, response_json FROM llm_cache WHERE prompt_hash IN ({','.join('?' * len(chunk))})"
            for h, js in conn.execute(q, chunk):
                out[h] = json.loads(js)
        return out

    def _store(self, conn: sqlite3.Connection, key: str, prompt: str, parsed: dict) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache (prompt_hash, model, variant, prompt, response_json, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (
                key,
                self.model,
                self.variant,
                prompt,
                json.dumps(parsed),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    # -- calls -------------------------------------------------------------
    def _call(self, prompt: str) -> dict | None:
        for attempt in range(_MAX_RETRIES):
            try:
                return _parse(self._client.complete(prompt))
            except Exception as e:  # network, quota, malformed JSON
                if "PerDay" in str(e):
                    raise RuntimeError(f"daily quota exhausted: {e}") from e
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_BASE_DELAY * (2**attempt))
                else:
                    logger.warning(f"LLM call failed after {_MAX_RETRIES} attempts: {e}")
        return None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        prompts = [render_prompt(row, self.variant) for _, row in df.iterrows()]
        keys = [prompt_key(self.model, p) for p in prompts]
        conn = get_conn(self.db_path)
        results = self._cached(conn, keys)
        todo = [(k, p) for k, p in dict(zip(keys, prompts)).items() if k not in results]
        logger.info(f"{self.name}: {len(df)} rows, {len(results)} cached, {len(todo)} to call")

        lock = threading.Lock()
        done = 0

        def work(item):
            nonlocal done
            k, p = item
            parsed = self._call(p)
            with lock:
                if parsed is not None:
                    results[k] = parsed
                    self._store(conn, k, p, parsed)
                done += 1
                if done % 200 == 0:
                    conn.commit()
                    logger.info(f"{self.name}: {done}/{len(todo)} calls done")

        if todo:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                list(ex.map(work, todo))
            conn.commit()
        conn.close()

        rows = [results.get(k) for k in keys]
        self.n_failed = sum(r is None for r in rows)
        self.last_details = pd.DataFrame(
            [r or {"p_play": np.nan, "expected_minutes_share": np.nan, "rationale": ""} for r in rows],
            index=df.index,
        )
        return self.last_details["p_play"].to_numpy(dtype=float)
