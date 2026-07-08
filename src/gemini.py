"""Thin wrappers around the Gemini REST API.

We call the REST endpoints directly with `requests` instead of the SDK so the
code is transparent, dependency-light, and version-proof. Two things are used:

  * embeddings  -> semantic search (turn text into vectors)
  * generation  -> summaries and RAG answers

Run this module directly to list the models your key can access:
    python -m src.gemini
"""
from __future__ import annotations

import time

import requests

from . import config

BASE = "https://generativelanguage.googleapis.com/v1beta"
TIMEOUT = 60


class GeminiError(RuntimeError):
    pass


def _require_key() -> str:
    if not config.GEMINI_API_KEY:
        raise GeminiError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
            "(free key: https://aistudio.google.com/apikey)."
        )
    return config.GEMINI_API_KEY


def embed_texts(
    texts: list[str], *, batch_size: int = 20, pause: float = 15.0, max_retries: int = 4
) -> list[list[float]]:
    """Embed a list of texts.

    The free tier meters embeddings per minute, so we send small batches with a
    short pause between them and back off on HTTP 429 (rate-limit) responses.
    """
    key = _require_key()
    url = f"{BASE}/models/{config.EMBED_MODEL}:batchEmbedContents?key={key}"
    vectors: list[list[float]] = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for b, start in enumerate(range(0, len(texts), batch_size), 1):
        chunk = texts[start : start + batch_size]
        payload = {
            "requests": [
                {"model": f"models/{config.EMBED_MODEL}", "content": {"parts": [{"text": t or " "}]}}
                for t in chunk
            ]
        }
        for attempt in range(max_retries):
            resp = requests.post(url, json=payload, timeout=TIMEOUT)
            if resp.ok:
                vectors.extend(item["values"] for item in resp.json()["embeddings"])
                break
            if resp.status_code == 429 and attempt < max_retries - 1:
                wait = 60 * (attempt + 1)
                print(f"  rate-limited, waiting {wait}s (batch {b}/{n_batches}, retry {attempt + 1}) ...")
                time.sleep(wait)
                continue
            raise GeminiError(f"Embedding request failed ({resp.status_code}): {resp.text[:300]}")

        print(f"  embedded batch {b}/{n_batches}")
        if b < n_batches:
            time.sleep(pause)

    return vectors


def embed_query(text: str, *, max_retries: int = 3) -> list[float]:
    """Embed a single query string, retrying on transient rate-limit."""
    key = _require_key()
    url = f"{BASE}/models/{config.EMBED_MODEL}:embedContent?key={key}"
    payload = {"model": f"models/{config.EMBED_MODEL}", "content": {"parts": [{"text": text}]}}
    for attempt in range(max_retries):
        resp = requests.post(url, json=payload, timeout=TIMEOUT)
        if resp.ok:
            return resp.json()["embedding"]["values"]
        if resp.status_code in (429, 503) and attempt < max_retries - 1:
            time.sleep(3 * (attempt + 1))
            continue
        raise GeminiError(f"Query embedding failed ({resp.status_code}): {resp.text[:300]}")
    raise GeminiError("Query embedding failed after retries")


def generate(prompt: str, *, temperature: float = 0.3, max_retries: int = 3) -> str:
    """Generate text from a prompt, retrying on transient rate-limit / overload."""
    key = _require_key()
    url = f"{BASE}/models/{config.GEN_MODEL}:generateContent?key={key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    for attempt in range(max_retries):
        resp = requests.post(url, json=payload, timeout=TIMEOUT)
        if resp.ok:
            data = resp.json()
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except (KeyError, IndexError):
                raise GeminiError(f"Unexpected generation response: {str(data)[:300]}")
        # 429 = rate limited, 503 = temporarily overloaded -> back off and retry.
        if resp.status_code in (429, 503) and attempt < max_retries - 1:
            time.sleep(3 * (attempt + 1))
            continue
        raise GeminiError(f"Generation failed ({resp.status_code}): {resp.text[:300]}")
    raise GeminiError("Generation failed after retries")


def list_models() -> list[dict]:
    """Return the models available to this key (useful for picking model names)."""
    key = _require_key()
    resp = requests.get(f"{BASE}/models?key={key}", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("models", [])


if __name__ == "__main__":
    # Handy diagnostic: which model names does my key support?
    for m in list_models():
        methods = ", ".join(m.get("supportedGenerationMethods", []))
        print(f"{m['name']:45s}  [{methods}]")
