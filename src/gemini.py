"""LangChain adapters for the project's existing Gemini models.

The public helpers deliberately retain batching and retry/backoff behaviour so
index builds remain friendly to Gemini rate limits.  Gemini itself is now
accessed only through LangChain's Google integration.
"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from . import config


class GeminiError(RuntimeError):
    """A user-facing error which FastAPI can return without a 500 response."""


def _require_key() -> str:
    if not config.GEMINI_API_KEY:
        raise GeminiError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
            "(free key: https://aistudio.google.com/apikey)."
        )
    return config.GEMINI_API_KEY


@lru_cache(maxsize=1)
def embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return the LangChain embedding client for the configured existing model."""
    # Loading a persisted FAISS index never calls Gemini. A harmless placeholder
    # lets the API keep its existing keyword-search fallback when no key is set;
    # embed_texts/embed_query still validate the key before making a request.
    return GoogleGenerativeAIEmbeddings(
        model=f"models/{config.EMBED_MODEL}",
        google_api_key=config.GEMINI_API_KEY or "not-used-for-local-faiss-load",
    )


def chat_model(*, temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Build a LangChain chat model without changing the configured Gemini model."""
    return ChatGoogleGenerativeAI(
        model=config.GEN_MODEL,
        google_api_key=_require_key(),
        temperature=temperature,
    )


def embed_texts(
    texts: list[str], *, batch_size: int = 20, pause: float = 15.0, max_retries: int = 4
) -> list[list[float]]:
    """Embed documents in bounded batches with retry/backoff through LangChain."""
    _require_key()
    client = embeddings()
    vectors: list[list[float]] = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_number, start in enumerate(range(0, len(texts), batch_size), 1):
        chunk = [text or " " for text in texts[start : start + batch_size]]
        for attempt in range(max_retries):
            try:
                vectors.extend(client.embed_documents(chunk))
                break
            except Exception as exc:  # provider errors do not expose stable exception classes
                if attempt == max_retries - 1:
                    raise GeminiError(f"Embedding request failed: {exc}") from exc
                wait = 15 * (attempt + 1)
                print(
                    f"  embedding request failed; waiting {wait}s "
                    f"(batch {batch_number}/{n_batches}, retry {attempt + 1}) ..."
                )
                time.sleep(wait)
        print(f"  embedded batch {batch_number}/{n_batches}")
        if batch_number < n_batches:
            time.sleep(pause)
    return vectors


def embed_query(text: str, *, max_retries: int = 3) -> list[float]:
    """Embed one query through the same LangChain embedding abstraction."""
    _require_key()
    client = embeddings()
    for attempt in range(max_retries):
        try:
            return client.embed_query(text)
        except Exception as exc:
            if attempt == max_retries - 1:
                raise GeminiError(f"Query embedding failed: {exc}") from exc
            time.sleep(3 * (attempt + 1))
    raise GeminiError("Query embedding failed after retries")


def list_models() -> list[dict[str, Any]]:
    """The LangChain integration does not expose model enumeration.

    Keep the old command callable while directing model discovery to AI Studio.
    """
    _require_key()
    return []


if __name__ == "__main__":
    print("Model listing is not provided by LangChain; see Google AI Studio for available models.")
