"""In-memory vector store: loads the indexed jobs + embeddings and ranks them.

For a few thousand jobs, a brute-force cosine similarity in NumPy is instant and
keeps the project dependency-free. At larger scale you would swap this for a
proper vector index (FAISS, or a managed service like Azure AI Search) -- the
retrieval interface below would stay the same.
"""
from __future__ import annotations

import json

import numpy as np

from . import config


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows so a dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class JobStore:
    def __init__(self) -> None:
        self.jobs: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.load()

    @property
    def ready(self) -> bool:
        return bool(self.jobs) and self.embeddings is not None

    def load(self) -> None:
        if config.JOBS_PATH.exists() and config.EMBEDDINGS_PATH.exists():
            self.jobs = json.loads(config.JOBS_PATH.read_text(encoding="utf-8"))
            raw = np.load(config.EMBEDDINGS_PATH).astype("float32")
            self.embeddings = _normalize(raw)

    def semantic_search(
        self, query_vec: list[float], *, top_k: int = 8, category: str | None = None
    ) -> list[dict]:
        """Rank jobs by cosine similarity to the query vector."""
        if not self.ready:
            return []
        q = np.asarray(query_vec, dtype="float32")
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self.embeddings @ q  # cosine similarity, one dot product per job

        order = np.argsort(-scores)
        results: list[dict] = []
        for idx in order:
            job = self.jobs[idx]
            if category and job.get("category") != category:
                continue
            results.append({**job, "score": round(float(scores[idx]), 4)})
            if len(results) >= top_k:
                break
        return results

    def lexical_search(self, query: str, *, top_k: int = 8) -> list[dict]:
        """Keyword fallback used when no API key is configured."""
        terms = [t for t in query.lower().split() if t]
        scored = []
        for job in self.jobs:
            haystack = f"{job.get('title', '')} {job.get('company', '')} " \
                       f"{job.get('category', '')} {job.get('summary', '')}".lower()
            hits = sum(haystack.count(t) for t in terms)
            if hits:
                scored.append({**job, "score": hits})
        scored.sort(key=lambda j: -j["score"])
        return scored[:top_k]

    def categories(self) -> list[str]:
        return sorted({j.get("category", "") for j in self.jobs if j.get("category")})
