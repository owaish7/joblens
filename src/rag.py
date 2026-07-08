"""Retrieval-Augmented Generation layer.

Two capabilities on top of the vector store:
  * search()  -> semantic retrieval, with an optional AI overview of the results
  * ask()     -> RAG: retrieve relevant jobs, then have Gemini answer grounded
                 ONLY in those jobs, with inline [n] citations.
"""
from __future__ import annotations

from . import config, gemini
from .store import JobStore

store = JobStore()


def _has_key() -> bool:
    return bool(config.GEMINI_API_KEY)


def search(query: str, *, top_k: int = 8, category: str | None = None) -> dict:
    """Semantic search (falls back to keyword search without an API key)."""
    if not store.ready:
        return {"mode": "empty", "results": [], "message": "Index not built yet. Run: python -m src.ingest"}

    if _has_key():
        query_vec = gemini.embed_query(query)
        results = store.semantic_search(query_vec, top_k=top_k, category=category)
        return {"mode": "semantic", "results": results}

    results = store.lexical_search(query, top_k=top_k)
    return {"mode": "lexical", "results": results,
            "message": "No API key set -- using keyword search. Add GEMINI_API_KEY for semantic search."}


def overview(query: str, jobs: list[dict]) -> str:
    """A 2-3 sentence AI summary of what the matched jobs have in common."""
    if not (_has_key() and jobs):
        return ""
    listing = "\n".join(f"- {j['title']} at {j['company']} ({j['location']})" for j in jobs[:8])
    prompt = (
        f'A user searched for "{query}". Here are the top matching remote jobs:\n{listing}\n\n'
        "In 2-3 sentences, summarise what kinds of roles matched and any common "
        "themes (seniority, skills, regions). Be concise and factual."
    )
    return gemini.generate(prompt)


def ask(question: str, *, top_k: int = 6) -> dict:
    """RAG: answer a question grounded strictly in the retrieved job listings."""
    if not store.ready:
        return {"answer": "The job index has not been built yet.", "sources": []}
    if not _has_key():
        return {"answer": "Set GEMINI_API_KEY to enable the AI assistant.", "sources": []}

    query_vec = gemini.embed_query(question)
    jobs = store.semantic_search(query_vec, top_k=top_k)

    context = "\n\n".join(
        f"[{i + 1}] {j['title']} at {j['company']}\n"
        f"Location: {j['location']} | Type: {j['job_type']} | Category: {j['category']}\n"
        f"{j['summary']}"
        for i, j in enumerate(jobs)
    )
    prompt = (
        "You are a job-search assistant. Answer the user's question using ONLY the "
        "job listings below. Cite the jobs you rely on with their number in square "
        "brackets, e.g. [2]. If the listings don't contain the answer, say so plainly.\n\n"
        f"=== JOB LISTINGS ===\n{context}\n\n"
        f"=== QUESTION ===\n{question}"
    )
    answer = gemini.generate(prompt)
    return {"answer": answer, "sources": jobs}
