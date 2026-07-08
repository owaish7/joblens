"""FastAPI application: JSON API + a small single-page UI.

Endpoints
    GET  /              -> the web UI
    GET  /api/health    -> status + index size
    GET  /api/categories-> distinct job categories (for the filter dropdown)
    POST /api/search    -> semantic search + optional AI overview
    POST /api/ask       -> RAG question answering over the jobs
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, rag
from .gemini import GeminiError

app = FastAPI(title="JobLens", description="Semantic job search + RAG assistant")

STATIC_DIR = config.ROOT / "static"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 8
    category: str | None = None
    overview: bool = True


class AskRequest(BaseModel):
    question: str


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "index_ready": rag.store.ready,
        "jobs_indexed": len(rag.store.jobs),
        "ai_enabled": bool(config.GEMINI_API_KEY),
        "gen_model": config.GEN_MODEL,
        "embed_model": config.EMBED_MODEL,
    }


@app.get("/api/categories")
def categories() -> dict:
    return {"categories": rag.store.categories()}


@app.post("/api/search")
def search(req: SearchRequest) -> dict:
    try:
        result = rag.search(req.query, top_k=req.top_k, category=req.category)
        if req.overview and result.get("mode") == "semantic":
            result["overview"] = rag.overview(req.query, result["results"])
        return result
    except GeminiError as e:
        return {"mode": "error", "results": [], "message": str(e)}


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    try:
        return rag.ask(req.question)
    except GeminiError as e:
        return {"answer": str(e), "sources": []}


# Static assets + SPA entry point. Mounted last so /api/* takes precedence.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
