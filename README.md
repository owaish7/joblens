# 🔎 JobLens — Semantic Job Search + RAG Assistant

Search **real remote jobs** by meaning, not just keywords — and ask an AI
assistant questions that are answered *only* from the retrieved listings (with
citations). A compact, production-shaped demo of the modern GenAI stack:
**LangChain embeddings → FAISS retrieval → LangGraph-grounded generation**, wrapped in a FastAPI service.

> **Live demo:** https://joblens-a6sg.onrender.com  ·  **Source:** https://github.com/owaish7/joblens
>
> _(free tier sleeps after inactivity — first load may take ~50s to wake)_

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-8e75ff)

---

## What it does

| Feature | How |
| --- | --- |
| **Semantic search** | LangChain embeds each job with Gemini; FAISS ranks L2-normalized vectors by cosine similarity — so "roles building LLM pipelines" finds relevant jobs even without those exact words. |
| **AI overview** | A short Gemini-generated summary of what the top matches have in common. |
| **Ask AI (RAG)** | Retrieves the most relevant jobs, then asks Gemini to answer **grounded only in those listings**, with `[n]` citations — no hallucinated jobs. |
| **Clean JSON API** | `/api/search` and `/api/ask` — the UI is just a client of the same API. |

## Architecture

```
Remotive + RemoteOK ──► ingest.py ──► jobs.faiss + jobs.pkl + jobs.json
                                        │
   query ──► LangChain Gemini embeddings ──► FAISS cosine retrieval ──► top-k jobs
                                        │
                          ┌─────────────┴─────────────┐
                    LangChain overview         LangGraph grounded RAG
                                                  retrieve → generate → cite
                                        │
                                  FastAPI (main.py) ──► unchanged web UI
```

Design notes:
- **Retrieval uses LangChain's FAISS integration.** It uses inner-product search
  over L2-normalized vectors, preserving the cosine similarity score returned by
  the existing API.
- **Gemini is called through LangChain's Google integration** while retaining the
  configured generation and embedding model names, batching, retry/backoff, and
  rate-limit pauses during ingestion.
- **Ask AI is a minimal LangGraph:** retrieve documents → generate a grounded
  answer → attach ordered citation sources.
- **The index is built offline and committed**, so the deployed app starts
  instantly and only needs the API key at query time.

## Refactor story: before and now

JobLens was intentionally refactored in place rather than rebuilt. The API,
frontend, Gemini model configuration, Docker deployment, and user-facing search
results remain compatible; only the implementation behind them was modernized.

| Concern | Before | Now | Why it is better |
| --- | --- | --- | --- |
| **Gemini integration** | Hand-written REST requests for embeddings and generation | LangChain Google Gemini embeddings, chat model, and prompt templates | Centralizes provider integration and makes prompts/model calls composable without changing the configured models. |
| **Vector retrieval** | `embeddings.npy` plus NumPy brute-force cosine similarity | LangChain FAISS vector store with job metadata | Preserves cosine-like scores while providing a production vector-index abstraction that can grow beyond a small in-memory matrix. |
| **RAG orchestration** | Retrieval, prompting, and source attachment in one manual function | LangGraph stateful workflow: retrieve -> generate grounded answer -> attach citations | Makes the RAG path explicit, testable, and easier to extend with observability or validation steps later. |
| **Operational resilience** | Custom REST retry loops | Retained batching, retry/backoff, and rate-limit pauses around LangChain clients | Keeps ingestion safe for a metered API while using standard LLM tooling. |
| **Application contract** | FastAPI endpoints and static frontend | The same endpoints, request/response shapes, and frontend | Lets the deployment evolve internally without creating a breaking client migration. |

The key engineering decision was to use **FAISS inner-product search over
L2-normalized vectors**. This keeps the score equivalent to cosine similarity,
so the UI and API still receive the same kind of `score` value even though the
retrieval engine changed.

## Interview preparation

### 45-second project explanation

> JobLens is a FastAPI job-search application that ingests listings from
> Remotive and RemoteOK, embeds them with Gemini, and supports semantic search
> plus grounded RAG answers with citations. I refactored the production system
> without changing its APIs or frontend: manual Gemini REST calls became
> LangChain components, a NumPy cosine scan became a persisted FAISS index, and
> the Ask AI flow became a small LangGraph with retrieval, grounded generation,
> and citation attachment. The important constraint was backward compatibility,
> so I preserved the model configuration, metadata, scoring semantics, retries,
> rate limiting, Docker, and Render deployment.

### Talking points and strong answers

**Why FAISS if the old NumPy search worked?**  NumPy was adequate for a few
hundred or thousand records, but FAISS gives a standard persistent vector-store
boundary and a clearer path to larger indexes. I kept exact inner-product search
and normalized vectors, so this was a safe infrastructure improvement rather
than an unnecessary behavioral change.

**Why use LangGraph for such a small workflow?**  The workflow has real state:
the question, retrieved jobs, formatted context, answer, and sources. LangGraph
makes those transitions explicit without overengineering it into a multi-agent
system. It also leaves a clean extension point for future guardrails, reranking,
or tracing.

**How do you prevent hallucination?**  The generation prompt instructs Gemini to
use only retrieved listings and cite them as `[n]`. The graph returns the same
ordered job records as sources, so each citation maps directly to a rendered job
card in the frontend.

**How did you avoid a breaking migration?**  I preserved the FastAPI endpoint
contracts and metadata schema. The only data migration is an offline index
rebuild from `embeddings.npy` to `jobs.faiss` plus `jobs.pkl`; the service reports
that the index is not ready until those artifacts are deployed.

**What would you improve next?**  Add automated API and retrieval regression
tests, LangSmith/OpenTelemetry tracing, metadata filtering and reranking, and a
managed vector database only if index size or update frequency outgrows a
committed local FAISS artifact.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

copy .env.example .env           # then paste your free Gemini key
python -m src.ingest             # fetch jobs + build the index (one time)
uvicorn src.main:app --reload    # open http://127.0.0.1:8000
```

### Existing-index migration

`embeddings.npy` is no longer used. After upgrading, rebuild the index once with
`python -m src.ingest`; this creates `data/jobs.faiss` and `data/jobs.pkl` while
retaining the metadata export in `data/jobs.json`. Commit/deploy all three files.
The API reports `index_ready: false` until the new FAISS artifacts are present.

Get a free Gemini key at <https://aistudio.google.com/apikey>. If a model name
errors, check the models enabled for your key in Google AI Studio, then update
`GEN_MODEL` / `EMBED_MODEL` in `.env`.

## Deploy (Render, free)

This repo ships a `render.yaml` blueprint and a `Dockerfile`, so deploy is
near one-click:

1. Push this repo to GitHub (done).
2. On [Render](https://render.com): **New + → Blueprint** → connect this repo.
   Render reads `render.yaml` and provisions the service.
3. When prompted, paste your `GEMINI_API_KEY` (marked secret; not stored in git).
4. Render builds the `Dockerfile` and serves the app. The free tier sleeps after
   ~15 min idle and cold-starts in ~50s.

Build the FAISS index locally with the production `GEMINI_API_KEY`, commit
`jobs.faiss`, `jobs.pkl`, and `jobs.json`, then deploy. The committed `data/`
index means Render starts instantly and only needs the key for query-time AI.

## Tech stack

Python · FastAPI · LangChain · FAISS · LangGraph · Google Gemini · Docker

## Credits

Job data © [Remotive](https://remotive.com) and [RemoteOK](https://remoteok.com)
via their public APIs, used per their attribution terms. This is an independent
educational demo.
