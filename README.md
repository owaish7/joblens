# 🔎 JobLens — Semantic Job Search + RAG Assistant

Search **real remote jobs** by meaning, not just keywords — and ask an AI
assistant questions that are answered *only* from the retrieved listings (with
citations). A compact, production-shaped demo of the modern GenAI stack:
**embeddings → vector retrieval → LLM generation**, wrapped in a FastAPI service.

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
| **Semantic search** | Each job is embedded with Gemini; queries are embedded and ranked by cosine similarity — so "roles building LLM pipelines" finds relevant jobs even without those exact words. |
| **AI overview** | A short Gemini-generated summary of what the top matches have in common. |
| **Ask AI (RAG)** | Retrieves the most relevant jobs, then asks Gemini to answer **grounded only in those listings**, with `[n]` citations — no hallucinated jobs. |
| **Clean JSON API** | `/api/search` and `/api/ask` — the UI is just a client of the same API. |

## Architecture

```
Remotive + RemoteOK ──► ingest.py ──► embeddings.npy + jobs.json   (built once, committed)
                                        │
   query ──► Gemini embed ──► cosine similarity (store.py) ──► top-k jobs
                                        │
                          ┌─────────────┴─────────────┐
                    AI overview                  RAG answer (rag.py)
                                        │
                                  FastAPI (main.py) ──► web UI
```

Design notes:
- **Retrieval is brute-force cosine similarity in NumPy.** For a few thousand
  jobs that is instant and dependency-free; the `JobStore` interface is a drop-in
  spot for FAISS or a managed vector index (e.g. Azure AI Search) at scale.
- **Gemini is called over plain REST** (`src/gemini.py`) — transparent and
  version-proof, and the whole app runs without heavyweight ML dependencies.
- **The index is built offline and committed**, so the deployed app starts
  instantly and only needs the API key at query time.

## Run locally

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

copy .env.example .env           # then paste your free Gemini key
python -m src.ingest             # fetch jobs + build the index (one time)
uvicorn src.main:app --reload    # open http://127.0.0.1:8000
```

Get a free Gemini key at <https://aistudio.google.com/apikey>. If a model name
errors, run `python -m src.gemini` to list the models your key can use and update
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

The committed `data/` index means the app starts instantly and only needs the key
at query time.

## Tech stack

Python · FastAPI · Google Gemini (embeddings + generation) · NumPy · Docker

## Credits

Job data © [Remotive](https://remotive.com) and [RemoteOK](https://remoteok.com)
via their public APIs, used per their attribution terms. This is an independent
educational demo.
