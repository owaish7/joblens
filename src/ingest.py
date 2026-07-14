"""One-time index builder.

Pulls remote jobs from two public APIs (Remotive + RemoteOK), normalises them
into one schema, embeds each with Gemini through LangChain, and saves
`data/jobs.json` + the LangChain FAISS artifact (`jobs.faiss` and `jobs.pkl`).
Commit those files so the deployed app starts
instantly (it only needs the API key at query time, not to rebuild the index).

    python -m src.ingest

Sources: Remotive (https://remotive.com) and RemoteOK (https://remoteok.com),
both via their public APIs. Please keep the attribution links in the UI, per
their terms.
"""
from __future__ import annotations

import html
import json
import re

import requests
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from . import config, gemini

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_UA = {"User-Agent": "Mozilla/5.0 (JobLens educational demo)"}

# Map free-text tags/titles to a coarse category so the filter works across sources.
_CATEGORY_KEYWORDS = [
    ("Software Development", ["dev", "engineer", "developer", "backend", "frontend",
                              "full stack", "python", "java", "react", "node", "golang"]),
    ("Data / AI", ["data", "machine learning", " ml", " ai", "analytics", "scientist", "llm"]),
    ("DevOps / Cloud", ["devops", "sre", "infrastructure", "cloud", "sysadmin", "kubernetes"]),
    ("Design", ["design", "ux", "ui"]),
    ("Marketing", ["marketing", "seo", "content", "social media", "growth"]),
    ("Sales / Business", ["sales", "account exec", "partnership", "business development"]),
    ("Product", ["product manager", "product"]),
    ("Customer Support", ["support", "customer success"]),
    ("Finance / Legal", ["finance", "accounting", "legal", "payroll"]),
]


def _clean(text: str) -> str:
    """Strip HTML tags/entities and collapse whitespace."""
    text = html.unescape(text or "")
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _categorize(*text_parts: str) -> str:
    blob = " ".join(text_parts).lower()
    for label, keywords in _CATEGORY_KEYWORDS:
        if any(k in blob for k in keywords):
            return label
    return "Other"


def fetch_remotive() -> list[dict]:
    r = requests.get(config.REMOTIVE_API, headers=_UA, timeout=60)
    r.raise_for_status()
    jobs = []
    for j in r.json().get("jobs", []):
        description = _clean(j.get("description", ""))
        jobs.append({
            "id": f"remotive-{j.get('id')}",
            "title": (j.get("title") or "").strip(),
            "company": (j.get("company_name") or "").strip(),
            "category": (j.get("category") or "Other").strip(),
            "job_type": (j.get("job_type") or "").strip(),
            "location": (j.get("candidate_required_location") or "").strip(),
            "salary": (j.get("salary") or "").strip(),
            "url": j.get("url", ""),
            "source": "Remotive",
            "summary": description[:400],
            "_text": description,
        })
    print(f"  Remotive: {len(jobs)} jobs")
    return jobs


def fetch_remoteok() -> list[dict]:
    r = requests.get("https://remoteok.com/api", headers=_UA, timeout=60)
    r.raise_for_status()
    data = r.json()
    jobs = []
    for j in data:
        if not isinstance(j, dict) or "position" not in j:  # skip the metadata element
            continue
        tags = j.get("tags", []) or []
        description = _clean(j.get("description", ""))
        smin, smax = j.get("salary_min") or 0, j.get("salary_max") or 0
        salary = f"${smin:,} - ${smax:,}" if smin and smax else ""
        job_type = next((t for t in tags if t in ("full time", "part time", "contract",
                                                   "freelance", "internship")), "")
        jobs.append({
            "id": f"remoteok-{j.get('id')}",
            "title": (j.get("position") or "").strip(),
            "company": (j.get("company") or "").strip(),
            "category": _categorize(j.get("position", ""), " ".join(tags)),
            "job_type": job_type.title(),
            "location": (j.get("location") or "").strip() or "Remote",
            "salary": salary,
            "url": j.get("url", ""),
            "source": "RemoteOK",
            "summary": (description or " ".join(tags))[:400],
            "_text": f"{description} Tags: {', '.join(tags)}",
        })
    print(f"  RemoteOK: {len(jobs)} jobs")
    return jobs


def fetch_jobs() -> list[dict]:
    print("Fetching jobs from public APIs ...")
    jobs: list[dict] = []
    for fetch in (fetch_remotive, fetch_remoteok):
        try:
            jobs.extend(fetch())
        except Exception as e:  # one source being down shouldn't kill the build
            print(f"  WARN: {fetch.__name__} failed: {e}")

    # De-duplicate by (title, company).
    seen, unique = set(), []
    for j in jobs:
        key = (j["title"].lower(), j["company"].lower())
        if j["title"] and key not in seen:
            seen.add(key)
            unique.append(j)

    unique = unique[: config.MAX_JOBS]
    print(f"  total after dedupe/cap: {len(unique)} jobs")
    return unique


def build_embedding_text(job: dict) -> str:
    """The text we actually turn into a vector -- richer than what we display."""
    parts = [
        job["title"],
        f"Company: {job['company']}",
        f"Category: {job['category']}",
        f"Location: {job['location']}",
        job["_text"][:1500],
    ]
    return "\n".join(p for p in parts if p)


def main() -> None:
    jobs = fetch_jobs()
    if not jobs:
        raise SystemExit("No jobs returned from any source -- aborting.")

    print(f"Embedding {len(jobs)} jobs with {config.EMBED_MODEL} ...")
    texts = [build_embedding_text(job) for job in jobs]
    vectors = gemini.embed_texts(texts)

    display_jobs = [{key: value for key, value in job.items() if key != "_text"} for job in jobs]
    vectorstore = FAISS.from_embeddings(
        zip(texts, vectors),
        gemini.embeddings(),
        metadatas=display_jobs,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        normalize_L2=True,
    )

    config.DATA_DIR.mkdir(exist_ok=True)
    config.JOBS_PATH.write_text(json.dumps(display_jobs, ensure_ascii=False, indent=2), encoding="utf-8")
    vectorstore.save_local(str(config.DATA_DIR), index_name=config.FAISS_INDEX_NAME)

    print(
        f"Saved {len(display_jobs)} jobs -> {config.JOBS_PATH.name}, "
        f"{config.FAISS_INDEX_PATH.name}, {config.FAISS_DOCSTORE_PATH.name}"
    )
    print("Done. Start the app with:  uvicorn src.main:app --reload")


if __name__ == "__main__":
    main()
