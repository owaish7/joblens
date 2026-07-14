"""Search and grounded RAG capabilities built with LangChain and LangGraph."""
from __future__ import annotations

from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

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
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarise search results concisely and factually. Do not invent details.",
            ),
            (
                "human",
                "A user searched for: {query}\n\nTop matching remote jobs:\n{listing}\n\n"
                "In 2-3 sentences, summarise the roles and common themes "
                "(seniority, skills, regions).",
            ),
        ]
    )
    try:
        chain = prompt | gemini.chat_model(temperature=0.3) | StrOutputParser()
        return chain.invoke({"query": query, "listing": listing}).strip()
    except gemini.GeminiError:
        raise
    except Exception as exc:
        raise gemini.GeminiError(f"Generation failed: {exc}") from exc


class AskState(TypedDict, total=False):
    """State carried through the deliberately small grounded-answer graph."""

    question: str
    top_k: int
    jobs: list[dict]
    context: str
    answer: str
    sources: list[dict]


def _retrieve_documents(state: AskState) -> AskState:
    jobs = store.semantic_search(
        gemini.embed_query(state["question"]), top_k=state.get("top_k", 6)
    )
    context = "\n\n".join(
        f"[{index}] {job['title']} at {job['company']}\n"
        f"Location: {job['location']} | Type: {job['job_type']} | Category: {job['category']}\n"
        f"{job['summary']}"
        for index, job in enumerate(jobs, 1)
    )
    return {"jobs": jobs, "context": context}


_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a job-search assistant. Answer using ONLY the supplied job listings. "
            "Cite every job you rely on with its number in square brackets, for example [2]. "
            "If the listings do not contain the answer, say so plainly.",
        ),
        (
            "human",
            "=== JOB LISTINGS ===\n{context}\n\n=== QUESTION ===\n{question}",
        ),
    ]
)


def _generate_grounded_answer(state: AskState) -> AskState:
    try:
        chain = _RAG_PROMPT | gemini.chat_model(temperature=0.3) | StrOutputParser()
        answer = chain.invoke({"question": state["question"], "context": state["context"]})
        return {"answer": answer.strip()}
    except gemini.GeminiError:
        raise
    except Exception as exc:
        raise gemini.GeminiError(f"Generation failed: {exc}") from exc


def _attach_citations(state: AskState) -> AskState:
    """Expose source records in precisely the order used by [1], [2], ... citations."""
    return {"sources": state.get("jobs", [])}


def _build_ask_graph():
    workflow = StateGraph(AskState)
    workflow.add_node("retrieve_documents", _retrieve_documents)
    workflow.add_node("generate_grounded_answer", _generate_grounded_answer)
    workflow.add_node("attach_citations", _attach_citations)
    workflow.add_edge(START, "retrieve_documents")
    workflow.add_edge("retrieve_documents", "generate_grounded_answer")
    workflow.add_edge("generate_grounded_answer", "attach_citations")
    workflow.add_edge("attach_citations", END)
    return workflow.compile()


ask_graph = _build_ask_graph()


def ask(question: str, *, top_k: int = 6) -> dict:
    """Run the LangGraph retrieval -> answer -> citation workflow."""
    if not store.ready:
        return {"answer": "The job index has not been built yet.", "sources": []}
    if not _has_key():
        return {"answer": "Set GEMINI_API_KEY to enable the AI assistant.", "sources": []}

    result = ask_graph.invoke({"question": question, "top_k": top_k})
    return {"answer": result["answer"], "sources": result["sources"]}
