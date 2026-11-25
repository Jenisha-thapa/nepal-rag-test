"""
Nepal Sewa AI — FastAPI app (web UI + chat API).

Run:
    uv run fastapi dev      # development: auto-reload, http://127.0.0.1:8000
    uv run fastapi run      # production-style: listens on 0.0.0.0:8000
(entrypoint configured in pyproject.toml → [tool.fastapi] entrypoint = "src.main:app")

On startup (`lifespan`) the app reads web/index.html (served at GET /) and starts loading the RAG
engine in a background thread: chunks (built if data/processed/all_chunks.json is missing),
embedding model, BM25, the stored ChromaDB index (embedded only if none exists yet), LLM chain.
Until it is ready /api/health reports {"status": "loading", "progress": {stage, done, total}}
and /api/chat answers 503; the page shows a progress bar.

Routes:
  GET    /                         web UI
  GET    /api/health               engine status, LLM chain, chunk count
  GET    /api/domains              service domains + example questions (config/domains.yaml)
  GET    /api/sources              indexed source pages + when they were fetched
  POST   /api/chat                 {"message", "user_id", "conversation_id"?, "domain"?}
                                   -> answer, sources, passages, query analysis, conversation_id
  GET    /api/conversations?user_id=...           the user's conversations (newest first)
  GET    /api/conversations/{id}?user_id=...      one conversation with its messages
  PATCH  /api/conversations/{id}?user_id=...      {"title": "..."} rename
  DELETE /api/conversations/{id}?user_id=...      delete one conversation
  DELETE /api/conversations?user_id=...           delete all of the user's conversations
  PUT    /api/messages/{id}/feedback?user_id=...  {"helpful": true|false|null} rate an answer
  GET    /docs                     interactive API docs

Conversations are stored in SQLite (src/storage.py). There is no login: the page makes a random
user id and keeps it in the browser; follow-up questions use the conversation's last turns.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.storage import ConversationStore
from src.utils import REPO_ROOT, get_settings, load_json, load_yaml, setup_logging

logger = logging.getLogger(__name__)

INDEX_HTML = REPO_ROOT / "web" / "index.html"
ID_PATTERN = r"^[A-Za-z0-9_-]{3,64}$"

# Populated once at startup (see `lifespan`).
state: dict[str, Any] = {"session": None, "html": "", "domains": [], "sources": [], "startup_seconds": 0.0,
                         "store": None, "engine_error": ""}
_engine_lock = threading.Lock()  # one query at a time through the retriever + LLM


def _store() -> ConversationStore:
    if state["store"] is None:
        state["store"] = ConversationStore(get_settings()["paths"].get(
            "conversations_db", str(REPO_ROOT / "data" / "conversations.db")))
    return state["store"]


# ── Schemas ────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000,
                         examples=["७० वर्ष पुगेका जेष्ठ नागरिकले मासिक कति भत्ता पाउँछन्?"])
    user_id: str = Field("guest", pattern=ID_PATTERN, description="Browser-generated user id")
    conversation_id: str | None = Field(None, pattern=ID_PATTERN,
                                        description="Continue this conversation; omit to start a new one")
    domain: str | None = Field(None, description="Optional domain id to search in; default: auto-classified",
                               examples=["health_social_welfare"])


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=80)


class Source(BaseModel):
    name: str
    url: str
    fetched_at: str = ""


class Passage(BaseModel):
    index: int
    source_name: str
    source_url: str
    domain_id: str
    section: str = ""
    content_type: str = ""   # faq | procedure | documents | fees | contact | notice | ... (or auto: table/list/text)
    date_bs: str = ""
    score: float
    text: str


class Analysis(BaseModel):
    """Step 1 of the pipeline: how the question was understood."""
    domain_id: str = ""
    standalone: str = ""
    search_queries: list[str] = []
    method: str = ""          # llm | keywords


class ChatResponse(BaseModel):
    answer: str
    domain_id: str
    domain_name: str
    llm_used: str
    refused: bool
    latency_ms: int
    sources: list[Source]
    passages: list[Passage]
    analysis: Analysis
    conversation_id: str
    conversation_title: str
    message_id: int


# ── Startup ────────────────────────────────────────────────────────────────────
def _load_domains() -> list[dict[str, Any]]:
    cfg = load_yaml(REPO_ROOT / "config" / "domains.yaml")
    return [
        {
            "id": d["id"],
            "name_en": d.get("name_en", d["id"]),
            "name_np": d.get("name_np", ""),
            "authority": d.get("authority", ""),
            "examples": d.get("examples", []),
        }
        for d in cfg.get("domains", [])
    ]


def _load_sources() -> list[dict[str, Any]]:
    """One entry per source URL (newest fetch wins), from data/raw/**/*.meta.json."""
    raw_dir = Path(get_settings()["paths"]["raw_dir"])
    by_url: dict[str, dict[str, Any]] = {}
    for p in sorted(raw_dir.glob("*/*.meta.json")):
        m = load_json(p)
        if m.get("status") == "ok":
            by_url[m["source_url"]] = {
                "name": m.get("source_name", ""),
                "url": m.get("source_url", ""),
                "domain_id": m.get("domain_id", ""),
                "fetched_at": m.get("fetched_at", ""),
            }
    return list(by_url.values())


def _ensure_chunks() -> None:
    chunks_path = Path(get_settings()["paths"]["processed_dir"]) / "all_chunks.json"
    if not (chunks_path.exists() and any(c.get("text") for c in load_json(chunks_path) or [])):
        logger.info("No chunks found — processing raw documents first...")
        from src.processing.pipeline import build_chunks
        build_chunks()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    state["html"] = INDEX_HTML.read_text(encoding="utf-8")
    state["domains"] = _load_domains()
    state["sources"] = _load_sources()
    # The engine loads in a background thread, so the page opens at once and shows a progress
    # bar (GET /api/health -> progress) instead of the browser waiting on a closed port.
    threading.Thread(target=_load_engine, name="load-engine", daemon=True).start()
    yield
    state["session"] = None


def _load_engine() -> None:
    from src.retrieval.indexer import set_progress
    t0 = time.perf_counter()
    try:
        set_progress("chunks")
        _ensure_chunks()
        from src.generation.chat import ChatSession
        settings = get_settings()
        logger.info("Loading RAG engine (embedding model, BM25, ChromaDB, LLM)...")
        session = ChatSession(provider=settings["generation"]["provider"], model=settings["generation"]["model"])
        state["startup_seconds"] = round(time.perf_counter() - t0, 1)
        state["session"] = session
        set_progress("ready")
        logger.info("Ready in %.1fs — %d chunks indexed", state["startup_seconds"], session.num_chunks)
    except Exception as exc:  # shown on the page instead of a silent hang
        logger.exception("The RAG engine failed to start")
        state["engine_error"] = f"{exc.__class__.__name__}: {exc}"


app = FastAPI(
    title="Nepal Sewa AI",
    description="Citation-grounded RAG over official Nepal government service pages.",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> str:
    return state["html"]


@app.get("/api/health")
async def health() -> dict[str, Any]:
    session = state["session"]
    settings = get_settings()
    from src.retrieval.indexer import PROGRESS
    return {
        "status": "ready" if session else ("error" if state.get("engine_error") else "loading"),
        "progress": {**PROGRESS, "error": state.get("engine_error", "")},
        "chunks": session.num_chunks if session else 0,
        "llm_chain": ([c.label for c in session.generator.llms] if session else []) + ["extractive-fallback"],
        "embedding_model": settings["retrieval"]["embedding_model"],
        "retrieval_mode": settings["retrieval"]["mode"],
        "startup_seconds": state["startup_seconds"],
    }


@app.get("/api/domains")
async def domains() -> list[dict[str, Any]]:
    return state["domains"]


@app.get("/api/sources")
async def sources() -> list[dict[str, Any]]:
    return state["sources"]


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # Plain `def`: FastAPI runs it in a worker thread, so CPU-bound retrieval and the
    # blocking LLM call don't freeze the event loop.
    session = state["session"]
    if session is None:
        raise HTTPException(status_code=503, detail="RAG engine is still loading")
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Empty message")
    if req.domain and req.domain not in {d["id"] for d in state["domains"]}:
        raise HTTPException(status_code=422, detail=f"Unknown domain '{req.domain}'")

    store = _store()
    if req.conversation_id:
        conv = store.get(req.user_id, req.conversation_id, with_messages=False)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        history = store.history(conv["id"])
    else:
        conv, history = store.create(req.user_id, title=message), []

    t0 = time.perf_counter()
    with _engine_lock:
        resp = session.ask(message, domain=req.domain or None, history=history)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    fetched = {s["url"]: s["fetched_at"] for s in state["sources"]}
    domain_names = {d["id"]: d["name_en"] for d in state["domains"]}

    unique_sources: dict[str, Source] = {}
    for c in resp.citations:
        if c["url"] and c["url"] not in unique_sources:
            unique_sources[c["url"]] = Source(name=c["name"], url=c["url"], fetched_at=fetched.get(c["url"], ""))

    result = {
        "answer": resp.answer,
        "domain_id": resp.domain_id,
        "domain_name": domain_names.get(resp.domain_id, ""),
        "llm_used": resp.llm_used,
        "refused": resp.refused,
        "latency_ms": latency_ms,
        "sources": [s.model_dump() for s in unique_sources.values()],
        "passages": [
            Passage(index=c["index"], source_name=c["name"], source_url=c["url"],
                    domain_id=c.get("domain_id", ""), section=c.get("section", ""),
                    content_type=c.get("content_type", ""), date_bs=c.get("date_bs", ""),
                    score=c.get("score", 0.0), text=c.get("text", "")).model_dump()
            for c in resp.citations
        ],
        "analysis": resp.analysis,
    }
    store.add_message(conv["id"], "user", message, {"domain": req.domain or ""})
    message_id = store.add_message(conv["id"], "assistant", resp.answer,
                                   {k: v for k, v in result.items() if k != "answer"})
    return ChatResponse(**result, conversation_id=conv["id"], conversation_title=conv["title"],
                        message_id=message_id)


# ── Conversations ──────────────────────────────────────────────────────────────
UserId = Query(..., pattern=ID_PATTERN, description="Browser-generated user id")


@app.get("/api/conversations")
def list_conversations(user_id: str = UserId) -> list[dict[str, Any]]:
    return _store().list(user_id)


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str, user_id: str = UserId) -> dict[str, Any]:
    conv = _store().get(user_id, conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, req: RenameRequest, user_id: str = UserId) -> dict[str, Any]:
    if not _store().rename(user_id, conversation_id, req.title):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"id": conversation_id, "title": req.title.strip()[:80]}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, user_id: str = UserId) -> dict[str, Any]:
    if not _store().delete(user_id, conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": 1}


class FeedbackRequest(BaseModel):
    helpful: bool | None = Field(..., description="true = helpful, false = not helpful, null = clear")


@app.put("/api/messages/{message_id}/feedback")
def message_feedback(message_id: int, req: FeedbackRequest, user_id: str = UserId) -> dict[str, Any]:
    if not _store().set_feedback(user_id, message_id, req.helpful):
        raise HTTPException(status_code=404, detail="Answer not found")
    return {"id": message_id, "helpful": req.helpful}


@app.delete("/api/conversations")
def delete_all_conversations(user_id: str = UserId) -> dict[str, Any]:
    return {"deleted": _store().delete_all(user_id)}
