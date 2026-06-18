"""
LogMind - FastAPI backend.

    uvicorn app:app --reload --port 8000

Endpoints:
    GET  /api/health     -> provider + whether the index is built
    GET  /api/documents  -> list of indexed source files
    POST /api/query      -> { question } -> { answer, sub_queries, sources, ... }
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import config

app = FastAPI(title="LogMind")

# Allow the Vite dev server to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryIn(BaseModel):
    question: str


def _index_ready() -> bool:
    return os.path.isdir(config.DB_DIR) and bool(os.listdir(config.DB_DIR))


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "provider": config.PROVIDER,
        "model": config.active_model_label(),
        "embeddings": config.embedding_id(),
        "reranker": config.RERANKER,
        "index_ready": _index_ready(),
    }


@app.get("/api/documents")
def documents():
    if not _index_ready():
        return {"documents": [], "index_ready": False}
    from rag_crew import get_retriever

    r = get_retriever()
    return {"documents": r.sources(), "chunks": r.doc_count(), "index_ready": True}


@app.post("/api/query")
async def query(body: QueryIn):
    question = (body.question or "").strip()
    if len(question) < 3:
        return {"error": "Ask a fuller question."}
    if not _index_ready():
        return {"error": "No index found. Run `python ingest.py` first."}

    from rag_crew import run_query

    try:
        # crewAI runs synchronously; keep the event loop free.
        return await run_in_threadpool(run_query, question)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/query/stream")
async def query_stream(body: QueryIn):
    question = (body.question or "").strip()
    if len(question) < 3:
        return {"error": "Ask a fuller question."}
    if not _index_ready():
        return {"error": "No index found. Run `python ingest.py` first."}

    from rag_crew import stream_query

    return StreamingResponse(
        stream_query(question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
