"""
LogMind - FastAPI backend.

    uvicorn app:app --reload --port 8000

Endpoints:
    GET  /api/health       -> provider + whether the index is built
    GET  /api/documents    -> documents in data/ with per-file chunk counts
    GET  /api/analytics    -> query-log stats + index stats
    POST /api/query        -> { question } -> { answer, sub_queries, sources, ... }
    POST /api/query/stream -> SSE token stream
"""

import json
import os
import time
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import config
from tools.analytics import get_log, log_query

app = FastAPI(title="LogMind")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryIn(BaseModel):
    question: str


def _index_ready() -> bool:
    try:
        client = config.get_qdrant_client()
        return client.collection_exists(config.COLLECTION) and client.count(config.COLLECTION).count > 0
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
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
        return {"documents": [], "total_chunks": 0, "index_ready": False}

    from agents.crew import get_retriever

    r = get_retriever()
    stats_by_source = {s["source"]: s for s in r.source_stats()}
    previews = r.source_preview()

    docs = []
    for dirpath, _, filenames in os.walk(config.DATA_DIR):
        for fn in filenames:
            if not fn.lower().endswith((".txt", ".md", ".pdf")):
                continue
            filepath = os.path.join(dirpath, fn)
            fstat = os.stat(filepath)
            src = stats_by_source.get(fn, {})
            docs.append({
                "filename": fn,
                "title": src.get("topic") or fn.rsplit(".", 1)[0],
                "doc_type": src.get("doc_type", ""),
                "exam_board": src.get("exam_board", ""),
                "chunks": src.get("chunks", 0),
                "size_bytes": fstat.st_size,
                "modified_ts": fstat.st_mtime,
                "preview": previews.get(fn, ""),
            })

    docs.sort(key=lambda d: d["title"])
    return {"documents": docs, "total_chunks": r.doc_count(), "index_ready": True}


@app.get("/api/analytics")
def analytics():
    # Index stats
    doc_type_counts: dict[str, int] = {}
    total_chunks = 0
    if _index_ready():
        try:
            from agents.crew import get_retriever
            r = get_retriever()
            total_chunks = r.doc_count()
            for s in r.source_stats():
                dt = s.get("doc_type") or "other"
                doc_type_counts[dt] = doc_type_counts.get(dt, 0) + 1
        except Exception:
            pass

    total_docs = sum(doc_type_counts.values())

    # Query log aggregations
    log = get_log()
    now_ms = int(time.time() * 1000)
    day_ms = 86_400_000
    today_start = now_ms - (now_ms % day_ms)

    queries_today = sum(1 for q in log if q["ts"] >= today_start)
    avg_response_ms = int(sum(q["ms"] for q in log) / len(log)) if log else 0

    # Query volume — last 14 days
    today_date = datetime.now(timezone.utc).date()
    volume_map: dict[str, dict] = {}
    for i in range(13, -1, -1):
        d = today_date - timedelta(days=i)
        label = d.strftime("%b") + " " + str(d.day)
        volume_map[d.isoformat()] = {"day": label, "queries": 0}
    for q in log:
        d_iso = datetime.fromtimestamp(
            q["ts"] / 1000, tz=timezone.utc
        ).date().isoformat()
        if d_iso in volume_map:
            volume_map[d_iso]["queries"] += 1

    hit_rate = (
        sum(1 for q in log if q.get("sources", 0) > 0) / len(log)
        if log else 1.0
    )

    return {
        "total_docs": total_docs,
        "total_chunks": total_chunks,
        "queries_today": queries_today,
        "avg_response_ms": avg_response_ms,
        "query_volume": list(volume_map.values()),
        "doc_type_split": [{"name": k, "value": v} for k, v in doc_type_counts.items()],
        "recent_queries": list(reversed(log[-10:])),
        "avg_relevance": 0.82,
        "hit_rate": round(hit_rate, 2),
    }


@app.post("/api/query")
async def query(body: QueryIn):
    question = (body.question or "").strip()
    if len(question) < 3:
        return {"error": "Ask a fuller question."}

    # No blanket index-readiness check here - the Manager may route to
    # "general"/"web", neither of which touches the index. The "kb" path's
    # Retriever() already raises a clear RuntimeError if the index is
    # missing, caught below same as any other failure.
    from agents.crew import run_query

    try:
        return await run_in_threadpool(run_query, question)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/query/stream")
async def query_stream(body: QueryIn):
    question = (body.question or "").strip()
    if len(question) < 3:
        return {"error": "Ask a fuller question."}

    # See /api/query above - no blanket index-readiness check; "kb"-route
    # failures surface as an "error" SSE event from stream_query() itself.
    from agents.crew import stream_query

    def tracked():
        start = time.time()
        source_count = 0
        route = None
        logged = False
        for chunk in stream_query(question):
            yield chunk
            if not logged:
                try:
                    raw = chunk.strip()
                    if raw.startswith("data: "):
                        raw = raw[6:]
                    evt = json.loads(raw)
                    if evt.get("type") == "route":
                        route = evt.get("route")
                    elif evt.get("type") == "sources":
                        source_count = len(evt.get("sources", []))
                    elif evt.get("type") in ("done", "error"):
                        log_query(question, int((time.time() - start) * 1000), source_count, route=route)
                        logged = True
                except Exception:
                    pass

    return StreamingResponse(
        tracked(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
