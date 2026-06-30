"""
LogMind - query analytics: persists a rolling log of recent queries to
query_log.json (capped at 500 entries) and serves the data consumed by
GET /api/analytics. Anchored at config.ROOT (not this file's own directory)
so query_log.json always lives at backend/query_log.json regardless of which
package this module lives in.
"""

import json
import os
import time

import config

_LOG_FILE = os.path.join(config.ROOT, "query_log.json")
_query_log: list[dict] = []


def load_query_log() -> None:
    global _query_log
    if os.path.exists(_LOG_FILE):
        try:
            _query_log = json.loads(open(_LOG_FILE, encoding="utf-8").read())
        except Exception:
            _query_log = []


def save_query_log() -> None:
    try:
        with open(_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(_query_log[-500:], f)
    except Exception:
        pass


def log_query(question: str, elapsed_ms: int, source_count: int, route: str | None = None) -> None:
    _query_log.append({
        "query": question,
        "ts": int(time.time() * 1000),
        "ms": elapsed_ms,
        "sources": source_count,
        "feedback": None,
        "route": route,
    })
    save_query_log()


def get_log() -> list[dict]:
    return _query_log


load_query_log()
