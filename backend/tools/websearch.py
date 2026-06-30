"""
LogMind - web search, backing the Manager's "web" route (agents/manager.py).

Returns the same langchain_core.documents.Document shape knowledge/retrieval.py
produces, so web results flow through the exact same SourceCollector / [n]
citation / Synthesizer machinery as indexed-document retrieval - no separate
citation path needed.
"""

from functools import lru_cache

from langchain_core.documents import Document

import config


@lru_cache(maxsize=1)
def _client():
    if not config.TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set - cannot perform a web search.")
    from tavily import TavilyClient

    return TavilyClient(api_key=config.TAVILY_API_KEY)


def websearch(query: str, max_results: int | None = None) -> list[Document]:
    """Raises RuntimeError if TAVILY_API_KEY is unset or the API call fails -
    callers (agents/crew.py) catch this to degrade gracefully rather than
    failing the request."""
    try:
        response = _client().search(query, max_results=max_results or config.WEB_SEARCH_MAX_RESULTS)
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Tavily search failed: {type(exc).__name__}: {exc}") from exc

    return [
        Document(
            page_content=r.get("content", ""),
            metadata={
                "source": r.get("url", ""),
                "title": r.get("title", ""),
                "similarity": r.get("score"),
                "kind": "web",
            },
        )
        for r in response.get("results", [])
    ]
