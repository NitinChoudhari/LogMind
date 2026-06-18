"""
LogMind - cross-encoder reranking.

After hybrid retrieval produces a candidate pool, a cross-encoder scores each
(query, passage) pair directly - far more accurate than the bi-encoder/BM25
ranking, because it reads the query and passage together. We keep only the top
few, which is what actually goes to the synthesizer.

Backends (config.RERANKER):
  * flashrank    - lightweight ONNX cross-encoder, local, no torch (default)
  * crossencoder - sentence-transformers CrossEncoder (heavier, uses your GPU)
  * none         - skip; return hybrid order

Any failure (package missing, model can't load) degrades gracefully to the
hybrid order rather than breaking the request.
"""

from functools import lru_cache

import config


@lru_cache(maxsize=1)
def _flashrank():
    from flashrank import Ranker, RerankRequest

    model = config.RERANK_MODEL or "ms-marco-MiniLM-L-12-v2"
    return Ranker(model_name=model), RerankRequest


@lru_cache(maxsize=1)
def _crossencoder():
    from sentence_transformers import CrossEncoder

    model = config.RERANK_MODEL or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    return CrossEncoder(model)


def rerank(query, docs, top_n=None):
    """Return the top_n docs for `query`, reranked by a cross-encoder."""
    top_n = top_n or config.RERANK_TOP_N
    if config.RERANKER == "none" or len(docs) <= 1:
        return docs[:top_n]

    try:
        if config.RERANKER == "crossencoder":
            ce = _crossencoder()
            scores = ce.predict([(query, d.page_content) for d in docs])
            order = sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)
            return [docs[i] for i in order[:top_n]]

        # default: flashrank
        ranker, RerankRequest = _flashrank()
        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
        results = ranker.rerank(RerankRequest(query=query, passages=passages))
        return [docs[r["id"]] for r in results[:top_n]]

    except Exception as exc:  # noqa: BLE001
        print(f"[rerank] '{config.RERANKER}' unavailable ({type(exc).__name__}: {exc}); "
              f"falling back to hybrid order.")
        return docs[:top_n]
