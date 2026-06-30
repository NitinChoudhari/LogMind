"""
LogMind - cross-encoder reranking.

After hybrid retrieval produces a candidate pool, a cross-encoder scores each
(query, passage) pair directly - far more accurate than the bi-encoder/BM25
ranking, because it reads the query and passage together. We keep only the top
few, which is what actually goes to the synthesizer.

The active reranker is selected in `.env` via RERANKER_MODEL (a models.yaml
registry name); config.RERANKER is the resolved entry's `loader` value:
  * cross-encoder        - a plain sequence-classification cross-encoder via
                            sentence-transformers, e.g. bge-reranker-v2-m3
                            (no custom code, unlike cross-encoder-causal)
  * cross-encoder-causal - a causal-LM reranker via sentence-transformers
                            CrossEncoder, e.g. Qwen3-Reranker-4B (needs
                            trust_remote_code + chat-template scoring)
  * none                 - RERANKER_MODEL=none, skip; return hybrid order

Any failure (package missing, model can't load) degrades gracefully to the
hybrid order rather than breaking the request.
"""

from functools import lru_cache

import config


@lru_cache(maxsize=1)
def _cross_encoder_causal():
    from sentence_transformers import CrossEncoder

    # e.g. Qwen3-Reranker is a causal LM scored via the yes/no token logits
    # (LogitScore module). The chat_template.jinja in the model folder formats
    # each (query, passage) pair; sentence-transformers handles this automatically.
    import torch

    return CrossEncoder(
        config.RERANK_MODEL,
        trust_remote_code=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_kwargs={"torch_dtype": torch.float16},
    )


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    # e.g. bge-reranker-v2-m3 is a plain sequence-classification cross-encoder
    # (config.json: XLMRobertaForSequenceClassification) - a standard
    # transformers architecture, no custom modeling code or chat template involved.
    import torch

    return CrossEncoder(
        config.RERANK_MODEL,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_kwargs={"torch_dtype": torch.float16},
    )


def rerank(query, docs, top_n=None):
    """Return the top_n docs for `query`, reranked by a cross-encoder."""
    top_n = top_n or config.RERANK_TOP_N
    if config.RERANKER == "none" or len(docs) <= 1:
        return docs[:top_n]

    try:
        ce = _cross_encoder_causal() if config.RERANKER == "cross-encoder-causal" else _cross_encoder()
        scores = ce.predict([(query, d.page_content) for d in docs])
        order = sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)
        return [docs[i] for i in order[:top_n]]

    except Exception as exc:  # noqa: BLE001
        print(f"[rerank] '{config.RERANKER}' unavailable ({type(exc).__name__}: {exc}); "
              f"falling back to hybrid order.")
        return docs[:top_n]
