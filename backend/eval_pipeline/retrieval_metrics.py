"""
LogMind - component-level retrieval metrics for the eval pipeline.

The end-to-end eval (runner.py + judge.py) can't tell whether a bad result came
from the embedder, the sparse/fusion stage, or the reranker. This module scores
each stage independently, with no LLM involved (fast and free):

  * dense-only  - the embedder in isolation (Retriever.dense_pool)
  * hybrid pool - dense + sparse, fused server-side via RRF (Retriever._hybrid_pool)
  * reranked    - the hybrid pool reordered by the cross-encoder (knowledge.rerank.rerank)

Relevance is judged at the *passage* level via token-overlap against a gold
passage (re-ingest proof - matches on text, not on fragile Qdrant point ids),
falling back to file-level (`expected_sources`) for items that have no
`gold_passage`. The metric helpers are pure functions over a ranked list of
booleans, so they're trivially checkable in isolation.
"""

import math
import re

from knowledge.rerank import rerank

# A retrieved chunk counts as the gold hit if at least this fraction of the gold
# passage's (distinct) tokens appear in the chunk. Tolerates chunk-boundary
# shifts after a re-ingest, unlike exact-substring or stored-id matching.
GOLD_HIT_THRESHOLD = 0.6

_TOKEN_RE = re.compile(r"\w+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def passage_match(gold_passages: list[str], chunk_text: str, threshold: float = GOLD_HIT_THRESHOLD) -> bool:
    """True if any gold passage's token-containment ratio against `chunk_text`
    (fraction of gold tokens present in the chunk) meets `threshold`."""
    chunk_toks = _tokens(chunk_text)
    if not chunk_toks:
        return False
    for gold in gold_passages:
        gold_toks = _tokens(gold)
        if not gold_toks:
            continue
        if len(gold_toks & chunk_toks) / len(gold_toks) >= threshold:
            return True
    return False


# --------------------------------------------------------------------------- #
# Pure ranking metrics: each takes `rels`, a ranked list of booleans
# (rels[i] is True if the i-th retrieved item is relevant).
# --------------------------------------------------------------------------- #
def mrr(rels: list[bool]) -> float:
    for i, r in enumerate(rels):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(rels: list[bool], k: int) -> float:
    """Binary: 1.0 if at least one relevant item is in the first k, else 0.0."""
    return 1.0 if any(rels[:k]) else 0.0


def ndcg_at_k(rels: list[bool], k: int) -> float:
    """NDCG@k with binary gain. IDCG places all relevant items first."""
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rels[:k]) if r)
    n_rel = sum(1 for r in rels if r)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_rel, k)))
    return dcg / idcg if idcg else 0.0


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _relevance_fn(item: dict):
    """A `doc -> bool` predicate. Passage-level when the item carries a
    `gold_passage` (str or list); otherwise file-level via `expected_sources`."""
    gold = item.get("gold_passage")
    if gold:
        golds = [gold] if isinstance(gold, str) else list(gold)
        return lambda doc: passage_match(golds, doc.page_content)
    sources = set(item.get("expected_sources", []))
    return lambda doc: doc.metadata.get("source") in sources


def _stage_metrics(rels: list[bool], ks: list[int]) -> dict:
    metrics = {"mrr": mrr(rels), "ndcg@10": ndcg_at_k(rels, 10)}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(rels, k)
    return metrics


def evaluate_retrieval(item: dict, retriever, final_k: int) -> dict:
    """Run the query through all three retrieval stages and score each against
    the item's gold passage(s). Returns per-stage metric dicts plus the
    reranker's delta over the pre-rerank hybrid pool."""
    q = item["question"]
    is_relevant = _relevance_fn(item)
    ks = sorted({1, 5, final_k})

    dense = retriever.dense_pool(q)
    hybrid, _ = retriever._hybrid_pool(q)
    # Full reorder (top_n=len(pool)) so the pre/post comparison is over the same
    # set of chunks - we want the reranker's effect on *ordering*, not truncation.
    reranked = rerank(q, hybrid, top_n=len(hybrid)) if hybrid else []

    dense_m = _stage_metrics([is_relevant(d) for d in dense], ks)
    hybrid_m = _stage_metrics([is_relevant(d) for d in hybrid], ks)
    reranked_m = _stage_metrics([is_relevant(d) for d in reranked], ks)
    rerank_delta = {key: round(reranked_m[key] - hybrid_m[key], 4) for key in hybrid_m}

    return {
        "dense": dense_m,
        "hybrid": hybrid_m,
        "reranked": reranked_m,
        "rerank_delta": rerank_delta,
    }
