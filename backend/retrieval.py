"""
LogMind - retrieval.

Hybrid retrieval done right:
  * Vector search with MMR (diversity, fewer near-duplicates).
  * BM25 lexical search built over the SAME corpus the vectors came from
    (the original built BM25 from only the current request's chunks while the
    vector half searched the whole store - so the two halves disagreed).
  * The two result lists are merged with weighted Reciprocal Rank Fusion. We do
    the fusion ourselves instead of importing EnsembleRetriever, whose import
    path keeps moving between LangChain versions (langchain.retrievers ->
    langchain_classic.retrievers). One less thing to break on an upgrade.

A SourceCollector assigns a stable [n] index to every distinct chunk used in a
request, so the answer's [n] citations line up exactly with the sources shown in
the UI.
"""

import hashlib
from dataclasses import dataclass, field
from typing import List

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

import config
from rerank import rerank


def _load_vectorstore() -> Chroma:
    store = Chroma(
        collection_name=config.COLLECTION,
        embedding_function=config.get_embeddings(),
        persist_directory=config.DB_DIR,
    )
    # Guard against an embedding-model mismatch between ingestion and now.
    meta = store.get(include=["metadatas"], limit=1).get("metadatas") or []
    if meta:
        built_with = meta[0].get("embedding_id")
        if built_with and built_with != config.embedding_id():
            raise RuntimeError(
                f"Index was built with embeddings '{built_with}' but PROVIDER is "
                f"now '{config.embedding_id()}'. Re-run `python ingest.py --reset` "
                f"after switching providers (embeddings are not interchangeable)."
            )
    return store


def _weighted_rrf(ranked_lists, weights, k_rrf: int = 60, top_n: int = 6) -> List[Document]:
    """Merge several ranked document lists via weighted Reciprocal Rank Fusion."""
    scores: dict = {}
    holder: dict = {}
    for docs, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(docs):
            key = hashlib.sha1(doc.page_content.strip().encode()).hexdigest()
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (k_rrf + rank + 1))
            holder.setdefault(key, doc)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [holder[k] for k in ordered[:top_n]]


class Retriever:
    """Loads the persisted store once and serves hybrid retrieval."""

    def __init__(self):
        self.candidate_k = config.CANDIDATE_K
        self.final_k = config.RERANK_TOP_N
        self.store = _load_vectorstore()

        # Pull the full corpus back out of Chroma to build BM25 over the SAME set.
        raw = self.store.get(include=["documents", "metadatas"])
        self._docs = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(raw.get("documents", []), raw.get("metadatas", []))
        ]

        self._vector = self.store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.candidate_k,
                "fetch_k": self.candidate_k * 2,
                "lambda_mult": 0.5,
            },
        )
        self._bm25 = None
        if self._docs:
            self._bm25 = BM25Retriever.from_documents(self._docs)
            self._bm25.k = self.candidate_k

    def search(self, query: str) -> List[Document]:
        # 1) hybrid candidate pool (vector MMR + BM25, fused)
        ranked = [self._vector.invoke(query)]
        weights = [0.6]
        if self._bm25 is not None:
            ranked.append(self._bm25.invoke(query))
            weights.append(0.4)
        pool = _weighted_rrf(ranked, weights, top_n=self.candidate_k)
        # 2) cross-encoder rerank down to the best few
        return rerank(query, pool, top_n=self.final_k)

    def doc_count(self) -> int:
        return len(self._docs)

    def sources(self) -> List[str]:
        seen = []
        for d in self._docs:
            s = d.metadata.get("source")
            if s and s not in seen:
                seen.append(s)
        return seen


@dataclass
class SourceEntry:
    n: int
    source: str
    section: str
    snippet: str
    content: str


@dataclass
class SourceCollector:
    """Assigns stable [n] indices to distinct chunks across a single request."""

    entries: List[SourceEntry] = field(default_factory=list)
    _seen: dict = field(default_factory=dict)

    def add(self, docs) -> List[SourceEntry]:
        added = []
        for d in docs:
            key = hashlib.sha1(d.page_content.strip().encode()).hexdigest()
            if key in self._seen:
                added.append(self._seen[key])
                continue
            entry = SourceEntry(
                n=len(self.entries) + 1,
                source=d.metadata.get("source", "unknown"),
                section=d.metadata.get("section", ""),
                snippet=d.page_content.strip()[:240],
                content=d.page_content.strip(),
            )
            self.entries.append(entry)
            self._seen[key] = entry
            added.append(entry)
        return added

    def numbered_context(self) -> str:
        blocks = []
        for e in self.entries:
            label = f"[{e.n}] (source: {e.source}" + (f" - {e.section}" if e.section else "") + ")"
            blocks.append(f"{label}\n{e.content}")
        return "\n\n".join(blocks)

    def as_payload(self) -> List[dict]:
        return [
            {"n": e.n, "source": e.source, "section": e.section, "snippet": e.snippet}
            for e in self.entries
        ]
