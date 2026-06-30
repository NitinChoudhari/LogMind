"""
LogMind - retrieval.

Hybrid retrieval via Qdrant's native dense+sparse search: a dense (embedding)
leg and a sparse (BM25, via fastembed's Qdrant/bm25 model) leg are queried in
one request and fused server-side with Reciprocal Rank Fusion. This replaced an
earlier hand-rolled weighted-RRF over a separately-pulled BM25 index (built from
the whole corpus pulled out of Chroma into memory at startup) - Qdrant does both
the lexical scoring and the fusion itself, so neither leg needs the full corpus
resident in process anymore.

The fused pool then goes through knowledge/rerank.py's cross-encoder. A
SourceCollector assigns a stable [n] index to every distinct chunk used in a
request, so the answer's [n] citations line up exactly with the sources shown
in the UI. Each selected source also carries a `similarity` score - real cosine
similarity between the query and that chunk's dense vector, fetched separately
from whatever the fusion/rerank ranking actually used, purely for display
(the "% match" badge in the UI) rather than as the ranking signal itself.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document
from qdrant_client import models

import config
from knowledge.rerank import rerank

# Recognizes "section 54", "sec. 80C", "Section80c" style mentions in a query so
# we can prefer Act chunks tagged with a matching `section` metadata value -
# the single most common precision win for a statute (exact lookups), without
# any agent/UI changes. Falls back silently to normal hybrid search if the
# filtered lookup comes up empty (e.g. the number doesn't exist in the Act).
_SECTION_QUERY_RE = re.compile(r"\bsec(?:tion)?\.?\s*(\d{1,3}[a-z]?)\b", re.IGNORECASE)


def _to_qdrant_filter(where: dict) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in where.items()])


def _point_to_doc(point) -> Document:
    payload = dict(point.payload or {})
    content = payload.pop("content", "")
    payload["_qdrant_id"] = point.id
    return Document(page_content=content, metadata=payload)


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class Retriever:
    """Connects to the Qdrant collection once and serves hybrid retrieval."""

    def __init__(self):
        self.candidate_k = config.CANDIDATE_K
        self.final_k = config.RERANK_TOP_N
        self.client = config.get_qdrant_client()
        self.embeddings = config.get_embeddings()
        self.sparse_model = config.get_sparse_embedder()

        if not self.client.collection_exists(config.COLLECTION):
            raise RuntimeError(
                f"Qdrant collection '{config.COLLECTION}' does not exist at {config.QDRANT_URL}. "
                f"Run `python ingest.py` first."
            )
        # Guard against an embedding-model mismatch between ingestion and now.
        probe = self.client.scroll(config.COLLECTION, limit=1, with_payload=["embedding_id"])[0]
        if probe:
            built_with = probe[0].payload.get("embedding_id")
            if built_with and built_with != config.embedding_id():
                raise RuntimeError(
                    f"Index was built with embeddings '{built_with}' but PROVIDER is "
                    f"now '{config.embedding_id()}'. Re-run `python ingest.py --reset` "
                    f"after switching providers (embeddings are not interchangeable)."
                )

    def _hybrid_pool(self, query: str, where: Optional[dict] = None) -> tuple[List[Document], List[float]]:
        """Dense + sparse vectors, fused server-side via Qdrant's native RRF.
        Returns (docs, dense_query_vector) - the dense vector is reused by
        search() to compute display-only cosine similarity on the final picks."""
        dense_vec = self.embeddings.embed_query(query)
        sparse_vec = next(self.sparse_model.embed([query]))
        qfilter = _to_qdrant_filter(where) if where else None

        result = self.client.query_points(
            config.COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", limit=self.candidate_k, filter=qfilter),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist()),
                    using="sparse", limit=self.candidate_k, filter=qfilter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=self.candidate_k,
            with_payload=True,
        )
        return [_point_to_doc(p) for p in result.points], dense_vec

    def dense_pool(self, query: str) -> List[Document]:
        """Dense-only retrieval (the embedder in isolation) - no sparse leg, no
        fusion, no rerank. Eval-only (component-level embedding quality); never
        used by the live request path."""
        dense_vec = self.embeddings.embed_query(query)
        result = self.client.query_points(
            config.COLLECTION,
            query=dense_vec,
            using="dense",
            limit=self.candidate_k,
            with_payload=True,
        )
        return [_point_to_doc(p) for p in result.points]

    def _attach_similarity(self, docs: List[Document], dense_vec: List[float]) -> None:
        """Mutates each doc's metadata in place with a real cosine-similarity
        score against the query - decoupled from whatever fusion/rerank order
        actually picked it, purely for the UI's "% match" badge."""
        ids = [d.metadata["_qdrant_id"] for d in docs]
        if not ids:
            return
        points = self.client.retrieve(config.COLLECTION, ids=ids, with_vectors=["dense"])
        vectors = {p.id: p.vector["dense"] for p in points}
        for d in docs:
            vec = vectors.get(d.metadata["_qdrant_id"])
            d.metadata["similarity"] = round(_cosine(dense_vec, vec), 4) if vec else None

    def search(self, query: str, where: Optional[dict] = None) -> List[Document]:
        # An explicit filter always wins (caller knows what they want).
        if where:
            pool, dense_vec = self._hybrid_pool(query, where)
            top = rerank(query, pool, top_n=self.final_k)
            self._attach_similarity(top, dense_vec)
            return top

        # Heuristic: an explicit "section N" mention should prefer Act chunks
        # tagged with that exact section, falling back to plain hybrid search
        # if the lookup comes up empty.
        m = _SECTION_QUERY_RE.search(query)
        if m:
            section = f"Section {m.group(1).upper()}"
            pool, dense_vec = self._hybrid_pool(query, {"doc_type": "act", "section": section})
            if pool:
                top = rerank(query, pool, top_n=self.final_k)
                self._attach_similarity(top, dense_vec)
                return top

        # 1) hybrid candidate pool (dense + sparse, fused server-side)
        pool, dense_vec = self._hybrid_pool(query)
        # 2) cross-encoder rerank down to the best few
        top = rerank(query, pool, top_n=self.final_k)
        self._attach_similarity(top, dense_vec)
        return top

    def doc_count(self) -> int:
        return self.client.count(config.COLLECTION).count

    def all_chunks(self) -> List[Document]:
        """Every indexed chunk, source metadata intact - used by eval_pipeline's
        dataset generator to sample passages for synthetic Q/A generation."""
        docs = []
        offset = None
        while True:
            points, offset = self.client.scroll(config.COLLECTION, limit=256, offset=offset, with_payload=True)
            docs.extend(_point_to_doc(p) for p in points)
            if offset is None:
                break
        return docs

    def sources(self) -> List[str]:
        seen = []
        for d in self.all_chunks():
            s = d.metadata.get("source")
            if s and s not in seen:
                seen.append(s)
        return seen

    def source_stats(self) -> list[dict]:
        """Per-source chunk count + doc metadata for the documents endpoint."""
        stats: dict[str, dict] = {}
        for d in self.all_chunks():
            src = d.metadata.get("source", "unknown")
            if src not in stats:
                stats[src] = {
                    "source": src,
                    "chunks": 0,
                    "doc_type": d.metadata.get("doc_type", ""),
                    "topic": d.metadata.get("topic", ""),
                    "exam_board": d.metadata.get("exam_board", ""),
                }
            stats[src]["chunks"] += 1
        return list(stats.values())

    def source_preview(self) -> dict[str, str]:
        """First 200-char excerpt per source, used as the knowledge-page preview."""
        previews: dict[str, str] = {}
        for d in self.all_chunks():
            src = d.metadata.get("source", "unknown")
            if src not in previews:
                previews[src] = d.page_content.strip()[:200]
        return previews


@dataclass
class SourceEntry:
    n: int
    source: str
    section: str
    snippet: str
    content: str
    similarity: Optional[float] = None
    kind: str = "kb"
    title: Optional[str] = None


@dataclass
class SourceCollector:
    """Assigns stable [n] indices to distinct chunks across a single request."""

    entries: List[SourceEntry] = field(default_factory=list)
    _seen: dict = field(default_factory=dict)

    def add(self, docs) -> List[SourceEntry]:
        added = []
        for d in docs:
            key = d.metadata.get("_qdrant_id") or d.page_content.strip()
            if key in self._seen:
                added.append(self._seen[key])
                continue
            entry = SourceEntry(
                n=len(self.entries) + 1,
                source=d.metadata.get("source", "unknown"),
                section=d.metadata.get("section", ""),
                snippet=d.page_content.strip()[:240],
                content=d.page_content.strip(),
                similarity=d.metadata.get("similarity"),
                kind=d.metadata.get("kind", "kb"),
                title=d.metadata.get("title"),
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
            {
                "n": e.n, "source": e.source, "section": e.section, "snippet": e.snippet,
                "similarity": e.similarity, "kind": e.kind, "title": e.title,
            }
            for e in self.entries
        ]
