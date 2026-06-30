"""
LogMind - ingestion (run once, or whenever documents change).

    python ingest.py            # ingest everything under data/ (recursively)
    python ingest.py --reset    # wipe the index first (do this when you change
                                # chunking strategy OR switch embedding provider)

Separating ingestion from the request path fixes the original system, which
re-read and re-chunked a single hard-coded file on every query and never indexed
the PDFs at all.

Two metadata conventions are supported:
  * Filenames under data/ may encode metadata as `name__type__env__section__date.ext`
    (the original convention) - parsed defensively, falls back to sane defaults.
  * Known files get an explicit entry in FILE_METADATA (see below) instead - this is
    how the Tax Knowledge Base PDFs are tagged, since their filenames are human chapter
    titles, not the `__`-delimited scheme.
"""

import argparse
import os
import re
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from qdrant_client import models

import config

# --------------------------------------------------------------------------- #
# Known-file metadata (Tax Knowledge Base)
# --------------------------------------------------------------------------- #
# doc_type: "act" (bare statute) | "study_chapter" (ICAI single-topic chapter) |
#           "study_book" (ICSI multi-topic book)
FILE_METADATA = {
    "Basic Concepts.pdf": {
        "doc_type": "study_chapter", "topic": "Basic Concepts", "exam_board": "ICAI",
    },
    "Residence and Scope of Total Income.pdf": {
        "doc_type": "study_chapter", "topic": "Residence and Scope of Total Income", "exam_board": "ICAI",
    },
    "Salaries.pdf": {
        "doc_type": "study_chapter", "topic": "Salaries", "exam_board": "ICAI",
    },
    "Income from House Property.pdf": {
        "doc_type": "study_chapter", "topic": "Income from House Property", "exam_board": "ICAI",
    },
    "Profits and Gains of Business or Profession.pdf": {
        "doc_type": "study_chapter", "topic": "Profits and Gains of Business or Profession", "exam_board": "ICAI",
    },
    "Capital Gains.pdf": {
        "doc_type": "study_chapter", "topic": "Capital Gains", "exam_board": "ICAI",
    },
    "Income from Other Sources.pdf": {
        "doc_type": "study_chapter", "topic": "Income from Other Sources", "exam_board": "ICAI",
    },
    "Income of Other Persons included in Assessee’s Total Income.pdf": {
        "doc_type": "study_chapter", "topic": "Income of Other Persons included in Assessee's Total Income",
        "exam_board": "ICAI",
    },
    "Aggregation of Income, Set-Off and Carry Forward of Losses.pdf": {
        "doc_type": "study_chapter", "topic": "Aggregation of Income, Set-Off and Carry Forward of Losses",
        "exam_board": "ICAI",
    },
    "Deductions from Gross Total Income.pdf": {
        "doc_type": "study_chapter", "topic": "Deductions from Gross Total Income", "exam_board": "ICAI",
    },
    "Advance Tax, Tax Deduction at Source and Tax Collection at Source.pdf": {
        "doc_type": "study_chapter",
        "topic": "Advance Tax, Tax Deduction at Source and Tax Collection at Source",
        "exam_board": "ICAI",
    },
    "Provisions for filing Return of Income and Self Assessment.pdf": {
        "doc_type": "study_chapter", "topic": "Provisions for filing Return of Income and Self Assessment",
        "exam_board": "ICAI",
    },
    "Income Tax Liability – Computation and Optimisation.pdf": {
        "doc_type": "study_chapter", "topic": "Income Tax Liability - Computation and Optimisation",
        "exam_board": "ICAI",
    },
    "INCOME-TAX ACT, 2025.pdf": {
        "doc_type": "act", "topic": "Income-tax Act, 2025", "exam_board": "statute",
    },
    "Final_Tax_Law_Book.pdf": {
        "doc_type": "study_book", "topic": "Tax Laws (ICSI Executive Programme)", "exam_board": "ICSI",
    },
}

# Stateful chapter/section tagger, applied only to doc_type == "act" files: tracks
# the most recently seen "CHAPTER <roman>" and "<N>. " section marker across pages
# so every chunk can be stamped with the section actually in force there - the
# single highest-value metadata field for a statute (users ask "what does section
# X say", not "what's on page 214").
_CHAPTER_RE = re.compile(r"^\s*CHAPTER\s+([IVXLCDM]+)\b", re.MULTILINE)
_SECTION_RE = re.compile(r"^\s*(\d{1,3}[A-Z]?)\.\s", re.MULTILINE)


def _tag_act_chunks(chunks: list[Document]) -> None:
    """Mutates chunks in place, stamping metadata['chapter']/['section'].

    Runs AFTER splitting, over the chunk sequence in order (not over pages
    pre-split): a chunk that doesn't start a new section must inherit the most
    recent state from whichever earlier chunk last advanced it - tagging at the
    page level and only refining within each chunk's own text independently
    would lose that carry-over for chunks that fall in the middle of a section.
    """
    chapter, section = "", ""
    for chunk in chunks:
        for m in _CHAPTER_RE.finditer(chunk.page_content):
            chapter = f"Chapter {m.group(1)}"
        for m in _SECTION_RE.finditer(chunk.page_content):
            section = f"Section {m.group(1)}"
        chunk.metadata["chapter"] = chapter
        chunk.metadata["section"] = section


def _parse_metadata(filename: str) -> dict:
    """Original `name__type__env__section__date` convention, for files with no
    FILE_METADATA entry (e.g. anything dropped directly into data/ in the future)."""
    stem = os.path.splitext(filename)[0]
    parts = [p for p in stem.split("__") if p]
    md = {"source": filename, "doc_type": "", "environment": "", "topic": "", "last_updated": ""}
    keys = ["name", "doc_type", "environment", "topic", "last_updated"]
    for i, key in enumerate(keys[1:], start=1):  # skip name
        if i < len(parts):
            md[key] = parts[i]
    return md


def _base_metadata(filename: str) -> dict:
    known = FILE_METADATA.get(filename)
    if known:
        return {"source": filename, "section": "", "chapter": "", **known}
    return {"source": filename, **_parse_metadata(filename)}


def _load_page_docs(path: str, filename: str) -> list[Document]:
    """Load a file as one Document per page (PDFs) or a single Document (text)."""
    if filename.lower().endswith(".pdf"):
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(path).load()
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [Document(page_content=f.read())]


def _ensure_collection(reset: bool, dense_dim: int) -> None:
    client = config.get_qdrant_client()
    exists = client.collection_exists(config.COLLECTION)
    if reset and exists:
        client.delete_collection(config.COLLECTION)
        print(f"Wiped existing Qdrant collection '{config.COLLECTION}'")
        exists = False
    if not exists:
        client.create_collection(
            config.COLLECTION,
            vectors_config={"dense": models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )


def _upsert_batches(chunks: list[Document], batch_size: int = 128) -> None:
    client = config.get_qdrant_client()
    embeddings = config.get_embeddings()
    sparse_model = config.get_sparse_embedder()

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [c.page_content for c in batch]
        dense_vecs = embeddings.embed_documents(texts)
        sparse_vecs = list(sparse_model.embed(texts))

        points = [
            models.PointStruct(
                id=chunk.metadata["id"],
                vector={
                    "dense": dense_vec,
                    "sparse": models.SparseVector(
                        indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist()
                    ),
                },
                payload={"content": chunk.page_content, **chunk.metadata},
            )
            for chunk, dense_vec, sparse_vec in zip(batch, dense_vecs, sparse_vecs)
        ]
        client.upsert(config.COLLECTION, points=points)
        print(f"  upserted {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")


def build(reset: bool = False):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=180)

    files = []
    for dirpath, _dirnames, filenames in os.walk(config.DATA_DIR):
        for fn in filenames:
            if fn.lower().endswith((".txt", ".md", ".pdf")):
                files.append(os.path.relpath(os.path.join(dirpath, fn), config.DATA_DIR))
    if not files:
        print(f"No documents found under {config.DATA_DIR}. Add .txt/.md/.pdf files and re-run.")
        return

    all_docs = []
    for relpath in files:
        path = os.path.join(config.DATA_DIR, relpath)
        filename = os.path.basename(relpath)
        base_md = _base_metadata(filename)

        page_docs = _load_page_docs(path, filename)
        for doc in page_docs:
            doc.metadata.update(base_md)

        chunks = splitter.split_documents(page_docs)
        if base_md.get("doc_type") == "act":
            _tag_act_chunks(chunks)
        for chunk in chunks:
            chunk.metadata.setdefault("section", "")
            chunk.metadata.setdefault("chapter", "")
            chunk.metadata["id"] = str(uuid.uuid4())
            chunk.metadata["embedding_id"] = config.embedding_id()  # pin the embedder
            all_docs.append(chunk)
        print(f"  {relpath}: {len(chunks)} chunks")

    print(f"Embedding {len(all_docs)} chunks with {config.embedding_id()} (dense) + Qdrant/bm25 (sparse) ...")
    dense_dim = len(config.get_embeddings().embed_query("dimension probe"))
    _ensure_collection(reset, dense_dim)
    _upsert_batches(all_docs)
    print(f"Done. Indexed {len(all_docs)} chunks from {len(files)} files into Qdrant collection '{config.COLLECTION}' ({config.QDRANT_URL})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="wipe the index before ingesting")
    args = ap.parse_args()
    build(reset=args.reset)
