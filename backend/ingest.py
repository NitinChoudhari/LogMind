"""
LogMind - ingestion (run once, or whenever documents change).

    python ingest.py            # ingest everything under data/
    python ingest.py --reset    # wipe the index first (do this when you change
                                # chunking strategy OR switch embedding provider)

Separating ingestion from the request path fixes the original system, which
re-read and re-chunked a single hard-coded file on every query and never indexed
the PDFs at all.

Filenames may optionally encode metadata as `name__type__env__section__date.ext`
(your existing convention); we parse it defensively and fall back to sane values.
"""

import argparse
import os
import shutil
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

import config


def _parse_metadata(filename: str) -> dict:
    stem = os.path.splitext(filename)[0]
    parts = [p for p in stem.split("__") if p]
    md = {"source": filename, "type": "", "environment": "", "section": "", "last_updated": ""}
    keys = ["name", "type", "environment", "section", "last_updated"]
    for i, key in enumerate(keys[1:], start=1):  # skip name
        if i < len(parts):
            md[key] = parts[i]
    return md


def _load_file(path: str, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        from langchain_community.document_loaders import PyPDFLoader

        pages = PyPDFLoader(path).load()
        return "\n".join(p.page_content for p in pages)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def build(reset: bool = False):
    if reset and os.path.isdir(config.DB_DIR):
        shutil.rmtree(config.DB_DIR)
        print(f"Wiped existing index at {config.DB_DIR}")
    os.makedirs(config.DB_DIR, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

    files = [
        f for f in os.listdir(config.DATA_DIR)
        if f.lower().endswith((".txt", ".md", ".pdf"))
    ]
    if not files:
        print(f"No documents found in {config.DATA_DIR}. Add .txt/.md/.pdf files and re-run.")
        return

    all_docs = []
    for filename in files:
        path = os.path.join(config.DATA_DIR, filename)
        text = _load_file(path, filename)
        base_md = _parse_metadata(filename)
        chunks = splitter.split_text(text)
        for chunk in chunks:
            md = dict(base_md)
            md["id"] = str(uuid.uuid4())
            md["embedding_id"] = config.embedding_id()  # pin the embedder
            all_docs.append(Document(page_content=chunk, metadata=md))
        print(f"  {filename}: {len(chunks)} chunks")

    print(f"Embedding {len(all_docs)} chunks with {config.embedding_id()} ...")
    Chroma.from_documents(
        documents=all_docs,
        embedding=config.get_embeddings(),
        collection_name=config.COLLECTION,
        persist_directory=config.DB_DIR,
    )
    print(f"Done. Indexed {len(all_docs)} chunks from {len(files)} files into {config.DB_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="wipe the index before ingesting")
    args = ap.parse_args()
    build(reset=args.reset)
