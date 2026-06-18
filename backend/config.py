"""
LogMind - central configuration.

One environment switch (PROVIDER=openai|ollama) drives BOTH the LLM and the
embedding model, and the embedding model name is recorded in the Chroma
collection at ingestion time so retrieval can refuse to run against a store that
was built with a different embedder. (That silent mismatch was the single most
damaging bug in the original system.)
"""

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "openai").lower()

# Where the persistent Chroma store lives, and where source documents are read from.
ROOT = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.getenv("DB_DIR", os.path.join(ROOT, "db", "chroma"))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(ROOT, "data"))
COLLECTION = os.getenv("COLLECTION", "logmind")

# Model names per provider (override via env).
OPENAI_LLM = os.getenv("OPENAI_LLM", "gpt-4o-mini")
OPENAI_EMBED = os.getenv("OPENAI_EMBED", "text-embedding-3-small")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM = os.getenv("OLLAMA_LLM", "llama3.1:8b-instruct-q4_K_M")
OLLAMA_EMBED = os.getenv("OLLAMA_EMBED", "nomic-embed-text")

# Reranking (cross-encoder stage after hybrid retrieval).
#   flashrank      - lightweight cross-encoder, local, no torch (default)
#   crossencoder   - sentence-transformers CrossEncoder (heavier, GPU-capable)
#   none           - skip reranking, use hybrid order
RERANKER = os.getenv("RERANKER", "flashrank").lower()
RERANK_MODEL = os.getenv("RERANK_MODEL", "")  # empty -> backend default
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))   # chunks kept after rerank
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "12"))    # pool size fed to the reranker


def embedding_id() -> str:
    """Stable identifier for the active embedding model, stored with the index."""
    if PROVIDER == "ollama":
        return f"ollama:{OLLAMA_EMBED}"
    return f"openai:{OPENAI_EMBED}"


@lru_cache(maxsize=1)
def get_embeddings():
    """LangChain embeddings used for BOTH ingestion and retrieval."""
    if PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=OLLAMA_EMBED, base_url=OLLAMA_BASE_URL)

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=OPENAI_EMBED, api_key=os.environ["OPENAI_API_KEY"])


def get_crew_llm():
    """crewAI LLM (LiteLLM under the hood) for the agents."""
    from crewai import LLM

    if PROVIDER == "ollama":
        return LLM(
            model=f"ollama/{OLLAMA_LLM}",
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        )

    return LLM(
        model=OPENAI_LLM,
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.1,
    )


def get_chat_model():
    """LangChain chat model used for token-streamed answer synthesis.

    Both backends expose `.stream()` yielding chunks with `.content`, so the
    streaming code stays provider-agnostic.
    """
    if PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=OLLAMA_LLM, base_url=OLLAMA_BASE_URL, temperature=0.1)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=OPENAI_LLM,
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.1,
        streaming=True,
    )


def active_model_label() -> str:
    return OLLAMA_LLM if PROVIDER == "ollama" else OPENAI_LLM
