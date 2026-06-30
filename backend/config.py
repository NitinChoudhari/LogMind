"""
LogMind - central configuration.

Each of the four independent pipeline axes - LLM, embeddings, reranker,
router - is selected in `.env` by a short registry name (`LLM_MODEL`,
`EMBEDDING_MODEL`, `RERANKER_MODEL`, `ROUTER_MODEL`), resolved here against
`models.yaml` (see that file's header for the registry schema) into a
`ModelSpec`. `.env` only ever says *which* model to use; `models.yaml` says
*where it lives and how to load it*. Switching the LLM never silently swaps
the embedding model out from under an existing index - the embedding
model's registry name is recorded with every chunk at ingestion time so
retrieval can refuse to run against a store built with a different
embedder (that silent mismatch was the single most damaging bug in the
original system).

`PROVIDER`/`EMBED_PROVIDER`/`RERANKER`/`ROUTER_PROVIDER` still exist as
module-level attributes for backward compatibility with existing call
sites (app.py, agents/crew.py, agents/manager.py, knowledge/rerank.py,
eval_pipeline/runner.py) - they are now *derived* from the resolved
ModelSpecs rather than read raw from env.
"""

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Model registry (models.yaml)
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.abspath(__file__))
_REGISTRY_PATH = os.path.join(ROOT, "models.yaml")

_ALLOWED_ROLES = {"llm", "embedding", "reranker", "router"}
_ALLOWED_PROVIDERS = {"openai", "lmstudio", "huggingface", "sentence-transformers"}
_ALLOWED_LOADERS = {
    "llm": {"chat"},
    "embedding": {"embedding"},
    "reranker": {"cross-encoder", "cross-encoder-causal"},
    "router": {"chat", "zeroshot"},
}


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to locate and load one registry entry. The only
    object resolvers hand back - never a raw dict or bare string."""

    name: str
    role: str
    provider: str
    loader: str
    identifier: str
    display_name: str | None = None


@lru_cache(maxsize=1)
def _load_registry() -> dict:
    """Loads + validates models.yaml once. Fails fast with a descriptive
    error naming the offending entry rather than surfacing a confusing
    error later at model-load time."""
    import yaml

    with open(_REGISTRY_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    seen_identifiers: dict[tuple, str] = {}  # (role, identifier) -> entry name
    for name, entry in raw.items():
        missing = [k for k in ("role", "provider", "loader", "identifier") if not entry.get(k)]
        if missing:
            raise ValueError(f"models.yaml: '{name}' is missing required field(s): {missing}")

        role, provider, loader, identifier = (
            entry["role"], entry["provider"], entry["loader"], entry["identifier"],
        )
        if role not in _ALLOWED_ROLES:
            raise ValueError(
                f"models.yaml: '{name}' has unknown role '{role}' (expected one of {sorted(_ALLOWED_ROLES)})"
            )
        if provider not in _ALLOWED_PROVIDERS:
            raise ValueError(
                f"models.yaml: '{name}' has unknown provider '{provider}' "
                f"(expected one of {sorted(_ALLOWED_PROVIDERS)})"
            )
        if loader not in _ALLOWED_LOADERS[role]:
            raise ValueError(
                f"models.yaml: '{name}' has loader '{loader}' invalid for role '{role}' "
                f"(expected one of {sorted(_ALLOWED_LOADERS[role])})"
            )

        dup_key = (role, identifier)
        if dup_key in seen_identifiers:
            raise ValueError(
                f"models.yaml: '{name}' duplicates identifier '{identifier}' already used by "
                f"'{seen_identifiers[dup_key]}' for role '{role}'"
            )
        seen_identifiers[dup_key] = name
        # display_name/enabled/tags and any other optional metadata are
        # tolerated without validation - no application code reads them yet.

    return raw


def _resolve(name: str, role: str) -> ModelSpec:
    """The only way any code reads a registry entry. `name` is empty/None
    when its env var is unset - reported the same way as 'not found' since
    both are equally a missing-config error."""
    registry = _load_registry()
    entry = registry.get(name or "")
    if entry is None:
        raise ValueError(
            f"models.yaml has no entry named '{name}' (referenced as {role.upper()}_MODEL)"
        )
    if entry["role"] != role:
        raise ValueError(
            f"models.yaml: '{name}' is tagged role '{entry['role']}', but was referenced as "
            f"a {role} model ({role.upper()}_MODEL={name})"
        )
    return ModelSpec(
        name=name,
        role=entry["role"],
        provider=entry["provider"],
        loader=entry["loader"],
        identifier=entry["identifier"],
        display_name=entry.get("display_name"),
    )


def _apply_llm(name: str) -> None:
    global LLM_MODEL, _LLM_SPEC, PROVIDER, LLM_IDENTIFIER
    LLM_MODEL = name
    _LLM_SPEC = _resolve(name, "llm")
    PROVIDER = _LLM_SPEC.provider
    LLM_IDENTIFIER = _LLM_SPEC.identifier


def _apply_embedding(name: str) -> None:
    global EMBEDDING_MODEL, _EMBEDDING_SPEC, EMBED_PROVIDER, EMBEDDING_IDENTIFIER
    EMBEDDING_MODEL = name
    _EMBEDDING_SPEC = _resolve(name, "embedding")
    EMBED_PROVIDER = _EMBEDDING_SPEC.provider
    EMBEDDING_IDENTIFIER = _EMBEDDING_SPEC.identifier


def _apply_reranker(name: str) -> None:
    """RERANKER_MODEL=none (case-insensitive) is a reserved literal meaning
    "skip reranking entirely" - it is never looked up in the registry."""
    global RERANKER_MODEL, _RERANKER_SPEC, RERANKER, RERANK_MODEL
    RERANKER_MODEL = name
    if (name or "none").lower() == "none":
        _RERANKER_SPEC = None
        RERANKER = "none"
        RERANK_MODEL = ""
        return
    _RERANKER_SPEC = _resolve(name, "reranker")
    RERANKER = _RERANKER_SPEC.loader  # rerank.py's branch is "which loading code path"
    RERANK_MODEL = _RERANKER_SPEC.identifier


def _apply_router(name: str) -> None:
    global ROUTER_MODEL, _ROUTER_SPEC, ROUTER_PROVIDER, ROUTER_LOADER, ROUTER_MODEL_PATH
    ROUTER_MODEL = name
    _ROUTER_SPEC = _resolve(name, "router")
    ROUTER_PROVIDER = _ROUTER_SPEC.provider
    ROUTER_LOADER = _ROUTER_SPEC.loader
    ROUTER_MODEL_PATH = _ROUTER_SPEC.identifier


# Resolve all four axes at import time. set_*_model() (used by
# eval_pipeline/runner.py's --matrix mode to swap axes between combos) call
# the same _apply_* functions plus bust the caches that would otherwise keep
# serving the previous combo's objects.
_apply_llm(os.getenv("LLM_MODEL"))
_apply_embedding(os.getenv("EMBEDDING_MODEL"))
_apply_reranker(os.getenv("RERANKER_MODEL", "none"))
_apply_router(os.getenv("ROUTER_MODEL"))


def set_llm_model(name: str) -> None:
    _apply_llm(name)


def set_embedding_model(name: str) -> None:
    _apply_embedding(name)
    get_embeddings.cache_clear()


def set_reranker_model(name: str) -> None:
    _apply_reranker(name)


def set_router_model(name: str) -> None:
    _apply_router(name)


# --------------------------------------------------------------------------- #
# Everything else: connection details + runtime behavior (NOT model identity)
# --------------------------------------------------------------------------- #
# Where source documents are read from, and where the Qdrant index lives.
# Qdrant runs as a separate service (see docker-compose.yml: `docker compose up -d`),
# not in-process - QDRANT_URL points at it. Vectors are named "dense" (the active
# embedding model) and "sparse" (BM25 via fastembed's Qdrant/bm25 model); both are
# queried in one request and fused server-side (see knowledge/retrieval.py).
DATA_DIR = os.getenv("DATA_DIR", os.path.join(ROOT, "data"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION", "logmind")

# LM Studio server connection (shared by PROVIDER=lmstudio and/or
# ROUTER_PROVIDER=lmstudio - there's one local server, not one per axis).
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")

# PROVIDER=huggingface / ROUTER_PROVIDER=huggingface loading knobs - not
# model identity, so these stay plain env vars rather than registry fields.
HF_DEVICE = os.getenv("HF_DEVICE", "")  # "" -> auto: cuda if available else cpu
HF_LOAD_IN_4BIT = os.getenv("HF_LOAD_IN_4BIT", "1") != "0"
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "2048"))  # thinking + answer share this budget

# Reranking pool/result sizes (not model identity).
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))   # chunks kept after rerank
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "12"))    # pool size fed to the reranker

# Candidate-label text for the zeroshot router, scored via NLI entailment.
# Tied to the CURRENT corpus (Tax Knowledge base) on purpose: NLI models score
# concrete topical labels far more accurately than abstract meta-descriptions
# like "needs a document lookup" - so unlike the generative-LLM router
# branches, this one needs a corpus-specific label to work well, and must be
# updated here (or overridden) if the indexed corpus changes domain.
ROUTER_ZEROSHOT_KB_LABEL = os.getenv(
    "ROUTER_ZEROSHOT_KB_LABEL",
    "a question about tax law, income tax, deductions, or financial regulations",
)
ROUTER_ZEROSHOT_WEB_LABEL = os.getenv(
    "ROUTER_ZEROSHOT_WEB_LABEL",
    "a question that needs current, real-time, or recent news information",
)
ROUTER_ZEROSHOT_GENERAL_LABEL = os.getenv(
    "ROUTER_ZEROSHOT_GENERAL_LABEL",
    "a general question not related to tax law or financial regulations",
)

# `web` route (agents/manager.py + tools/websearch.py) - Tavily search API.
# The router always offers "web" as a possible classification regardless of
# whether this is set; a missing/invalid key is handled per-request (graceful
# degradation to the general-knowledge answer), not by hiding the route.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))


def embedding_id() -> str:
    """Stable identifier for the active embedding model, stored with the
    index. Just the registry name - guaranteed unique by construction
    (it's the models.yaml dict key)."""
    return EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings():
    """LangChain-compatible embeddings used for BOTH ingestion and
    retrieval. Selected via EMBEDDING_MODEL, independent of LLM_MODEL."""
    if _EMBEDDING_SPEC.provider == "sentence-transformers":
        from sentence_transformers import SentenceTransformer

        import torch

        _st = SentenceTransformer(
            _EMBEDDING_SPEC.identifier,
            model_kwargs={"torch_dtype": torch.float16},
            device="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
        )

        # Inner class captures _st from the closure.
        # embed_query uses prompt_name="query" to prepend the instruction prefix
        # defined in the model's config_sentence_transformers.json; embed_documents
        # uses no prompt (documents are embedded as-is, which is what the model expects).
        class _Qwen3Embeddings:
            def embed_documents(self, texts):
                return _st.encode(texts, show_progress_bar=False).tolist()

            def embed_query(self, text: str):
                return _st.encode(
                    [text], prompt_name="query", show_progress_bar=False
                )[0].tolist()

        return _Qwen3Embeddings()

    if _EMBEDDING_SPEC.provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        import torch

        device = HF_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
        return HuggingFaceEmbeddings(
            model_name=_EMBEDDING_SPEC.identifier, model_kwargs={"device": device}
        )


@lru_cache(maxsize=1)
def get_qdrant_client():
    """Shared client for the Qdrant service (see docker-compose.yml). Both
    ingest.py and knowledge/retrieval.py go through this single accessor."""
    from qdrant_client import QdrantClient

    return QdrantClient(url=QDRANT_URL)


@lru_cache(maxsize=1)
def get_sparse_embedder():
    """fastembed's BM25 sparse-vector model - the lexical half of Qdrant's native
    dense+sparse hybrid search, replacing the old in-process BM25Retriever-over-
    the-full-corpus approach."""
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name="Qdrant/bm25")


def get_crew_llm():
    """crewAI LLM for the agents (LiteLLM-backed for openai/lmstudio; a local
    in-process transformers model for huggingface - see llm/huggingface.py)."""
    if _LLM_SPEC.provider == "lmstudio":
        from crewai import LLM

        return LLM(
            model=_LLM_SPEC.identifier,
            base_url=LMSTUDIO_BASE_URL,
            api_key=LMSTUDIO_API_KEY,
            temperature=0.1,
        )

    if _LLM_SPEC.provider == "huggingface":
        from llm.huggingface import HFTransformersLLM

        return HFTransformersLLM(model=_LLM_SPEC.identifier)

    from crewai import LLM

    return LLM(
        model=_LLM_SPEC.identifier,
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.1,
    )


def get_chat_model():
    """Chat model used for token-streamed answer synthesis.

    All backends expose `.stream()` yielding chunks with `.content`, so the
    streaming code in agents/crew.py stays provider-agnostic.
    """
    if _LLM_SPEC.provider == "lmstudio":
        from llm.lmstudio import LMStudioChatModel

        return LMStudioChatModel(
            model=_LLM_SPEC.identifier,
            base_url=LMSTUDIO_BASE_URL,
            api_key=LMSTUDIO_API_KEY,
            temperature=0.1,
        )

    if _LLM_SPEC.provider == "huggingface":
        from llm.huggingface import HFChatModel

        return HFChatModel(temperature=0.1)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=_LLM_SPEC.identifier,
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.1,
        streaming=True,
    )


def get_router_llm_label() -> str:
    """Model label used for the Manager's classification call."""
    return _ROUTER_SPEC.display_name or _ROUTER_SPEC.identifier


def active_model_label() -> str:
    return _LLM_SPEC.display_name or _LLM_SPEC.identifier


@lru_cache(maxsize=1)
def _tiktoken_encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Token count for the streamed answer text, used for the tok/sec UI metric.

    Exact for `huggingface` (uses the loaded model's own tokenizer). For
    openai/lmstudio there is no in-process tokenizer for the active model, so
    this falls back to OpenAI's cl100k_base encoding via tiktoken - a real
    count, just not necessarily byte-identical to the active model's own
    tokenizer.
    """
    if not text:
        return 0
    if PROVIDER == "huggingface":
        from llm.huggingface import count_tokens as hf_count_tokens

        return hf_count_tokens(text)
    try:
        return len(_tiktoken_encoding().encode(text))
    except Exception:
        return len(text.split())
