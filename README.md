# LogMind

**Agentic RAG over your documents** — ask a question, the system routes it, retrieves grounded passages via hybrid search, and answers with `[n]` citations you can trace back to the exact source. Runs on **OpenAI**, **fully local via Hugging Face transformers** (no server, model loaded in-process), or **LMStudio** (local server). Each of the four pipeline axes — LLM, embeddings, reranker, router — is an independent switch, so changing one never silently changes another.

A ground-up rebuild of the original `rag-system`: a clean crewAI agent pipeline, a one-time ingestion step, hybrid retrieval over Qdrant, real citations, a Claude-style thinking UI, and a component-level eval harness.

## ✨ Demo

### Example Query
<img width="2523" height="1267" alt="Logmind screenshot_4" src="https://github.com/user-attachments/assets/4866397e-34dc-4c33-b999-ef2ca675c96d" />
<img width="1640" height="969" alt="Logmind screenshot_5" src="https://github.com/user-attachments/assets/0c285b1a-c558-444d-99a8-9e983b07140f" />


---

## Architecture

```
                         User Query
                              │
                              ▼
                        Manager Agent
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
   Knowledge Base        Web Search       Direct Reasoning
          │                   │                   │
          ▼                   ▼                   │
  Hybrid Retrieval     Tavily Results             │
  (Dense + Sparse)            │                   │
          │                   │                   │
          ▼                   │                   │
    Cross-Encoder             │                   │
      Reranker                │                   │
          └───────────────────┴───────────────────┘
                              │
                              ▼
                      Context Assembly
                              │
                              ▼
                         GPT-OSS-20B
                              │
                              ▼
                  Streaming Response (SSE)
                              │
                              ▼
                            User
```

**Where crewAI earns its place:** the two reasoning steps (planning, synthesis) are real LLM agents; retrieval is a tool; the Manager/router is deliberately *not* an agent (a routing decision doesn't need that overhead, and every extra sequential LLM call costs real latency). Citations are deterministic by default so the `[n]` in the answer always matches the sources. Set `AGENTIC_RETRIEVAL=1` to let the Researcher agent drive retrieval via the tool (better with capable models).

## The model registry

`.env` only ever says **which** model to use (a short registry name); `backend/models.yaml` says **where it lives and how to load it**. Four independent axes:

| `.env` var | picks the… | resolved against `models.yaml` into a `ModelSpec` |
|---|---|---|
| `LLM_MODEL` | pipeline LLM | `provider` (openai/huggingface/lmstudio) + `loader` |
| `EMBEDDING_MODEL` | embedder | stamped into the index at ingest; retrieval refuses a mismatched store |
| `RERANKER_MODEL` | cross-encoder reranker | or the literal `none` to skip reranking |
| `ROUTER_MODEL` | Manager/router classifier | a generative `chat` model **or** a `zeroshot` NLI classifier |

Switching `LLM_MODEL` never touches embeddings or forces a re-ingest — only changing `EMBEDDING_MODEL` does. `config.py` is the single resolver; the old `PROVIDER`/`EMBED_PROVIDER`/`RERANKER`/`ROUTER_PROVIDER` names still exist as backward-compatible derived attributes.

## Highlights

- **Three-way routing** — a dedicated, fast `ROUTER_MODEL` (e.g. the zeroshot `deberta-v3-large-zeroshot` NLI classifier) decides `kb` / `web` / `general` before any expensive work. Ambiguity always defaults to `kb` (a false positive costs one retrieval pass; a false negative would produce a confidently-ungrounded answer).
- **Hybrid retrieval over Qdrant** — dense + sparse (BM25 via fastembed) queried together and fused **server-side** with Reciprocal Rank Fusion, run as a Docker service (`docker-compose.yml`). Replaced the old in-process Chroma + hand-rolled weighted-RRF.
- **Real model reasoning in the UI** — for LMStudio reasoning models (e.g. gpt-oss-20b), `llm/lmstudio.py` talks to LM Studio's HTTP API directly to capture the `reasoning` delta field that LangChain's `ChatOpenAI` silently drops, and decodes UTF-8 explicitly (no mojibake). The frontend shows it as a collapsible **thinking timeline** (search step + reasoning + done), with retrieved sources rendered inline.
- **Cross-encoder rerank** (`knowledge/rerank.py`) — the fused pool is reranked by scoring each (query, passage) pair directly, then truncated to `RERANK_TOP_N`; degrades to hybrid order if the reranker is unavailable.
- **Citations actually work** — context is numbered and source-labelled, so `[n]` references resolve to real sources; a click expands the thinking timeline and scrolls to that source.
- **Component-level eval** — recall/coverage/LLM-judge **plus** stage-isolated retrieval metrics (embedding / hybrid / reranker), see below.

---

## Run it

### 1. Backend
```bash
cd backend
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Create .env (gitignored) with at least LLM_MODEL / EMBEDDING_MODEL / RERANKER_MODEL / ROUTER_MODEL
# (registry names in models.yaml) + the connection vars their providers need (OPENAI_API_KEY,
# LMSTUDIO_BASE_URL/LMSTUDIO_API_KEY, HF_DEVICE/HF_LOAD_IN_4BIT). See config.py + models.yaml.

docker compose up -d          # start Qdrant (http://localhost:6333) — required before ingest/queries
python ingest.py              # build the index from data/ (run once)
uvicorn app:app --reload --port 8000     # omit --reload for a huggingface LLM_MODEL (keeps weights cached)
```

### 2. Frontend (separate terminal)
```bash
cd frontend
npm install
npm run dev                   # http://localhost:5173  (proxies /api → :8000)
```

Open http://localhost:5173 and ask away.

### Switching providers
Edit `backend/.env` — each axis names a `models.yaml` entry:

| | `provider: openai` | `provider: huggingface` (local, no server) | `provider: lmstudio` (local server) |
|---|---|---|---|
| Needs | `OPENAI_API_KEY` | CUDA `torch` + the HF repo cached on first run | LMStudio running with the model loaded |

> **Re-run `python ingest.py --reset` only when `EMBEDDING_MODEL` changes** — embeddings from different models aren't interchangeable (LogMind refuses to query a mismatched index rather than return garbage). Switching `LLM_MODEL`/`RERANKER_MODEL`/`ROUTER_MODEL` never requires a re-ingest.

> **`huggingface` loads the model in-process** (`transformers`, 4-bit by default via `bitsandbytes`) — no server. The agent pipeline (`HFTransformersLLM`) and streamed synthesis (`HFChatModel`) share a per-model-path cache (`@lru_cache(maxsize=2)`), so the main model and the router's model can both stay resident. A "thinking" model that emits `<think>...</think>` (e.g. `Qwen3-4B-Thinking-2507`) has that block stripped from output and streamed to the UI as the thinking timeline. See `HF_DEVICE`/`HF_LOAD_IN_4BIT`/`HF_MAX_NEW_TOKENS` in `config.py`.

> **`lmstudio` bypasses LangChain for the chat-streaming path** (`llm/lmstudio.py`) to capture the raw `reasoning` field and avoid encoding mojibake — see Highlights above.

> **The router** (`ROUTER_MODEL`) is a fully independent axis so it can stay small/fast even when the main pipeline is large/slow. `config.ROUTER_LOADER` selects a generative one-word `chat` classifier or a `zeroshot` NLI classifier (the current default). Falls back to `kb` on any failure.

### Eval
```bash
cd backend
python -m eval_pipeline.runner                              # end-to-end: recall + coverage + LLM-judge + retrieval metrics
python -m eval_pipeline.runner --retrieval-only             # ONLY component retrieval metrics — no LLM, seconds not minutes
python -m eval_pipeline.runner --no-judge                   # skip the LLM judge (keep recall/coverage/retrieval)
python -m eval_pipeline.runner --matrix eval_pipeline/data/matrix.json   # across LLM/embedder/reranker/router combos
python -m eval_pipeline.dataset_gen --n 60                  # generate candidate Q/A pairs (with gold_passage) for review
```

`eval_pipeline/` grades two layers. **End-to-end** (always, unless `--retrieval-only`): retrieval recall, term coverage, and LLM-judge faithfulness/correctness (a fixed OpenAI judge, regardless of the LLM under test). **Component-level retrieval** (always): for each gold item it measures where the gold passage ranks at three stages — **dense-only** (the embedder in isolation), **hybrid pool** (dense+sparse+RRF), and **reranked** — reporting MRR / Recall@k / NDCG@10 and the reranker's *delta* over the hybrid pool (the "is the cross-encoder earning its keep?" signal). Relevance is judged at the **passage** level via token-overlap (re-ingest proof). `dataset_gen.py` auto-captures `gold_passage` and runs a self-retrieval gate (drops generated questions whose own source chunk isn't retrievable). Every run appends to `eval_pipeline/data/history.jsonl`; `--threshold N` exits 1 if any combo scores below it (CI-gateable).

---

## Layout
```
logmind/
├── backend/
│   ├── app.py             # FastAPI: /api/query(/stream), /api/documents, /api/analytics, /api/health
│   ├── config.py          # resolves LLM_MODEL/EMBEDDING_MODEL/RERANKER_MODEL/ROUTER_MODEL → models.yaml
│   ├── models.yaml         # the model registry: short name → {role, provider, loader, identifier}
│   ├── ingest.py          # one-time: load → chunk → embed (dense+sparse) → upsert to Qdrant
│   ├── docker-compose.yml  # Qdrant service
│   ├── agents/
│   │   ├── crew.py        # crewAI agents (Strategist/Researcher/Synthesizer) + run_query()/stream_query()
│   │   └── manager.py     # kb/web/general router — single classification call, not a crewAI agent
│   ├── knowledge/
│   │   ├── retrieval.py   # hybrid (dense + sparse, Qdrant server-side RRF) + numbered-source collector
│   │   └── rerank.py      # cross-encoder rerank of the hybrid candidate pool
│   ├── llm/
│   │   ├── huggingface.py # provider: huggingface — local transformers, no server
│   │   ├── lmstudio.py    # provider: lmstudio — raw reasoning capture for the chat-streaming path
│   │   └── zeroshot.py    # loader: zeroshot — NLI-based router classifier
│   ├── tools/
│   │   ├── analytics.py   # query_log.json read/write + /api/analytics aggregation
│   │   └── websearch.py   # Tavily-backed `web` route
│   ├── eval_pipeline/     # runner.py, judge.py, retrieval_metrics.py, dataset_gen.py + data/
│   ├── data/              # source documents (drop your PDFs here too)
│   └── requirements.txt
└── frontend/              # Vite + React (TypeScript, TanStack Router/Query) chat console
    └── src/ (routes/, components/ThoughtBlock/ChatMessage/AppShell, lib/api.ts)
```

## Notes / honest limits
- Add your own PDFs to `backend/data/` (subfolders are fine — ingestion walks recursively) and re-run `ingest.py` — the loader handles `.txt`, `.md`, `.pdf`.
- Filenames are tagged with metadata one of two ways: the `name__type__env__section__date.ext` convention (`_parse_metadata`), or an explicit per-filename entry in `ingest.py`'s `FILE_METADATA` table for human-named files (e.g. the bundled `Tax Knowledge base/` corpus — see the workspace-root `CLAUDE.md`). Files matching neither get sane empty defaults.
- Reranking is selected by `RERANKER_MODEL` (a `models.yaml` name, e.g. `ms-marco-minilm-l6-v2` / `bge-reranker-v2-m3`, or `none`).
- Single-tenant prototype: no auth; Qdrant runs as a local Docker service (`docker compose up -d` before any query); `.env` is gitignored (no `.env.example` checked in — hand-write one from `config.py` + `models.yaml`).
