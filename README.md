# LogMind

**Agentic RAG over your documents** — ask a question, watch it plan sub-queries, retrieve grounded passages, and answer with citations you can trace back to the exact source. Runs on **OpenAI** or **fully local via Ollama** with one switch.

A ground-up rebuild of the original `rag-system`: a clean crewAI agent pipeline, a proper one-time ingestion step, hybrid retrieval done right, real citations, and an eval harness.

## ✨ Demo

### Empty State
<img width="2185" height="828" alt="Logmind screenshot_1" src="https://github.com/user-attachments/assets/06960202-c2f9-4b43-8993-4c59efb05890" />

### Example Query
<img width="2167" height="1244" alt="Logmind screenshot_2" src="https://github.com/user-attachments/assets/847bf7c1-bf0c-454d-a755-e0805c07010d" />

---

## Architecture

```
  question
     │
     ▼
  ┌──────────────────┐   crewAI agent
  │ Query Strategist │   → focused sub-queries (1–4, or leaves simple Qs alone)
  └──────────────────┘
     │
     ▼
  ┌──────────────────┐   hybrid retrieval (deterministic, or agentic via tool)
  │   Researcher     │   → vector (MMR) + BM25 over the SAME corpus, in Chroma
  └──────────────────┘
     │  numbered, source-labelled context  [1] [2] [3] …
     ▼
  ┌──────────────────┐   crewAI agent
  │ Answer Synthesizer│  → grounded answer with [n] citations; flags missing info
  └──────────────────┘
     │
     ▼
  React UI: answer + clickable [n] citations + retrieved sources + the planned sub-queries
```

**Where crewAI earns its place:** the two reasoning steps (planning, synthesis) are real LLM agents; retrieval is a tool. Citations are deterministic by default so the `[n]` in the answer always matches the sources panel. Set `AGENTIC_RETRIEVAL=1` to let the Researcher agent drive retrieval via the tool (better with capable models).

## What changed vs. the original (the review, applied)

- **Ingestion is now a separate one-time step** (`ingest.py`) — no more re-reading and re-chunking one hard-coded file on every request, and PDFs are actually indexed.
- **Hybrid retrieval is consistent** — BM25 is built over the *same* corpus as the vectors (not just the current request's chunks), with **MMR** for diversity.
- **Embeddings can't drift** — one provider switch drives the LLM *and* embeddings, and the embedding model is pinned into the index; retrieval refuses to run against a mismatched store.
- **Citations actually work** — context is numbered and source-labelled, so `[n]` references resolve to real sources.
- **One planning call, robustly parsed** — replaces the 3-call decompose/dedup chain and the fragile `split("\n")`.
- **Eval harness** (`eval.py`) — measure retrieval hit-rate instead of guessing.
- Dead/duplicate modules and the committed `venv` are gone.

---

## Run it

### 1. Backend
```bash
cd backend
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then edit: pick PROVIDER + keys/models

python ingest.py              # build the index from data/ (run once)
uvicorn app:app --reload --port 8000
```

### 2. Frontend (separate terminal)
```bash
cd frontend
npm install
npm run dev                   # http://localhost:5173  (proxies /api → :8000)
```

Open http://localhost:5173 and ask away.

### Switching providers
Edit `backend/.env`:

| | `PROVIDER=openai` | `PROVIDER=ollama` (local) |
|---|---|---|
| LLM | `OPENAI_LLM` (gpt-4o-mini) | `OLLAMA_LLM` (llama3.1:8b…) |
| Embeddings | `text-embedding-3-small` | `nomic-embed-text` |
| Needs | `OPENAI_API_KEY` | `ollama pull <llm>` + `ollama pull nomic-embed-text` |

> **Re-run `python ingest.py --reset` when you switch providers** — embeddings from different models aren't interchangeable (LogMind will refuse to query a mismatched index rather than return garbage).

### Eval
```bash
cd backend && python eval.py     # retrieval hit-rate + answer coverage
```

---

## Layout
```
logmind/
├── backend/
│   ├── app.py          # FastAPI: /api/query, /api/documents, /api/health
│   ├── config.py       # provider switch: LLM + embeddings, pinned together
│   ├── ingest.py       # one-time: load → chunk → embed → persist to Chroma
│   ├── retrieval.py    # hybrid (vector MMR + BM25) + numbered-source collector
│   ├── rag_crew.py     # crewAI agents + HybridSearch tool + run_query()
│   ├── eval.py         # retrieval hit-rate / answer coverage
│   ├── data/           # source documents (drop your PDFs here too)
│   └── requirements.txt, .env.example
└── frontend/           # Vite + React answer console
    └── src/ (App, components/, api.js, styles.css)
```

## Notes / honest limits
- Add your own PDFs to `backend/data/` and re-run `ingest.py` — the loader handles `.txt`, `.md`, `.pdf`.
- No reranker yet — hybrid + MMR is the baseline; a cross-encoder rerank is the natural next precision bump.
- Single-tenant prototype: no auth, local Chroma file store.
