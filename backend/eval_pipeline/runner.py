"""
LogMind - eval pipeline runner.

    python -m eval_pipeline.runner                                  # current .env config
    python -m eval_pipeline.runner --matrix eval_pipeline/data/matrix.json
    python -m eval_pipeline.runner --no-judge                       # skip LLM-judge calls (fast/free)
    python -m eval_pipeline.runner --retrieval-only                 # ONLY component metrics, no LLM
    python -m eval_pipeline.runner --threshold 0.8                  # exit 1 if any combo scores below this
    python -m eval_pipeline.runner --eval-set path/to/other_set.json

For each labelled question in the eval set, measures:
  * retrieval_recall - fraction of `expected_sources` actually retrieved (skipped in --retrieval-only)
  * term_coverage    - does the answer contain every `expect_terms` entry (cheap, deterministic,
                        always computed even with --no-judge; skipped in --retrieval-only)
  * faithfulness     - LLM-judge: is the answer supported by the numbered context the synthesizer
                        actually saw? (skipped with --no-judge / --retrieval-only)
  * correctness      - LLM-judge: does the answer match reference_answer's key facts? (skipped
                        with --no-judge / --retrieval-only)
  * retrieval metrics - component-level MRR / Recall@k / NDCG@10 at three stages (dense-only,
                        hybrid pool, reranked) plus the reranker's delta over the hybrid pool, via
                        eval_pipeline/retrieval_metrics.py. No LLM involved - always computed.

`--retrieval-only` skips run_query()/the judge entirely and reports just the component metrics
(seconds, not minutes) - its combined_score is the mean of hybrid MRR, reranked MRR, and reranked
Recall@RERANK_TOP_N. Otherwise `combined_score` is the mean of the judge/recall metrics as before.
Results are appended to eval_pipeline/data/history.jsonl (one line per run, all combos) so
regressions show up as a number over time rather than a vibe.

Matrix mode swaps axes between combos via config's set_*_model() setters (which re-resolve each
models.yaml entry into a fresh ModelSpec) and busts the relevant caches (`config.get_embeddings`,
the `agents.crew._retriever` singleton) so each combo's retriever/LLM actually reflect the override
- see `_apply_combo`. A combo that changes `embedding_model` needs an index that was actually built
with that embedder; the Qdrant collection only holds one index at a time, so such a combo is skipped
(not silently run against the wrong embeddings) unless a matching index is already there - see
`_index_matches`.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import config
import agents.crew as agents_crew
from agents.crew import run_query
from eval_pipeline.retrieval_metrics import evaluate_retrieval

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_EVAL_SET = os.path.join(DATA_DIR, "eval_set.json")
HISTORY_PATH = os.path.join(DATA_DIR, "history.jsonl")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Matrix mode plumbing
# --------------------------------------------------------------------------- #
def _apply_combo(combo: dict) -> None:
    """Switch the named axes to different models.yaml entries for this combo,
    via config's set_*_model() setters (which re-resolve the ModelSpec and
    bust the caches that would otherwise keep serving the previous combo's
    objects)."""
    if "llm_model" in combo:
        config.set_llm_model(combo["llm_model"])
    if "embedding_model" in combo:
        config.set_embedding_model(combo["embedding_model"])
    if "reranker_model" in combo:
        config.set_reranker_model(combo["reranker_model"])
    if "router_model" in combo:
        config.set_router_model(combo["router_model"])

    agents_crew._retriever = None  # forces get_retriever() to rebuild against the new embeddings


def _index_matches(expected_embedding_id: str) -> bool:
    """Mirrors retrieval.py's own embedding-mismatch guard, checked up front so a
    mismatched combo can be skipped with a clear message instead of failing on
    every single eval item with the same RuntimeError."""
    try:
        client = config.get_qdrant_client()
        if not client.collection_exists(config.COLLECTION):
            return True  # can't tell - let it try and fail loudly rather than false-skip
        points, _ = client.scroll(config.COLLECTION, limit=1, with_payload=["embedding_id"])
        return not points or points[0].payload.get("embedding_id") == expected_embedding_id
    except Exception:
        return True  # can't tell - let it try and fail loudly rather than false-skip


def _combo_label(combo: dict) -> str:
    return combo.get("label") or (
        f"{combo.get('llm_model', config.LLM_MODEL)}"
        f"/{combo.get('embedding_model', config.EMBEDDING_MODEL)}"
        f"/{combo.get('reranker_model', config.RERANKER_MODEL)}"
    )


# --------------------------------------------------------------------------- #
# Per-item / per-combo evaluation
# --------------------------------------------------------------------------- #
def _run_item(item: dict, use_judge: bool) -> dict:
    res = run_query(item["question"])
    sources = {s["source"] for s in res.get("sources", [])}
    expected = set(item.get("expected_sources", []))
    recall = (len(expected & sources) / len(expected)) if expected else 1.0

    answer = res.get("answer") or ""
    answer_lower = answer.lower()
    expect_terms = item.get("expect_terms", [])
    term_coverage = all(t.lower() in answer_lower for t in expect_terms) if expect_terms else True

    row = {
        "id": item.get("id", item["question"][:40]),
        "retrieval_recall": recall,
        "term_coverage": term_coverage,
    }

    if use_judge:
        from eval_pipeline.judge import score_correctness, score_faithfulness

        faith = score_faithfulness(res.get("context", ""), answer)
        correct = score_correctness(item["question"], item.get("reference_answer", ""), answer)
        row["faithfulness"] = faith.score
        row["correctness"] = correct.score

    return row


def _combined_score(rows: list[dict]) -> float:
    metric_keys = ["retrieval_recall", "term_coverage", "faithfulness", "correctness"]
    totals, counts = {}, {}
    for row in rows:
        for k in metric_keys:
            if k in row:
                totals[k] = totals.get(k, 0.0) + (1.0 if row[k] is True else 0.0 if row[k] is False else row[k])
                counts[k] = counts.get(k, 0) + 1
    per_metric_avgs = [totals[k] / counts[k] for k in totals]
    return sum(per_metric_avgs) / len(per_metric_avgs) if per_metric_avgs else 0.0


def _retr_avg(rows: list[dict], stage: str, metric: str) -> float:
    vals = [r["retrieval"][stage][metric] for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


def _aggregate_retrieval(rows: list[dict], rkey: str) -> dict:
    """Average each stage's metrics across all items into the combo summary
    (per-item retrieval dicts are stripped from history along with `rows`)."""
    return {
        "dense_mrr": _retr_avg(rows, "dense", "mrr"),
        "hybrid_mrr": _retr_avg(rows, "hybrid", "mrr"),
        "reranked_mrr": _retr_avg(rows, "reranked", "mrr"),
        "hybrid_ndcg@10": _retr_avg(rows, "hybrid", "ndcg@10"),
        "reranked_ndcg@10": _retr_avg(rows, "reranked", "ndcg@10"),
        f"dense_{rkey}": _retr_avg(rows, "dense", rkey),
        f"hybrid_{rkey}": _retr_avg(rows, "hybrid", rkey),
        f"reranked_{rkey}": _retr_avg(rows, "reranked", rkey),
        "rerank_delta_mrr": _retr_avg(rows, "rerank_delta", "mrr"),
        f"rerank_delta_{rkey}": _retr_avg(rows, "rerank_delta", rkey),
        "rerank_delta_ndcg@10": _retr_avg(rows, "rerank_delta", "ndcg@10"),
    }


def run_combo(label: str, eval_set: list[dict], use_judge: bool, retrieval_only: bool) -> dict:
    retriever = agents_crew.get_retriever()
    final_k = config.RERANK_TOP_N
    rkey = f"recall@{final_k}"

    rows = []
    for item in eval_set:
        row = {"id": item.get("id", item["question"][:40])}
        if not retrieval_only:
            row.update(_run_item(item, use_judge))
        row["retrieval"] = evaluate_retrieval(item, retriever, final_k)
        rows.append(row)

    print(f"\n=== {label} ===")

    # End-to-end judge/recall table (full runs only).
    if not retrieval_only:
        def _fmt(v):
            return f"{v:<7.2f}" if isinstance(v, float) else f"{'-':<7}"
        print(f"{'id':<32} {'recall':<8} {'terms':<7} {'faith':<7} {'correct':<7}")
        print("-" * 70)
        for row in rows:
            print(
                f"{row['id'][:31]:<32} "
                f"{row['retrieval_recall']:<8.2f} "
                f"{('ok' if row['term_coverage'] else '--'):<7} "
                f"{_fmt(row.get('faithfulness'))} "
                f"{_fmt(row.get('correctness'))}"
            )

    # Component-level retrieval table (always).
    print(f"{'id':<32} {'emb_mrr':<8} {'hyb_mrr':<8} {'rr_mrr':<8} {'rr_dmrr':<8} {'rr_rec@N':<9}")
    print("-" * 78)
    for row in rows:
        r = row["retrieval"]
        print(
            f"{row['id'][:31]:<32} "
            f"{r['dense']['mrr']:<8.2f} "
            f"{r['hybrid']['mrr']:<8.2f} "
            f"{r['reranked']['mrr']:<8.2f} "
            f"{r['rerank_delta']['mrr']:<+8.2f} "
            f"{r['reranked'][rkey]:<9.2f}"
        )

    retrieval_agg = _aggregate_retrieval(rows, rkey)
    summary = {
        "label": label,
        "llm_model": config.LLM_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "reranker_model": config.RERANKER_MODEL,
        "n_items": len(rows),
        "retrieval": retrieval_agg,
        "rows": rows,
    }

    if retrieval_only:
        summary["combined_score"] = (
            retrieval_agg["hybrid_mrr"]
            + retrieval_agg["reranked_mrr"]
            + retrieval_agg[f"reranked_{rkey}"]
        ) / 3.0
    else:
        summary["avg_retrieval_recall"] = sum(r["retrieval_recall"] for r in rows) / len(rows)
        summary["term_coverage_rate"] = sum(1 for r in rows if r["term_coverage"]) / len(rows)
        summary["combined_score"] = _combined_score(rows)
        if use_judge:
            summary["avg_faithfulness"] = sum(r["faithfulness"] for r in rows) / len(rows)
            summary["avg_correctness"] = sum(r["correctness"] for r in rows) / len(rows)

    print("-" * 78)
    if not retrieval_only:
        print(f"recall={summary['avg_retrieval_recall']:.2f}  terms={summary['term_coverage_rate']:.0%}", end="")
        if use_judge:
            print(f"  faith={summary['avg_faithfulness']:.2f}  correct={summary['avg_correctness']:.2f}", end="")
        print()
    print(
        f"emb_mrr={retrieval_agg['dense_mrr']:.2f}  hyb_mrr={retrieval_agg['hybrid_mrr']:.2f}  "
        f"rr_mrr={retrieval_agg['reranked_mrr']:.2f}  rr_delta_mrr={retrieval_agg['rerank_delta_mrr']:+.2f}  "
        f"rr_{rkey}={retrieval_agg[f'reranked_{rkey}']:.2f}  combined={summary['combined_score']:.2f}"
    )

    return summary


def save_history(run_record: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default=DEFAULT_EVAL_SET)
    ap.add_argument("--matrix", default=None, help="path to a JSON list of {label, llm_model, embedding_model, reranker_model, router_model} combos (models.yaml registry names)")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM-judge faithfulness/correctness calls")
    ap.add_argument("--retrieval-only", action="store_true", help="only component retrieval metrics; skip run_query()/judge entirely (fast/free)")
    ap.add_argument("--threshold", type=float, default=None, help="exit 1 if any combo's combined_score is below this")
    args = ap.parse_args()

    eval_set = load_json(args.eval_set)
    use_judge = not args.no_judge and not args.retrieval_only

    combos = [{}]  # empty combo == "whatever's already in .env / config.py right now"
    if args.matrix:
        combos = load_json(args.matrix)

    t0 = time.time()
    summaries = []
    for combo in combos:
        if combo:
            _apply_combo(combo)

        if "embedding_model" in combo and not _index_matches(config.embedding_id()):
            print(
                f"\n=== {_combo_label(combo)} ===\n"
                f"[skip] index doesn't match embeddings '{config.embedding_id()}' - "
                f"run `python ingest.py --reset` with this EMBEDDING_MODEL first."
            )
            continue

        summaries.append(run_combo(_combo_label(combo), eval_set, use_judge, args.retrieval_only))

    run_record = {
        "ts": int(time.time() * 1000),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - t0, 1),
        "use_judge": use_judge,
        "retrieval_only": args.retrieval_only,
        "eval_set": os.path.basename(args.eval_set),
        "summaries": [
            {k: v for k, v in s.items() if k != "rows"}  # full per-item rows stay out of history
            for s in summaries
        ],
    }
    save_history(run_record)
    print(f"\nAppended run to {HISTORY_PATH}")

    if args.threshold is not None:
        if not summaries:
            print("\nFAIL: no combos ran (all skipped) - cannot check threshold.")
            sys.exit(1)
        worst = min(s["combined_score"] for s in summaries)
        if worst < args.threshold:
            print(f"\nFAIL: worst combined_score {worst:.2f} < threshold {args.threshold}")
            sys.exit(1)
        print(f"\nPASS: worst combined_score {worst:.2f} >= threshold {args.threshold}")

    sys.exit(0)


if __name__ == "__main__":
    main()
