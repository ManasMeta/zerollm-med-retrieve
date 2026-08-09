"""
Evaluator — measures retrieval quality.

Metrics:
  Recall@K  — did the relevant doc appear in top-K results?
  MRR       — Mean Reciprocal Rank (how high was the first hit?)
  NDCG@K    — Normalized Discounted Cumulative Gain (rank-weighted quality)

Eval set: generated from PubMedQA ground truth QA pairs.
LLM used OFFLINE only — not in hot path.
"""

import numpy as np
from tqdm import tqdm
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOP_K


def recall_at_k(retrieved_ids, relevant_ids, k):
    """
    Recall@K = # relevant docs in top-K / total relevant docs
    For single-answer QA: 1 if answer found in top-K, else 0.
    """
    top_k_ids = set(retrieved_ids[:k])
    relevant   = set(relevant_ids)
    hits = len(top_k_ids & relevant)
    return hits / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved_ids, relevant_ids):
    """
    RR = 1 / rank_of_first_relevant_doc
    MRR = mean of RR across all queries.
    """
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    """
    NDCG@K — rewards finding relevant docs higher in the list.
    Binary relevance: 1 if relevant, 0 if not.
    """
    relevant = set(relevant_ids)

    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / np.log2(rank + 1)

    # Ideal DCG — all relevant docs at top
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def build_eval_set(corpus, n_queries=100):
    """
    Build evaluation set from PubMedQA ground truth.

    PubMedQA has real medical questions with known relevant passages.
    We use this as ground truth WITHOUT calling any LLM.

    For a production system, you'd use an offline LLM to generate
    synthetic QA pairs over your own corpus.
    """
    eval_set = []

    for i, doc in enumerate(corpus[:n_queries]):
        if doc.get("question") and doc.get("text"):
            eval_set.append({
                "query":       doc["question"],
                "relevant_id": str(doc["id"]),
                "doc_idx":     i
            })

    print(f"Built eval set with {len(eval_set)} queries from PubMedQA ground truth")
    return eval_set


def evaluate(retriever, eval_set, k=TOP_K):
    """
    Run full evaluation across all queries.
    Reports Recall@K, MRR, NDCG@K and latency stats.
    """
    recall_scores = []
    mrr_scores    = []
    ndcg_scores   = []
    latencies     = []

    print(f"\nEvaluating {len(eval_set)} queries (K={k})...")

    for item in tqdm(eval_set):
        query        = item["query"]
        relevant_id  = item["relevant_id"]
        relevant_ids = [relevant_id]

        results, timing = retriever.retrieve(query, top_k=k)
        retrieved_ids = [str(r["id"]) for r in results]

        recall_scores.append(recall_at_k(retrieved_ids, relevant_ids, k))
        mrr_scores.append(reciprocal_rank(retrieved_ids, relevant_ids))
        ndcg_scores.append(ndcg_at_k(retrieved_ids, relevant_ids, k))
        latencies.append(timing["total_ms"])

    results_summary = {
        f"Recall@{k}":        round(np.mean(recall_scores), 4),
        "MRR":                round(np.mean(mrr_scores), 4),
        f"NDCG@{k}":          round(np.mean(ndcg_scores), 4),
        "avg_latency_ms":     round(np.mean(latencies), 2),
        "p50_latency_ms":     round(np.percentile(latencies, 50), 2),
        "p95_latency_ms":     round(np.percentile(latencies, 95), 2),
        "p99_latency_ms":     round(np.percentile(latencies, 99), 2),
        "pct_under_100ms":    round(np.mean([l < 100 for l in latencies]) * 100, 1),
    }

    print("\n" + "="*45)
    print("  EVALUATION RESULTS")
    print("="*45)
    for metric, value in results_summary.items():
        unit = "%" if "pct" in metric else ("ms" if "ms" in metric else "")
        print(f"  {metric:<25} {value}{unit}")
    print("="*45)

    return results_summary