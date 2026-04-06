"""
Evaluation metrics for the RAG system.

Provides retrieval metrics (Precision@K, Recall@K, MRR), generation
metrics (BLEU, ROUGE), and performance benchmarks.
"""

import logging
import random
import time
from typing import Optional

import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from config.settings import settings
from retrieval.retriever import RAGRetriever

logger = logging.getLogger(__name__)


# ── Retrieval Metrics ─────────────────────────────────────────────────────────

def precision_at_k(relevant: list[bool], k: int) -> float:
    """Fraction of top-K results that are relevant."""
    top = relevant[:k]
    return sum(top) / k if k > 0 else 0.0


def recall_at_k(relevant: list[bool], k: int, total_relevant: int) -> float:
    """Fraction of all relevant documents found in top-K."""
    top = relevant[:k]
    return sum(top) / total_relevant if total_relevant > 0 else 0.0


def mean_reciprocal_rank(relevant: list[bool]) -> float:
    """Reciprocal of the rank of the first relevant result."""
    for i, r in enumerate(relevant, start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevance_scores: list[float], k: int) -> float:
    """Normalised Discounted Cumulative Gain at K."""
    dcg = sum(
        rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores[:k])
    )
    ideal = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ── Generation Metrics ────────────────────────────────────────────────────────

_bleu_smoother = SmoothingFunction().method1


def compute_bleu(reference: str, hypothesis: str) -> float:
    """BLEU score (0–1) between a reference and hypothesis answer."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    return sentence_bleu(
        [ref_tokens], hyp_tokens, smoothing_function=_bleu_smoother
    )


def compute_rouge(reference: str, hypothesis: str) -> dict[str, float]:
    """ROUGE-1, ROUGE-2, ROUGE-L F-scores."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


# ── Self-Retrieval Evaluation ────────────────────────────────────────────────

def run_retrieval_evaluation(
    retriever: RAGRetriever,
    num_samples: int = 100,
    top_k: int = 5,
) -> list[dict]:
    """
    Evaluate retrieval quality using self-retrieval:
    pick indexed questions, retrieve, and check if the source document
    appears in the top-K results.

    Returns a list of metric dicts suitable for the API response.
    """
    # Sample documents from the index
    docs = retriever.store.documents
    if len(docs) < num_samples:
        num_samples = len(docs)

    sample_indices = random.sample(range(len(docs)), num_samples)
    sample_docs = [docs[i] for i in sample_indices]

    precisions = []
    recalls = []
    mrrs = []
    latencies = []

    for doc in sample_docs:
        t0 = time.perf_counter()
        results = retriever.retrieve(doc.question, top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        # A result is "relevant" if it comes from the same original question
        relevant = [r.document.question == doc.question for r in results]
        total_relevant = sum(
            1 for d in docs if d.question == doc.question
        )

        precisions.append(precision_at_k(relevant, top_k))
        recalls.append(recall_at_k(relevant, top_k, total_relevant))
        mrrs.append(mean_reciprocal_rank(relevant))

    metrics = [
        {
            "name": f"precision@{top_k}",
            "value": round(float(np.mean(precisions)), 4),
            "description": f"Average precision at {top_k}",
        },
        {
            "name": f"recall@{top_k}",
            "value": round(float(np.mean(recalls)), 4),
            "description": f"Average recall at {top_k}",
        },
        {
            "name": "mrr",
            "value": round(float(np.mean(mrrs)), 4),
            "description": "Mean Reciprocal Rank",
        },
        {
            "name": "avg_latency_ms",
            "value": round(float(np.mean(latencies)), 2),
            "description": "Average retrieval latency (ms)",
        },
        {
            "name": "p95_latency_ms",
            "value": round(float(np.percentile(latencies, 95)), 2),
            "description": "95th percentile retrieval latency (ms)",
        },
        {
            "name": "throughput_qps",
            "value": round(1000.0 / max(float(np.mean(latencies)), 0.01), 2),
            "description": "Estimated queries per second",
        },
    ]

    logger.info(
        "Evaluation complete (%d samples): P@%d=%.3f, MRR=%.3f, avg_latency=%.1fms",
        num_samples,
        top_k,
        np.mean(precisions),
        np.mean(mrrs),
        np.mean(latencies),
    )
    return metrics
