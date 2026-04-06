"""
API route definitions for the RAG system.

Endpoints:
    POST /ask-question  — Main Q&A endpoint
    GET  /health        — System health check
    POST /evaluate      — Run evaluation on a test set
"""

import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from api.models import (
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    MetricResult,
    QuestionRequest,
    QuestionResponse,
    SourceInfo,
)
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_state():
    """Import state lazily to avoid circular imports."""
    from api.app import state
    return state


# ── POST /ask-question ────────────────────────────────────────────────────────

@router.post("/ask-question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Answer a question using the RAG pipeline.

    1. Preprocess the query (normalise, expand)
    2. Retrieve relevant passages from FAISS
    3. Generate an answer using Groq LLM (or fallback)
    4. Return the answer with sources and metadata
    """
    state = _get_state()
    t0 = time.perf_counter()

    if state.retriever is None or state.retriever.index_size == 0:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Run `python main.py build` first.",
        )

    # Check cache first
    cached = state.cache_manager.get_query_result(request.question)
    if cached:
        return cached

    # Process query
    processed = state.query_processor.process(
        request.question,
        expand=request.expand_query,
    )

    # Retrieve
    results = state.retriever.retrieve(
        processed["final"],
        top_k=request.top_k,
    )

    if not results:
        return QuestionResponse(
            answer="I couldn't find any relevant information for your question.",
            confidence=0.0,
            question_type=processed["question_type"],
        )

    # Generate answer
    conversation_id = request.conversation_id or str(uuid.uuid4())

    if request.use_llm:
        gen_result = state.llm_client.generate(
            query=request.question,
            retrieval_results=results,
            conversation_history=state.query_processor.history if request.conversation_id else None,
        )
    else:
        # Skip LLM, return top passage
        from generation.llm_client import BaseLLMClient
        gen_result = BaseLLMClient._fallback_answer(
            request.question,
            results,
            [
                {
                    "rank": r.rank,
                    "score": round(r.score, 4),
                    "question": r.document.question,
                    "answer_preview": r.document.answer_text[:200],
                    "metadata": r.document.metadata,
                }
                for r in results
            ],
        )

    total_ms = (time.perf_counter() - t0) * 1000

    # Update conversation history
    state.query_processor.add_to_history("user", request.question)
    state.query_processor.add_to_history("assistant", gen_result.answer)

    response = QuestionResponse(
        answer=gen_result.answer,
        sources=[SourceInfo(**s) for s in gen_result.sources],
        confidence=gen_result.confidence,
        latency_ms=total_ms,
        model=gen_result.model,
        is_fallback=gen_result.is_fallback,
        question_type=processed["question_type"],
        conversation_id=conversation_id,
    )

    # Cache the response
    state.cache_manager.set_query_result(request.question, response)

    logger.info("Answered in %.0f ms (confidence=%.2f)", total_ms, gen_result.confidence)
    return response


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Return system health and component status."""
    state = _get_state()

    index_loaded = state.retriever is not None and state.retriever.index_size > 0
    model_loaded = state.retriever is not None and state.retriever.encoder is not None

    return HealthResponse(
        status="healthy" if index_loaded else "degraded",
        index_loaded=index_loaded,
        index_size=state.retriever.index_size if state.retriever else 0,
        model_loaded=model_loaded,
        embedding_model=settings.embedding_model,
        cache_stats=state.cache_manager.get_stats() if state.cache_manager else {},
        uptime_seconds=time.time() - state.start_time,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ── POST /evaluate ────────────────────────────────────────────────────────────

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    """Run evaluation metrics on a random sample from the index."""
    state = _get_state()

    if state.retriever is None or state.retriever.index_size == 0:
        raise HTTPException(status_code=503, detail="Index not loaded.")

    t0 = time.perf_counter()

    try:
        from evaluation.metrics import run_retrieval_evaluation

        metrics = run_retrieval_evaluation(
            retriever=state.retriever,
            num_samples=request.num_samples,
            top_k=request.top_k,
        )
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")

    total_seconds = time.perf_counter() - t0

    return EvaluationResponse(
        metrics=[MetricResult(**m) for m in metrics],
        num_samples=request.num_samples,
        total_time_seconds=round(total_seconds, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
