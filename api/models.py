"""
Pydantic request/response schemas for the RAG API.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Request Models ────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    """Request body for the /ask-question endpoint."""

    question: str = Field(..., min_length=1, max_length=1000, description="The question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of passages to retrieve")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for multi-turn")
    expand_query: bool = Field(default=True, description="Whether to expand query with synonyms")
    use_llm: bool = Field(default=True, description="Whether to use LLM for generation")


class EvaluationRequest(BaseModel):
    """Request body for the /evaluate endpoint."""

    num_samples: int = Field(default=100, ge=10, le=1000, description="Number of test samples")
    top_k: int = Field(default=5, ge=1, le=20, description="Top-K for retrieval")


# ── Response Models ───────────────────────────────────────────────────────────

class SourceInfo(BaseModel):
    """Information about a retrieved source."""

    rank: int
    score: float
    question: str
    answer_preview: str
    metadata: dict


class QuestionResponse(BaseModel):
    """Response body for the /ask-question endpoint."""

    answer: str
    sources: list[SourceInfo] = []
    confidence: float = 0.0
    latency_ms: float = 0.0
    model: str = ""
    is_fallback: bool = False
    question_type: str = ""
    conversation_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str = "healthy"
    index_loaded: bool = False
    index_size: int = 0
    model_loaded: bool = False
    embedding_model: str = ""
    cache_stats: dict = {}
    uptime_seconds: float = 0.0
    timestamp: str = ""


class MetricResult(BaseModel):
    """A single metric value."""

    name: str
    value: float
    description: str = ""


class EvaluationResponse(BaseModel):
    """Response body for the /evaluate endpoint."""

    metrics: list[MetricResult] = []
    num_samples: int = 0
    total_time_seconds: float = 0.0
    timestamp: str = ""
