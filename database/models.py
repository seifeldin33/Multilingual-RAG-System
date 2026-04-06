"""
SQLAlchemy ORM models for query logging, conversations, and performance metrics.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    JSON,
    Boolean,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


class QueryLog(Base):
    """Log of every query and its response."""

    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    processed_query = Column(Text)
    answer = Column(Text)
    sources = Column(JSON)
    confidence = Column(Float, default=0.0)
    latency_ms = Column(Float, default=0.0)
    question_type = Column(String(50))
    is_fallback = Column(Boolean, default=False)
    model = Column(String(100))
    conversation_id = Column(String(100), index=True)
    timestamp = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )


class ConversationContext(Base):
    """Stores multi-turn conversation state."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(100), unique=True, nullable=False, index=True)
    messages = Column(JSON, default=list)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class PerformanceMetric(Base):
    """Time-series performance metrics."""

    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False, index=True)
    value = Column(Float, nullable=False)
    metadata_ = Column("metadata", JSON, default=dict)
    timestamp = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )
