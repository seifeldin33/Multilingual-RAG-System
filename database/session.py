"""
Database session management.

Provides SQLAlchemy engine, session factory, and table creation utilities.
"""

import logging
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from config.settings import settings
from database.models import Base

logger = logging.getLogger(__name__)

# ── Engine & Session Factory ──────────────────────────────────────────────────

engine = create_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ── Table Management ──────────────────────────────────────────────────────────

def create_tables():
    """Create all tables defined by ORM models (idempotent)."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured at %s", settings.database_url)


def drop_tables():
    """Drop all tables (use with caution)."""
    Base.metadata.drop_all(bind=engine)
    logger.info("All database tables dropped")


# ── Session Context Manager ──────────────────────────────────────────────────

@contextmanager
def get_session() -> Session:
    """Yield a managed database session that auto-commits or rolls back."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
