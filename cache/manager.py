"""
Intelligent caching for the RAG system.

Provides LRU-based caching for query results and embedding vectors,
with TTL expiry and hit/miss statistics.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from cachetools import TTLCache

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Track cache hit/miss statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / max(self.total, 1)

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "total": self.total,
            "hit_rate": round(self.hit_rate, 4),
        }


class CacheManager:
    """
    Manages separate caches for query responses and embedding vectors.

    Both caches use LRU eviction with configurable TTL.
    """

    def __init__(
        self,
        max_size: int = settings.cache_max_size,
        ttl: int = settings.cache_ttl,
    ):
        self._query_cache: TTLCache = TTLCache(maxsize=max_size, ttl=ttl)
        self._embedding_cache: TTLCache = TTLCache(maxsize=max_size * 2, ttl=ttl)

        self.query_stats = CacheStats()
        self.embedding_stats = CacheStats()

        logger.info(
            "CacheManager initialised (max_size=%d, ttl=%ds)", max_size, ttl
        )

    # ── Key Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _make_key(text: str) -> str:
        """Create a stable cache key from text."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    # ── Query Cache ───────────────────────────────────────────────────────

    def get_query_result(self, query: str) -> Optional[Any]:
        """Look up a cached query result."""
        key = self._make_key(query)
        result = self._query_cache.get(key)
        if result is not None:
            self.query_stats.hits += 1
            logger.debug("Query cache HIT: %s", query[:50])
            return result
        self.query_stats.misses += 1
        return None

    def set_query_result(self, query: str, result: Any) -> None:
        """Store a query result in the cache."""
        key = self._make_key(query)
        self._query_cache[key] = result

    # ── Embedding Cache ───────────────────────────────────────────────────

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Look up a cached embedding vector."""
        key = self._make_key(text)
        result = self._embedding_cache.get(key)
        if result is not None:
            self.embedding_stats.hits += 1
            return result
        self.embedding_stats.misses += 1
        return None

    def set_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Store an embedding vector in the cache."""
        key = self._make_key(text)
        self._embedding_cache[key] = embedding

    # ── Management ────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Clear all caches and reset stats."""
        self._query_cache.clear()
        self._embedding_cache.clear()
        self.query_stats = CacheStats()
        self.embedding_stats = CacheStats()
        logger.info("All caches cleared")

    def get_stats(self) -> dict:
        """Return combined cache statistics."""
        return {
            "query_cache": self.query_stats.to_dict(),
            "embedding_cache": self.embedding_stats.to_dict(),
            "query_cache_size": len(self._query_cache),
            "embedding_cache_size": len(self._embedding_cache),
        }
