"""
RAG Retriever — combines the embedding encoder and FAISS vector store
into a single retrieval interface.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from config.settings import settings
from data.processing import Document
from embeddings.encoder import EmbeddingEncoder
from vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval hit with score and rank."""

    document: Document
    score: float
    rank: int

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(rank={self.rank}, score={self.score:.4f}, "
            f"q='{self.document.question[:40]}…')"
        )


class RAGRetriever:
    """
    High-level retriever that wraps encoder + vector store.

    Usage
    -----
    >>> retriever = RAGRetriever()
    >>> retriever.load_index()
    >>> results = retriever.retrieve("What is photosynthesis?")
    """

    def __init__(
        self,
        encoder: Optional[EmbeddingEncoder] = None,
        store: Optional[FAISSStore] = None,
    ):
        self.encoder = encoder or EmbeddingEncoder()
        self.store = store or FAISSStore(dimension=self.encoder.dim)

    # ── Index lifecycle ───────────────────────────────────────────────────

    def build_index(
        self,
        documents: list[Document],
        use_ivf: bool = False,
    ) -> None:
        """Encode all documents and build a FAISS index."""
        texts = [doc.chunk_text for doc in documents]
        embeddings = self.encoder.encode_documents(texts)
        self.store.build_index(embeddings, documents, use_ivf=use_ivf)

    def save_index(self, directory: Optional[str] = None):
        """Persist the current index to disk."""
        self.store.save(directory)

    def load_index(self, directory: Optional[str] = None):
        """Load a previously saved index."""
        self.store.load(directory)

    # ── Retrieval ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = settings.top_k,
        score_threshold: float = settings.score_threshold,
    ) -> list[RetrievalResult]:
        """
        Retrieve the top-K documents most relevant to *query*.

        Parameters
        ----------
        query : str
            The user's question.
        top_k : int
            Maximum number of results.
        score_threshold : float
            Minimum similarity score (0–1 after normalisation).

        Returns
        -------
        list[RetrievalResult]  sorted by descending score.
        """
        t0 = time.perf_counter()

        query_embedding = self.encoder.encode_query(query)
        raw_results = self.store.search(query_embedding, top_k=top_k)

        results: list[RetrievalResult] = []
        for rank, (doc, score) in enumerate(raw_results, start=1):
            if score < score_threshold:
                continue
            results.append(RetrievalResult(document=doc, score=score, rank=rank))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Retrieved %d results in %.1f ms (query: '%s…')",
            len(results),
            elapsed_ms,
            query[:50],
        )
        return results

    def retrieve_batch(
        self,
        queries: list[str],
        top_k: int = settings.top_k,
    ) -> list[list[RetrievalResult]]:
        """Retrieve results for multiple queries."""
        return [self.retrieve(q, top_k=top_k) for q in queries]

    # ── Info ──────────────────────────────────────────────────────────────

    @property
    def index_size(self) -> int:
        return self.store.size

    def __repr__(self) -> str:
        return f"RAGRetriever(model='{self.encoder.model}', index_size={self.index_size})"
