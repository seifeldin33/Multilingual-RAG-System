"""
FAISS vector store management.

Build, save, load, and search a FAISS index. Document metadata is stored
as a parallel pickle file alongside the index.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from config.settings import settings
from data.processing import Document

logger = logging.getLogger(__name__)


class FAISSStore:
    """Manages a FAISS index and its associated document metadata."""

    INDEX_FILENAME = "index.faiss"
    META_FILENAME = "documents.pkl"

    def __init__(self, dimension: int = settings.embedding_dim):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.documents: list[Document] = []

    # ── Build ─────────────────────────────────────────────────────────────

    def build_index(
        self,
        embeddings: np.ndarray,
        documents: list[Document],
        use_ivf: bool = False,
        nlist: int = 100,
    ) -> None:
        """
        Build a FAISS index from pre-computed embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            (N, D) array of L2-normalised embeddings.
        documents : list[Document]
            Parallel list of Document objects.
        use_ivf : bool
            If True, use IndexIVFFlat for faster approximate search on
            large datasets (requires training).
        nlist : int
            Number of Voronoi cells for IVF index.
        """
        assert len(embeddings) == len(documents), (
            f"Mismatch: {len(embeddings)} embeddings vs {len(documents)} documents"
        )
        self.dimension = embeddings.shape[1]
        n = len(embeddings)
        logger.info("Building FAISS index: %d vectors, dim=%d", n, self.dimension)

        embeddings = embeddings.astype(np.float32)

        if use_ivf and n > nlist * 40:
            # IVF for large datasets
            quantiser = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantiser, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            logger.info("Training IVF index (nlist=%d) …", nlist)
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = min(10, nlist)
            logger.info("IVF index built (nprobe=%d)", self.index.nprobe)
        else:
            # Flat index — exact search
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings)

        self.documents = documents
        logger.info("Index built with %d vectors", self.index.ntotal)

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, directory: Optional[str] = None) -> Path:
        """Persist the index and metadata to disk."""
        dir_path = Path(directory) if directory else settings.index_path
        dir_path.mkdir(parents=True, exist_ok=True)

        index_path = dir_path / self.INDEX_FILENAME
        meta_path = dir_path / self.META_FILENAME

        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump(self.documents, f)

        logger.info("Index saved → %s (%d vectors)", dir_path, self.index.ntotal)
        return dir_path

    def load(self, directory: Optional[str] = None) -> None:
        """Load a previously saved index and metadata."""
        dir_path = Path(directory) if directory else settings.index_path

        index_path = dir_path / self.INDEX_FILENAME
        meta_path = dir_path / self.META_FILENAME

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Document metadata not found: {meta_path}")

        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            self.documents = pickle.load(f)

        self.dimension = self.index.d
        logger.info(
            "Index loaded ← %s (%d vectors, %d documents)",
            dir_path,
            self.index.ntotal,
            len(self.documents),
        )

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = settings.top_k,
    ) -> list[tuple[Document, float]]:
        """
        Search the index and return the top-K most similar documents.

        Parameters
        ----------
        query_embedding : np.ndarray
            1-D vector of shape (D,).
        top_k : int
            Number of results to return.

        Returns
        -------
        list of (Document, score) tuples, sorted by descending similarity.
        """
        if self.index is None:
            raise RuntimeError("Index not built or loaded — call build_index() or load() first")

        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_vec, top_k)

        results: list[tuple[Document, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing entries
                continue
            results.append((self.documents[idx], float(score)))

        return results

    # ── Info ──────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal if self.index else 0

    def __repr__(self) -> str:
        return f"FAISSStore(dim={self.dimension}, size={self.size})"
