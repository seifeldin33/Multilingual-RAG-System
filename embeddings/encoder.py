"""
Sentence-Transformers wrapper for encoding documents and queries.

Handles batch encoding with progress bars, GPU acceleration when available,
and L2 normalisation for cosine similarity via inner product.
"""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """Wraps a Sentence-Transformer model for document / query encoding."""

    def __init__(self, model_name: Optional[str] = None):
        model_name = model_name or settings.embedding_model
        logger.info("Loading embedding model '%s' …", model_name)
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(
            "Model loaded — dimension=%d, device=%s",
            self.dim,
            self.model.device,
        )

    # ── Encoding helpers ──────────────────────────────────────────────────

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = settings.encoding_batch_size,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of document texts into an (N, D) numpy array.

        Parameters
        ----------
        texts : list[str]
            The texts to encode.
        batch_size : int
            Number of texts to encode per batch.
        show_progress : bool
            Whether to display a tqdm progress bar.
        normalize : bool
            If True, L2-normalise each vector (needed for cosine similarity
            via inner product in FAISS).

        Returns
        -------
        np.ndarray of shape (len(texts), self.dim)
        """
        logger.info("Encoding %d documents (batch_size=%d) …", len(texts), batch_size)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

        logger.info("Encoding complete — shape %s", embeddings.shape)
        return embeddings

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single query string into a 1-D vector of shape (D,).
        """
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embedding[0]

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode multiple queries at once."""
        return self.model.encode(
            queries,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
