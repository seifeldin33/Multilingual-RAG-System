"""
Integration tests for the RAG pipeline.
"""

import os
import sys
import tempfile
import pytest
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Data Processing Tests ─────────────────────────────────────────────────────

class TestHTMLStripping:
    def test_basic_html(self):
        from data.processing import strip_html
        assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_empty_input(self):
        from data.processing import strip_html
        assert strip_html("") == ""
        assert strip_html(None) == ""

    def test_nested_tags(self):
        from data.processing import strip_html
        html = "<div><ul><li>Item 1</li><li>Item 2</li></ul></div>"
        text = strip_html(html)
        assert "Item 1" in text
        assert "Item 2" in text
        assert "<" not in text


class TestChunking:
    def test_short_text_no_split(self):
        from data.processing import chunk_text
        text = "Short text."
        chunks = chunk_text(text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_splits(self):
        from data.processing import chunk_text
        text = "Word " * 200  # ~1000 chars
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 110  # allow small overflow at word boundaries

    def test_empty_text(self):
        from data.processing import chunk_text
        assert chunk_text("") == []
        assert chunk_text(None, chunk_size=100) == []


class TestQuestionClassification:
    def test_who(self):
        from data.processing import classify_question_type
        assert classify_question_type("Who was the first president?") == "person"

    def test_when(self):
        from data.processing import classify_question_type
        assert classify_question_type("When did WWII end?") == "temporal"

    def test_how_many(self):
        from data.processing import classify_question_type
        assert classify_question_type("How many states are there?") == "numeric"

    def test_yes_no(self):
        from data.processing import classify_question_type
        assert classify_question_type("Is the earth round?") == "yes_no"


class TestDomainClassification:
    def test_science(self):
        from data.processing import classify_domain
        text = "The atom has protons and neutrons in its nucleus. DNA carries genetic information."
        assert classify_domain(text) == "science"

    def test_sports(self):
        from data.processing import classify_domain
        text = "The team won the championship match this season after the tournament."
        assert classify_domain(text) == "sports"


# ── Embedding Tests ───────────────────────────────────────────────────────────

class TestEmbeddingEncoder:
    @pytest.fixture(scope="class")
    def encoder(self):
        from embeddings.encoder import EmbeddingEncoder
        return EmbeddingEncoder()

    def test_dimension(self, encoder):
        assert encoder.dim == 384

    def test_single_query(self, encoder):
        vec = encoder.encode_query("What is photosynthesis?")
        assert vec.shape == (384,)
        # Should be normalised (L2 norm ≈ 1)
        assert abs(np.linalg.norm(vec) - 1.0) < 0.01

    def test_batch_documents(self, encoder):
        texts = ["Hello world", "Goodbye world", "Testing embeddings"]
        vecs = encoder.encode_documents(texts, show_progress=False)
        assert vecs.shape == (3, 384)


# ── FAISS Store Tests ─────────────────────────────────────────────────────────

class TestFAISSStore:
    def test_build_and_search(self):
        from vectorstore.faiss_store import FAISSStore
        from data.processing import Document

        dim = 8
        n = 20
        store = FAISSStore(dimension=dim)

        # Create fake data
        embeddings = np.random.randn(n, dim).astype(np.float32)
        # Normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        docs = [
            Document(id=i, question=f"q{i}", answer_text=f"a{i}",
                     chunk_text=f"chunk{i}", chunk_index=0, total_chunks=1)
            for i in range(n)
        ]

        store.build_index(embeddings, docs)
        assert store.size == n

        # Search with the first embedding — should return itself as top-1
        results = store.search(embeddings[0], top_k=3)
        assert len(results) == 3
        assert results[0][0].id == 0  # top result is the same vector

    def test_save_and_load(self, tmp_path):
        from vectorstore.faiss_store import FAISSStore
        from data.processing import Document

        dim = 4
        store = FAISSStore(dimension=dim)
        emb = np.array([[1, 0, 0, 0]], dtype=np.float32)
        docs = [Document(id=0, question="q", answer_text="a",
                         chunk_text="c", chunk_index=0, total_chunks=1)]
        store.build_index(emb, docs)
        store.save(str(tmp_path))

        store2 = FAISSStore(dimension=dim)
        store2.load(str(tmp_path))
        assert store2.size == 1


# ── Query Processor Tests ────────────────────────────────────────────────────

class TestQueryProcessor:
    def test_normalize(self):
        from query.processor import QueryProcessor
        qp = QueryProcessor()
        assert qp.normalize("  What IS this?  ") == "what is this"

    def test_process(self):
        from query.processor import QueryProcessor
        qp = QueryProcessor()
        result = qp.process("What is photosynthesis?", expand=False)
        assert result["question_type"] == "factoid"
        assert result["normalized"] == "what is photosynthesis"


# ── Cache Tests ───────────────────────────────────────────────────────────────

class TestCacheManager:
    def test_query_cache(self):
        from cache.manager import CacheManager
        cm = CacheManager(max_size=10, ttl=60)

        assert cm.get_query_result("test") is None
        cm.set_query_result("test", {"answer": "hello"})
        assert cm.get_query_result("test") == {"answer": "hello"}

        stats = cm.get_stats()
        assert stats["query_cache"]["hits"] == 1
        assert stats["query_cache"]["misses"] == 1

    def test_embedding_cache(self):
        from cache.manager import CacheManager
        cm = CacheManager(max_size=10, ttl=60)

        vec = np.array([1.0, 2.0, 3.0])
        cm.set_embedding("hello", vec)
        result = cm.get_embedding("hello")
        assert np.array_equal(result, vec)
