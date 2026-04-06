"""
Central configuration for the Multilingual RAG System.

Uses pydantic-settings with .env file support for all configurable parameters.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # ── Embedding Model ──────────────────────────────────────────────────
    embedding_model: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        description="HuggingFace Sentence-Transformer model name",
    )
    embedding_dim: int = Field(
        default=384,
        description="Dimensionality of the embedding vectors",
    )

    # ── LLM Configuration ────────────────────────────────────────────────
    llm_provider: str = Field(
        default="ollama", 
        description="Provider for answer generation (groq, ollama, openai, google)"
    )
    llm_model: str = Field(
        default="llama3.2:3b",
        description="Model identifier (e.g., llama-3.3-70b-versatile, gpt-4o, gemini-1.5-flash)",
    )
    llm_temperature: float = Field(default=0.3, description="LLM temperature")
    llm_max_tokens: int = Field(default=1024, description="Max tokens in LLM response")
    
    # API Keys & Endpoints for Providers
    groq_api_key: str = Field(default="", description="Groq API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    google_api_key: str = Field(default="", description="Google Gemini API key")
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", description="Ollama API URL")

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset_path: str = Field(
        default="Natural-Questions-Base.csv",
        description="Path to the source CSV dataset",
    )
    sample_size: int = Field(
        default=10_000,
        description="Number of rows to sample for development (0 = full dataset)",
    )

    # ── Chunking ──────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512, description="Max characters per text chunk")
    chunk_overlap: int = Field(default=50, description="Overlap between adjacent chunks")

    # ── FAISS ─────────────────────────────────────────────────────────────
    index_dir: str = Field(default="index_data", description="Directory for FAISS index files")
    top_k: int = Field(default=5, description="Default number of results to retrieve")
    score_threshold: float = Field(
        default=0.0,
        description="Minimum similarity score to include in results",
    )

    # ── Database ──────────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite:///rag_system.db",
        description="SQLAlchemy database connection string",
    )

    # ── Cache ─────────────────────────────────────────────────────────────
    cache_ttl: int = Field(default=3600, description="Cache time-to-live in seconds")
    cache_max_size: int = Field(default=1000, description="Max entries in LRU cache")

    # ── API ────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")

    # ── Batch Processing ──────────────────────────────────────────────────
    encoding_batch_size: int = Field(
        default=64, description="Batch size for encoding documents"
    )

    model_config = {
        "env_file": str(BASE_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def index_path(self) -> Path:
        """Full path to the FAISS index directory."""
        return BASE_DIR / self.index_dir

    @property
    def dataset_full_path(self) -> Path:
        """Full path to the dataset CSV."""
        path = Path(self.dataset_path)
        if path.is_absolute():
            return path
        return BASE_DIR / path


# Singleton settings instance
settings = Settings()
