"""
FastAPI application setup with CORS, lifespan events, and middleware.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from cache.manager import CacheManager
from embeddings.encoder import EmbeddingEncoder
from generation.llm_client import BaseLLMClient, get_llm_client
from query.processor import QueryProcessor
from retrieval.retriever import RAGRetriever
from vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)

# ── Global State ──────────────────────────────────────────────────────────────
 
class AppState:
    """Holds runtime components shared across requests."""

    retriever: RAGRetriever = None
    llm_client: BaseLLMClient = None
    query_processor: QueryProcessor = None
    cache_manager: CacheManager = None
    start_time: float = 0.0


state = AppState()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and index on startup, clean up on shutdown."""
    logger.info("Starting up RAG API …")
    state.start_time = time.time()

    # Initialise components
    state.cache_manager = CacheManager()
    state.query_processor = QueryProcessor()
    state.llm_client = get_llm_client()

    # Load encoder + FAISS index
    try:
        encoder = EmbeddingEncoder()
        store = FAISSStore(dimension=encoder.dim)
        store.load()
        state.retriever = RAGRetriever(encoder=encoder, store=store)
        logger.info("Index loaded — %d vectors ready", state.retriever.index_size)
    except FileNotFoundError:
        logger.warning(
            "No FAISS index found at '%s'. Run `python main.py build` first.",
            settings.index_path,
        )
        # Create a retriever without an index so the app still starts
        state.retriever = RAGRetriever(encoder=EmbeddingEncoder())

    yield  

    logger.info("Shutting down RAG API …")
    state.cache_manager.clear()


# ── App Factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Multilingual RAG System",
        description="Production-quality Retrieval-Augmented Generation API built on Natural Questions",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timing middleware
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - t0) * 1000
        response.headers["X-Response-Time-Ms"] = f"{elapsed:.1f}"
        return response

    # Register routes
    from api.routes import router
    app.include_router(router)

    return app


app = create_app()
