"""
Multilingual RAG System — Application Entry Point

Usage:
    python main.py build        Build the FAISS index from the dataset
    python main.py query        Interactive query mode (CLI)
    python main.py explore      Run data exploration and generate plots
    python main.py serve        Start the FastAPI server
    python main.py evaluate     Run evaluation metrics
"""

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-25s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def cmd_build(args):
    """Build the FAISS index from the dataset."""
    from config.settings import settings
    from data.processing import process_dataset
    from retrieval.retriever import RAGRetriever

    logger.info("═" * 60)
    logger.info("  Building RAG Index")
    logger.info("═" * 60)

    sample = args.sample if args.sample else settings.sample_size
    logger.info("Sample size: %s", sample if sample else "FULL DATASET")

    t0 = time.perf_counter()

    # 1. Process dataset
    documents = process_dataset(sample_size=sample)
    logger.info("Processed %d document chunks", len(documents))

    # 2. Build index
    retriever = RAGRetriever()
    use_ivf = len(documents) > 50_000
    retriever.build_index(documents, use_ivf=use_ivf)

    # 3. Save to disk
    retriever.save_index()
    elapsed = time.perf_counter() - t0

    logger.info("═" * 60)
    logger.info("  Index built in %.1f seconds", elapsed)
    logger.info("  %d vectors, dimension=%d", retriever.index_size, retriever.encoder.dim)
    logger.info("  Saved to: %s", settings.index_path)
    logger.info("═" * 60)


def cmd_query(args):
    """Interactive CLI query loop."""
    from retrieval.retriever import RAGRetriever
    from query.processor import QueryProcessor
    from generation.llm_client import GroqLLMClient
    from cache.manager import CacheManager

    logger.info("Loading index …")
    retriever = RAGRetriever()
    retriever.load_index()
    logger.info("Index loaded — %d vectors", retriever.index_size)

    qp = QueryProcessor()
    llm = GroqLLMClient()
    cache = CacheManager()

    print("\n" + "=" * 60)
    print("  Multilingual RAG — Interactive Query")
    print("  Type 'quit' to exit, 'clear' to reset history")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("❓ Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if query.lower() == "clear":
            qp.clear_history()
            cache.clear()
            print("🗑  History and cache cleared.\n")
            continue

        # Check cache
        cached = cache.get_query_result(query)
        if cached:
            print(f"\n💾 [Cached] {cached.answer}\n")
            continue

        # Process
        processed = qp.process(query)
        print(f"   → type: {processed['question_type']}, final query: {processed['final'][:80]}…")

        # Retrieve
        t0 = time.perf_counter()
        results = retriever.retrieve(processed["final"], top_k=args.top_k)

        if not results:
            print("   ⚠ No relevant results found.\n")
            continue

        # Generate
        gen_result = llm.generate(query=query, retrieval_results=results)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Display
        print(f"\n{'─' * 60}")
        print(f"📝 Answer (confidence: {gen_result.confidence:.2f}, {elapsed_ms:.0f}ms):")
        print(f"{'─' * 60}")
        print(gen_result.answer)
        print(f"{'─' * 60}")
        print(f"📚 Sources ({len(gen_result.sources)}):")
        for s in gen_result.sources[:3]:
            print(f"   [{s['rank']}] score={s['score']:.3f} — {s['question'][:60]}")
        print()

        # Update state
        qp.add_to_history("user", query)
        qp.add_to_history("assistant", gen_result.answer)
        cache.set_query_result(query, gen_result)


def cmd_explore(args):
    """Run data exploration."""
    from notebooks.exploration import explore_dataset
    explore_dataset(sample_size=args.sample or 0)


def cmd_serve(args):
    """Start the FastAPI server."""
    import uvicorn
    from config.settings import settings

    logger.info("Starting API server on %s:%d …", settings.api_host, settings.api_port)
    uvicorn.run(
        "api.app:app",
        host=settings.api_host,
        port=args.port or settings.api_port,
        reload=args.reload,
    )


def cmd_evaluate(args):
    """Run evaluation metrics."""
    from retrieval.retriever import RAGRetriever
    from evaluation.metrics import run_retrieval_evaluation

    retriever = RAGRetriever()
    retriever.load_index()

    metrics = run_retrieval_evaluation(
        retriever=retriever,
        num_samples=args.num_samples,
        top_k=args.top_k,
    )

    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    for m in metrics:
        print(f"  {m['name']:<20} {m['value']:>10}")
    print("=" * 60 + "\n")


# ── CLI Argument Parsing ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multilingual RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # build
    p_build = subparsers.add_parser("build", help="Build the FAISS index")
    p_build.add_argument("--sample", type=int, default=None, help="Number of rows to sample")
    p_build.set_defaults(func=cmd_build)

    # query
    p_query = subparsers.add_parser("query", help="Interactive query mode")
    p_query.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve")
    p_query.set_defaults(func=cmd_query)

    # explore
    p_explore = subparsers.add_parser("explore", help="Run data exploration")
    p_explore.add_argument("--sample", type=int, default=None, help="Number of rows to sample")
    p_explore.set_defaults(func=cmd_explore)

    # serve
    p_serve = subparsers.add_parser("serve", help="Start the FastAPI API server")
    p_serve.add_argument("--port", type=int, default=None, help="Port override")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload")
    p_serve.set_defaults(func=cmd_serve)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run evaluation metrics")
    p_eval.add_argument("--num-samples", type=int, default=100, help="Number of test samples")
    p_eval.add_argument("--top-k", type=int, default=5, help="Top-K for retrieval")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()