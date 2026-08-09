"""
Medical Retrieval System — Main Entry Point

Usage:
  python main.py --build     # Build indexes (run once)
  python main.py --query "What is the treatment for Type 2 diabetes?"
  python main.py --eval      # Run full evaluation
  python main.py --demo      # Interactive demo
"""

import argparse
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "NO"
import time
from sentence_transformers import SentenceTransformer, CrossEncoder

from index       import build_all, load_indexes
from retrieval   import HybridRetriever
from evaluator   import build_eval_set, evaluate
from config        import EMBEDDING_MODEL, TOP_K, RERANKER_MODEL


def get_retriever():
    """Load indexes and return retriever."""
    faiss_index, bm25_index, corpus = load_indexes()
    model = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    return HybridRetriever(faiss_index, bm25_index, corpus, model, reranker), corpus


def run_query(query: str):
    retriever, _ = get_retriever()
    results, timing = retriever.retrieve(query, top_k=TOP_K, verbose=True)

    print(f"\nTop {TOP_K} passages for: '{query}'\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] Score: {r['rrf_score']:.4f}")
        print(f"    {r['text'][:300]}...")
        print()


def run_eval():
    retriever, corpus = get_retriever()
    eval_set = build_eval_set(corpus, n_queries=100)
    evaluate(retriever, eval_set, k=TOP_K)


def run_demo():
    """Interactive demo — type queries and get results."""
    retriever, _ = get_retriever()
    print("\n" + "="*50)
    print("  Medical Retrieval System — Interactive Demo")
    print("  Type 'quit' to exit")
    print("="*50 + "\n")

    sample_queries = [
        "What is the recommended treatment for Type 2 diabetes?",
        "What are the risk factors for cardiovascular disease?",
        "How does metformin work in diabetic patients?",
        "What is the prognosis for early-stage lung cancer?",
    ]

    print("Sample queries to try:")
    for i, q in enumerate(sample_queries, 1):
        print(f"  {i}. {q}")
    print()

    while True:
        query = input("Enter query: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        results, timing = retriever.retrieve(query, top_k=TOP_K, verbose=True)
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] RRF Score: {r['rrf_score']:.4f}")
            print(f"    {r['text'][:400]}...")


def print_memory_report():
    """Estimate memory usage of the system."""
    import os
    import pickle
    from config import FAISS_INDEX_PATH, BM25_INDEX_PATH, CORPUS_PATH

    print("\n" + "="*45)
    print("  MEMORY FOOTPRINT ESTIMATE")
    print("="*45)

    sizes = {
        "FAISS IVF_PQ Index": FAISS_INDEX_PATH,
        "BM25 Index":         BM25_INDEX_PATH,
        "Corpus":             CORPUS_PATH,
    }

    total_mb = 0
    for name, path in sizes.items():
        if os.path.exists(path):
            mb = os.path.getsize(path) / (1024 * 1024)
            total_mb += mb
            print(f"  {name:<25} {mb:.1f} MB")

    # Embedding model is ~80MB on disk, ~80MB in RAM
    model_mb = 80
    total_mb += model_mb
    print(f"  {'Embedding Model':<25} ~{model_mb} MB")
    print(f"  {'-'*35}")
    print(f"  {'TOTAL':<25} ~{total_mb:.0f} MB")
    print(f"  {'Target':<25} < 2000 MB")
    print(f"  {'Status':<25} {'OK (Within budget)' if total_mb < 2000 else 'FAIL (Over budget)'}")
    print("="*45 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Retrieval System")
    parser.add_argument("--build",  action="store_true", help="Build indexes")
    parser.add_argument("--query",  type=str,            help="Single query")
    parser.add_argument("--eval",   action="store_true", help="Run evaluation")
    parser.add_argument("--demo",   action="store_true", help="Interactive demo")
    parser.add_argument("--memory", action="store_true", help="Memory report")
    args = parser.parse_args()

    if args.build:
        build_all()
    elif args.query:
        run_query(args.query)
    elif args.eval:
        run_eval()
    elif args.demo:
        run_demo()
    elif args.memory:
        print_memory_report()
    else:
        # Default: build if no index, then demo
        if not os.path.exists("indexes/faiss.index"):
            print("No index found. Building...")
            build_all()
        run_demo()