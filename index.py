"""
Indexer — builds FAISS IVF_PQ + BM25 indexes offline.
Run once; reuse at query time.
"""

import os
import pickle
import time
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import faiss
# pyrefly: ignore [missing-import]
from rank_bm25 import BM25Okapi
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
# pyrefly: ignore [missing-import]
from datasets import load_dataset
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    if len(words) <= chunk_words:
        return [text]
    chunks = []
    for i in range(0, len(words), chunk_words - overlap):
        chunk = " ".join(words[i:i + chunk_words])
        chunks.append(chunk)
        if i + chunk_words >= len(words):
            break
    return chunks


def load_pubmedqa(max_docs=MAX_DOCS):
    """Load PubMedQA corpus — medical QA dataset."""
    print("Loading PubMedQA dataset...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    corpus = []
    docs_processed = 0
    for split in ["train"]:
        for item in dataset[split]:
            # Each item has a long_answer and contexts
            context = item.get("long_answer", "")
            question = item.get("question", "")
            pubid = item.get("pubid", docs_processed)

            if context and len(context.strip()) > 50:
                chunks = chunk_text(context.strip())
                for i, chunk in enumerate(chunks):
                    corpus.append({
                        "id": f"{pubid}_{i}",
                        "text": chunk,
                        "question": question,
                        "label": item.get("final_decision", "")
                    })
                docs_processed += 1

            if max_docs and docs_processed >= max_docs:
                break

    print(f"Loaded {len(corpus)} documents from PubMedQA")
    return corpus


def build_bm25_index(corpus):
    """Build BM25 sparse index."""
    print("Building BM25 index...")
    tokenized = [doc["text"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    return bm25


def build_faiss_index(corpus, model):
    """
    Build FAISS IVF_PQ index.

    IVF  → clusters vectors into nlist cells; at query time only
           nprobe cells are searched → huge speed gain
    PQ   → compresses vectors via product quantization → memory efficient
    """
    print("Encoding documents with sentence-transformers...")
    texts = [doc["text"] for doc in corpus]

    # Encode in batches to avoid OOM
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True   # cosine sim via inner product
    )

    embeddings = embeddings.astype(np.float32)
    n_docs = len(embeddings)

    print(f"Building FAISS IndexFlatIP index (n_docs={n_docs})...")

    # Exact search index (inner product == cosine for normalized vectors)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)

    # Add vectors
    index.add(embeddings)

    print(f"FAISS index built: {index.ntotal} vectors")
    return index, embeddings


def save_indexes(faiss_index, bm25_index, corpus):
    """Persist indexes to disk."""
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Save FAISS
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    print(f"FAISS index saved -> {FAISS_INDEX_PATH}")

    # Save BM25 + corpus
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_index, f)
    print(f"BM25 index saved -> {BM25_INDEX_PATH}")

    with open(CORPUS_PATH, "wb") as f:
        pickle.dump(corpus, f)
    print(f"Corpus saved -> {CORPUS_PATH}")


def load_indexes():
    """Load pre-built indexes from disk."""
    print("Loading indexes from disk...")

    faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_index = pickle.load(f)

    with open(CORPUS_PATH, "rb") as f:
        corpus = pickle.load(f)

    print(f"Loaded: {faiss_index.ntotal} FAISS vectors, {len(corpus)} docs")
    return faiss_index, bm25_index, corpus


def build_all(max_docs=MAX_DOCS):
    """Full index build pipeline."""
    start = time.time()

    # Load data
    corpus = load_pubmedqa(max_docs=max_docs)

    # Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Build indexes
    bm25_index = build_bm25_index(corpus)
    faiss_index, _ = build_faiss_index(corpus, model)

    # Save
    save_indexes(faiss_index, bm25_index, corpus)

    elapsed = time.time() - start
    print(f"\nIndex build complete in {elapsed:.1f}s")
    return faiss_index, bm25_index, corpus, model