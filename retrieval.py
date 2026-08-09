"""
Retriever — Hybrid BM25 + Dense retrieval with Reciprocal Rank Fusion.

Hot path: zero LLM calls.
Query flow:
  1. BM25 sparse retrieval (keyword matching)
  2. Dense retrieval via FAISS IVF_PQ (semantic matching)
  3. RRF fusion → merge ranked lists
  4. Return top-K passages
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class HybridRetriever:
    def __init__(self, faiss_index, bm25_index, corpus, model: SentenceTransformer, reranker):
        self.faiss_index = faiss_index
        self.bm25_index  = bm25_index
        self.corpus      = corpus
        self.model       = model
        self.reranker    = reranker

    # ─── BM25 Retrieval ───────────────────────────────────────────────────────
    def _bm25_retrieve(self, query: str, top_k: int = BM25_TOP_K):
        """
        Sparse keyword-based retrieval.
        Fast, no GPU needed, great for exact medical terminology.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    # ─── Dense Retrieval ──────────────────────────────────────────────────────
    def _dense_retrieve(self, query: str, top_k: int = DENSE_TOP_K):
        """
        Semantic dense retrieval via FAISS IVF_PQ.
        Catches meaning-level matches BM25 misses.
        No LLM — pure embedding model inference.
        """
        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.faiss_index.search(query_vec, top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx != -1]

    # ─── Reciprocal Rank Fusion ───────────────────────────────────────────────
    def _rrf_fusion(self, bm25_results, dense_results, k: int = RRF_K):
        """
        Reciprocal Rank Fusion — combines two ranked lists without score normalization.

        Formula: RRF(d) = Σ 1 / (k + rank(d))

        Why RRF over score normalization?
        - BM25 and cosine scores are on different scales
        - Normalization amplifies noise; RRF is robust to score magnitude
        - k=60 is empirically optimal (Cormack et al. 2009)
        """
        rrf_scores = {}

        for rank, (doc_id, _) in enumerate(bm25_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)

        for rank, (doc_id, _) in enumerate(dense_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)

        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_ids

    # ─── Main Retrieve ─────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = TOP_K, mode: str = "hybrid", verbose: bool = False):
        """
        Full hybrid retrieval pipeline with Relevance Validation & Adaptive Search.
        """
        t0 = time.time()
        
        import feedback
        from relevance import validate_relevance

        negative_ids = feedback.get_negative_document_ids(query)

        if mode == "dense":
            t_dense = time.time()
            dense_results = self._dense_retrieve(query, top_k=DENSE_TOP_K)
            dense_ms = (time.time() - t_dense) * 1000
            bm25_ms = 0
            rrf_ms = 0
            fused = dense_results
        else:
            # Step 1: BM25
            t_bm25 = time.time()
            bm25_results = self._bm25_retrieve(query, top_k=BM25_TOP_K)
            bm25_ms = (time.time() - t_bm25) * 1000

            # Step 2: Dense
            t_dense = time.time()
            dense_results = self._dense_retrieve(query, top_k=DENSE_TOP_K)
            dense_ms = (time.time() - t_dense) * 1000

            # Step 3: RRF Fusion
            t_rrf = time.time()
            fused = self._rrf_fusion(bm25_results, dense_results)
            rrf_ms = (time.time() - t_rrf) * 1000

        # Filter out user feedback negatives
        filtered_fused = []
        for doc_id, score in fused:
            if 0 <= doc_id < len(self.corpus):
                doc_str_id = str(self.corpus[doc_id]['id'])
                if doc_str_id not in negative_ids:
                    filtered_fused.append((doc_id, score))

        # Step 4: Cross-Encoder Reranking and Relevance Validation
        t_rerank = time.time()
        rerank_candidates = filtered_fused[:RERANK_TOP_K]
        
        cross_inp = []
        for doc_id, _ in rerank_candidates:
            cross_inp.append([query, self.corpus[doc_id]["text"]])

        results = []
        confident = False
        
        if cross_inp:
            cross_scores = self.reranker.predict(cross_inp)
            scored_results = []
            for i, (doc_id, base_score) in enumerate(rerank_candidates):
                doc = self.corpus[doc_id]
                val = validate_relevance(query, doc["text"], float(cross_scores[i]))
                
                rel_score = val["relevance_score"]
                if rel_score >= CONFIDENCE_HIGH:
                    confidence = "high"
                elif rel_score >= CONFIDENCE_MEDIUM:
                    confidence = "medium"
                else:
                    confidence = "low"
                    
                scored_results.append({
                    "id":        doc["id"],
                    "text":      doc["text"],
                    "rrf_score": round(base_score, 6),
                    "bm25_score": 0.0,
                    "dense_score": 0.0,
                    "relevance_score": rel_score,
                    "confidence": confidence,
                    "matched_concepts": val["matched_concepts"],
                    "missing_concepts": val["missing_concepts"],
                    "doc_idx":   doc_id
                })
            
            # Sort by Relevance Score
            scored_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)
            
            # Adaptive Retrieval Check
            # Only consider results above the medium threshold
            confident_results = [r for r in scored_results if r["relevance_score"] >= CONFIDENCE_MEDIUM]
            
            if confident_results:
                confident = True
                results = confident_results[:top_k]
            else:
                confident = False
                results = []
        
        rerank_ms = (time.time() - t_rerank) * 1000
        total_ms = (time.time() - t0) * 1000

        if verbose:
            print(f"\n{'-'*50}")
            print(f"Query     : {query[:80]}...")
            print(f"Total     : {total_ms:.1f}ms")
            print(f"Results   : {len(results)} confident passages")
            print(f"{'-'*50}")

        return {
            "query": query,
            "results": results,
            "latency_ms": round(total_ms, 2),
            "mode": mode,
            "confident": confident,
            "timing": {
                "total_ms":  round(total_ms, 2),
                "bm25_ms":   round(bm25_ms, 2),
                "dense_ms":  round(dense_ms, 2),
                "rrf_ms":    round(rrf_ms, 2),
                "rerank_ms": round(rerank_ms, 2),
                "hit_target": total_ms < LATENCY_TARGET_MS
            }
        }

    def add_document(self, text: str, question: str = "", label: str = ""):
        """
        Dynamically index a new document:
          1. Chunk the text.
          2. Encode the chunks to vectors and add to FAISS index.
          3. Re-instantiate BM25 with updated corpus.
          4. Save changes to disk.
        """
        from index import chunk_text
        chunks = chunk_text(text)

        # 1. Encode chunks and add to FAISS
        query_vecs = self.model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        self.faiss_index.add(query_vecs)

        # 2. Append to corpus
        base_id = len(self.corpus) + 100000
        for i, chunk in enumerate(chunks):
            self.corpus.append({
                "id": f"{base_id}_{i}",
                "text": chunk,
                "question": question,
                "label": label
            })

        # 3. Rebuild BM25 index
        from rank_bm25 import BM25Okapi
        tokenized = [doc["text"].lower().split() for doc in self.corpus]
        self.bm25_index = BM25Okapi(tokenized)

        # 4. Save to disk for persistence
        import faiss
        import pickle
        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(self.faiss_index, FAISS_INDEX_PATH)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(self.bm25_index, f)
        with open(CORPUS_PATH, "wb") as f:
            pickle.dump(self.corpus, f)

        return str(base_id)