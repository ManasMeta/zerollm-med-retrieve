# Zero-LLM Hybrid Medical Retrieval System

This repository implements a lightweight, low-latency, and memory-optimized passage retrieval layer designed for natural-language question answering over a medical document corpus. It is built to operate under strict constraints (CPU-friendly, sub-100ms latency, and a sub-2GB memory footprint) without calling any LLMs on the query hot path.

---

## 1. System Architecture

The retrieval layer uses a **Hybrid Search** design combining keyword search and dense semantic search, merged via **Reciprocal Rank Fusion (RRF)**.

```
                  ┌───► Sparse Keyword Search (BM25) ───┐
User Query ───────┤                                     ├───► Rank Fusion (RRF) ───► Top K Passages
                  └───► Dense Semantic Search (FAISS) ──┘
```

### Core Components
- **[config.py](file:///c:/Users/Lenovo%20Thinkpad%20T14/Desktop/noLLM/medical-retrieval/config.py)**: Holds configuration hyperparameters, model definitions, and indexing parameters.
- **[index.py](file:///c:/Users/Lenovo%20Thinkpad%20T14/Desktop/noLLM/medical-retrieval/index.py)**: Builds offline search indexes. It parses the dataset, trains a BM25 model, generates sentence embeddings, and builds a FAISS index.
- **[retrieval.py](file:///c:/Users/Lenovo%20Thinkpad%20T14/Desktop/noLLM/medical-retrieval/retrieval.py)**: Houses the query pipeline. The [HybridRetriever](file:///c:/Users/Lenovo%20Thinkpad%20T14/Desktop/noLLM/medical-retrieval/retrieval.py#L20) class runs sparse and dense queries and fuses them.
- **[evaluator.py](file:///c:/Users/Lenovo%20Thinkpad%20T14/Desktop/noLLM/medical-retrieval/evaluator.py)**: Evaluates quality metrics (Recall@K, MRR, NDCG@K) and tracking latency stats over 100 queries from `PubMedQA` ground truth.
- **[Main.py](file:///c:/Users/Lenovo%20Thinkpad%20T14/Desktop/noLLM/medical-retrieval/Main.py)**: Central CLI entry point.

---

## 2. Constraints & Design Decisions

### Memory Footprint (< 2 GB Ceiling)
- **Model Selection**: We use `sentence-transformers/all-MiniLM-L6-v2` (only **80MB** on disk and RAM), which is fast and CPU-friendly.
- **Quantization (FAISS IVF_PQ)**: Raw vector search (Flat index) scales linearly in memory. We use **Inverted File (IVF) + Product Quantization (PQ)**. IVF clusters vectors into 100 cells, and PQ compresses float32 dimensions into byte-level codes. This yields a **~98% memory reduction** at scale (reducing 1 Million documents from 1.5 GB to 16 MB).

### Latency Budget (sub-100 ms)
- **Routing**: `FAISS_NPROBE = 10` forces the dense retriever to search only the closest 10 clusters (avoiding scanning the other 90 clusters).
- **Candidate Pooling**: We retrieve only the top 50 candidates from both sparse and dense searches before running RRF, keeping merge time under **1 ms**.

---

## 3. Experimental Benchmarks

We evaluated 6 different retrieval configurations over the `PubMedQA` dataset (1,000 documents, 100 queries):

| Experiment | Recall@5 | MRR | NDCG@5 | Avg Latency | p99 Latency | Index Size | Key Takeaway |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Baseline (Hybrid)** | 0.8700 | 0.8237 | 0.8354 | 45.76 ms | 73.73 ms | 0.545 MB | Balanced starting point (`nprobe=10`, pool=50) |
| **2. Max Accuracy (Hybrid)** | 0.8700 | 0.8250 | 0.8365 | 48.85 ms | 99.21 ms | 0.545 MB | Searching more clusters (`nprobe=100`) adds latency for minimal gain |
| **3. Low Latency (Hybrid)** | 0.8800 | 0.7928 | 0.8150 | 36.27 ms | **49.69 ms** | 0.545 MB | Shrinking RRF pool to 10 improves Recall & drops p99 latency to < 50ms |
| **4. Uncompressed (Hybrid)** | 0.8800 | 0.8267 | 0.8402 | 34.92 ms | 46.43 ms | 1.465 MB | Exact Float32 Flat index. Best theoretical accuracy, but uses 3x more RAM |
| **5. Sparse-Only (BM25)** | 0.7800 | 0.6818 | 0.7060 | **7.16 ms** | **15.49 ms** | **0.466 MB** | Lightning-fast and zero neural network RAM, but poor semantic accuracy |
| **6. Dense-Only (FAISS)** | **0.9100** | **0.8352** | **0.8535** | **36.49 ms** | 96.81 ms | 0.545 MB | **Best overall accuracy and lower average latency** |

### Key Architectural Takeaways

1. **Rank Noise in RRF**: 
   - Interestingly, **Dense-Only Search (Exp 6)** outperforms the Hybrid Baseline in accuracy (Recall@5 of **91.0%** vs **87.0%**).
   - *Reason*: BM25 keyword search ranks documents highly if they share exact generic medical terms (like *"patients"* or *"disease"*), even if they are semantically unrelated. RRF boosts these noisy matches, displacing highly relevant semantic results. Removing BM25 resolves this issue and saves latency.
2. **Low Latency Tuning**:
   - Limiting the candidate pool to 10 (Exp 3) instead of 50 (Exp 1) drops p99 latency from **73.73 ms** to **49.69 ms** and actually *improves* Recall (+1%) by pruning low-rank noise.
3. **The Quantization Tradeoff**:
   - On a small dataset (1,000 docs), exact search (Exp 4) is faster and more accurate. But at production scale (1M+ docs), uncompressed index sizes scale linearly to gigabytes. IVF_PQ (Exp 1) keeps memory consumption strictly bounded to megabytes at the cost of a minor (1%) accuracy loss.

---

## 4. Setup and Execution

### Prerequisites & Dependencies
To run this project, make sure the dependencies in [requirements.txt](file:///c:/Users/Lenovo%20Thinkpad%20T14/Desktop/noLLM/medical-retrieval/requirements.txt) are installed:
```bash
pip install -r requirements.txt
```

### Windows/Keras Version Compatibility
*Note: If Keras 3 is installed in your environment, Hugging Face `transformers` might run into conflicts. The code sets environment variables programmatically on startup in [Main.py](file:///c:/Users/Lenovo%20Thinkpad%20T14/Desktop/noLLM/medical-retrieval/Main.py) to ignore TensorFlow and bypass Keras version checks:*
```python
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "NO"
```

### CLI Command Reference

1. **Build Indexes** (Offline Phase - run once):
   ```bash
   python Main.py --build
   ```
2. **Execute Single Query**:
   ```bash
   python Main.py --query "What is the recommended treatment for Type 2 diabetes?"
   ```
3. **Run Performance Evaluation Suite**:
   ```bash
   python Main.py --eval
   ```
4. **Memory Footprint Report**:
   ```bash
   python Main.py --memory
   ```
5. **Interactive CLI Demo**:
   ```bash
   python Main.py --demo
   ```
