# ─────────────────────────────────────────────
# Configuration — Medical Retrieval System
# ─────────────────────────────────────────────

# Embedding model — Medical Domain Specific
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
EMBEDDING_DIM   = 768

# Reranker settings
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K    = 20

# FAISS index settings
# Using Exact Search (IndexFlatIP) for maximum recall
# Chunking settings
CHUNK_WORDS     = 150
CHUNK_OVERLAP   = 30

# Retrieval settings
TOP_K           = 5            # passages to return
BM25_TOP_K      = 50           # BM25 candidates before fusion
DENSE_TOP_K     = 50           # dense candidates before fusion
RRF_K           = 60           # RRF constant (standard = 60)

# Dataset
DATASET_NAME    = "qiaojin/PubMedQA"
DATASET_CONFIG  = "pqa_labeled"
MAX_DOCS        = 1000        

# Paths
INDEX_DIR       = "indexes"
FAISS_INDEX_PATH = "indexes/faiss.index"
BM25_INDEX_PATH  = "indexes/bm25.pkl"
CORPUS_PATH      = "indexes/corpus.pkl"

# Latency target
LATENCY_TARGET_MS = 500