import sqlite3
import os
import threading

DB_PATH = "feedback.db"
_lock = threading.Lock()

def init_db():
    with _lock:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                document_id TEXT,
                bm25_score REAL,
                dense_score REAL,
                rrf_score REAL,
                relevance_score REAL,
                rank INTEGER,
                feedback TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

def record_feedback(query, document_id, feedback_type, bm25_score=0.0, dense_score=0.0, rrf_score=0.0, relevance_score=0.0, rank=0):
    with _lock:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO feedback 
            (query, document_id, bm25_score, dense_score, rrf_score, relevance_score, rank, feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (query, document_id, bm25_score, dense_score, rrf_score, relevance_score, rank, feedback_type))
        conn.commit()
        conn.close()

def get_negative_document_ids(query: str):
    """Returns a set of document IDs that the user previously flagged as 'not_relevant' for this specific query."""
    with _lock:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT document_id FROM feedback 
            WHERE query = ? AND feedback = 'not_relevant'
        ''', (query,))
        rows = cursor.fetchall()
        conn.close()
    return {row[0] for row in rows}

# Initialize on import
init_db()
