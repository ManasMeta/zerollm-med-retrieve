import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_search_empty_query():
    response = client.post("/search", json={"query": "", "mode": "hybrid", "top_k": 5})
    assert response.status_code == 400

def test_synthesize_ungrounded():
    response = client.post("/synthesize", json={
        "query": "What is the capital of France?",
        "passages": []
    })
    assert response.status_code == 200
    res = response.json()
    assert res["grounded"] == False
    assert res["warning"] is not None

def test_synthesize_grounded():
    response = client.post("/synthesize", json={
        "query": "What does the passage say?",
        "passages": ["The passage says hello world."]
    })
    assert response.status_code == 200
    res = response.json()
    assert res["grounded"] == True
    assert res["warning"] is None

def test_feedback():
    response = client.post("/feedback", json={
        "query": "test query",
        "document_id": "doc_1",
        "feedback": "relevant"
    })
    assert response.status_code == 200
    
    response = client.post("/feedback", json={
        "query": "test query",
        "document_id": "doc_1",
        "feedback": "invalid_type"
    })
    assert response.status_code == 400
