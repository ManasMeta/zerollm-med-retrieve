import os
import time
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles
# pyrefly: ignore [missing-import]
from fastapi.responses import FileResponse
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
# pyrefly: ignore [missing-import]
import uvicorn

# Ensure TensorFlow is bypassed before importing sentence_transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "NO"

from Main import get_retriever
from config import TOP_K

app = FastAPI(title="Zero-LLM Medical Retrieval API")

# Global variables to hold our models in memory
retriever = None
corpus = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = TOP_K

@app.on_event("startup")
async def startup_event():
    global retriever, corpus
    print("Loading indexes and models into memory... This may take a moment.")
    try:
        retriever, corpus = get_retriever()
        print("Successfully loaded retriever and corpus.")
    except Exception as e:
        print(f"Error loading models: {e}")
        # In a real app, you might want to handle this gracefully or trigger a build if missing

@app.post("/api/search")
async def search(req: SearchRequest):
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever is not fully loaded yet.")
    
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        results, timing = retriever.retrieve(req.query, top_k=req.top_k, verbose=False)
        return {
            "results": results,
            "timing": timing
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

class DocumentRequest(BaseModel):
    text: str
    question: str = ""
    label: str = ""

@app.post("/api/documents")
async def add_document(req: DocumentRequest):
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever is not fully loaded yet.")
    
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Document text cannot be empty.")
    
    try:
        doc_id = retriever.add_document(
            text=req.text.strip(),
            question=req.question.strip(),
            label=req.label.strip()
        )
        return {
            "success": True,
            "doc_id": doc_id,
            "message": "Document successfully indexed and persisted."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def serve_frontend():
    """Serve the main frontend HTML file."""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    else:
        return {"message": "Frontend not found. Please create static/index.html"}

if __name__ == "__main__":
    print("Starting server on http://localhost:8000")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
