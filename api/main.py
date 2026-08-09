import os
import time
import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import openai

# Ensure TensorFlow is bypassed
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "NO"

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Main import get_retriever
from config import TOP_K
import feedback

app = FastAPI(title="Zero-LLM Relevance-Aware Hybrid Retrieval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = None
corpus = None

# Background build state
build_jobs = {}

class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    top_k: int = TOP_K

class SynthesizeRequest(BaseModel):
    query: str
    passages: List[str]

class FeedbackRequest(BaseModel):
    query: str
    document_id: str
    feedback: str

class DocumentRequest(BaseModel):
    text: str
    question: Optional[str] = ""
    label: Optional[str] = ""

@app.on_event("startup")
async def startup_event():
    global retriever, corpus
    print("Loading indexes and models...")
    try:
        retriever, corpus = get_retriever()
        print("Successfully loaded retriever and corpus.")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search")
async def search(req: SearchRequest):
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever is not fully loaded yet.")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        results = retriever.retrieve(req.query, top_k=req.top_k, mode=req.mode, verbose=False)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not req.passages:
        # Ungrounded mode
        if not api_key:
            return {
                "answer": "Error: OPENAI_API_KEY environment variable is not set. Cannot synthesize answer.",
                "grounded": False,
                "warning": "No sufficiently relevant evidence was found in the uploaded dataset."
            }
        
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant. Answer the user's question based on your general knowledge. Keep it concise."},
                    {"role": "user", "content": req.query}
                ],
                max_tokens=300
            )
            return {
                "answer": response.choices[0].message.content,
                "grounded": False,
                "warning": "No sufficiently relevant evidence was found in the uploaded dataset."
            }
        except Exception as e:
            return {"answer": str(e), "grounded": False, "warning": "API Error"}

    # Grounded mode
    if not api_key:
        return {
            "answer": "Error: OPENAI_API_KEY environment variable is not set. Cannot synthesize answer.",
            "grounded": True,
            "warning": None
        }

    context = "\n\n".join([f"Passage {i+1}:\n{p}" for i, p in enumerate(req.passages)])
    system_prompt = (
        "You are a medical data synthesis assistant.\n"
        "You MUST answer the user's query ONLY using the supplied passages.\n"
        "Do not invent evidence or fabricate citations.\n"
        "If the passages do not contain a confident answer to the query, explicitly say: "
        "'The provided passages do not contain a confident answer.'"
    )
    user_prompt = f"Query: {req.query}\n\n{context}"

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=300
        )
        return {
            "answer": response.choices[0].message.content,
            "grounded": True,
            "warning": None
        }
    except Exception as e:
        return {"answer": str(e), "grounded": True, "warning": "API Error"}

@app.post("/feedback")
async def add_feedback(req: FeedbackRequest):
    if req.feedback not in ["relevant", "not_relevant"]:
        raise HTTPException(status_code=400, detail="Invalid feedback type")
    
    feedback.record_feedback(req.query, req.document_id, req.feedback)
    return {"status": "success"}

@app.post("/documents")
async def add_document(req: DocumentRequest):
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever is not fully loaded yet.")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Document text cannot be empty.")
    
    try:
        doc_id = retriever.add_document(text=req.text, question=req.question, label=req.label)
        return {"status": "success", "doc_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def background_build(job_id: str):
    import index
    try:
        build_jobs[job_id]["status"] = "running"
        build_jobs[job_id]["progress"] = 10
        build_jobs[job_id]["message"] = "Starting build..."
        
        index.build_all() # This rebuilds everything
        
        # Reload models in memory
        global retriever, corpus
        retriever, corpus = get_retriever()
        
        build_jobs[job_id]["status"] = "completed"
        build_jobs[job_id]["progress"] = 100
        build_jobs[job_id]["message"] = "Indexing completed successfully."
    except Exception as e:
        build_jobs[job_id]["status"] = "failed"
        build_jobs[job_id]["message"] = str(e)

@app.post("/build")
async def trigger_build(background_tasks: BackgroundTasks):
    import uuid
    job_id = str(uuid.uuid4())
    build_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued for indexing."
    }
    background_tasks.add_task(background_build, job_id)
    return {"job_id": job_id, "status": "queued"}

@app.get("/build/status/{job_id}")
async def get_build_status(job_id: str):
    if job_id not in build_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return build_jobs[job_id]

# Mount static files for the frontend if they exist
if os.path.exists("../frontend/dist"):
    app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
