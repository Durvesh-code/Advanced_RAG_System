import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

def sanitize_metadata(metadata: dict) -> dict:
    """
    Converts any numpy scalar types in metadata to native Python types
    so Pydantic can serialize them to JSON without errors.
    FlashRank reranker adds numpy float32 relevance scores to metadata.
    """
    clean = {}
    for key, value in metadata.items():
        if isinstance(value, np.generic):  # catches float32, int64, bool_, etc.
            clean[key] = value.item()      # converts to native Python int/float/bool
        else:
            clean[key] = value
    return clean
# Ensure our API can find the 'core' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.ingestion import load_and_chunk_pdfs
from core.retrieval import build_advanced_retriever
from core.generation import generate_answer
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This runs exactly once when you start the server.
    It loads the PDFs, builds the embeddings, and prepares the LLM.
    """
    print("\n--- 🚀 Booting up Advanced RAG Pipeline ---")
    
    # 1. Load the chunks
    chunks = load_and_chunk_pdfs("data")
    
    if not chunks:
        print("⚠️ WARNING: No PDFs found in the 'data' folder. The API will run, but won't be able to answer questions.")
    else:
        # 2. Build the Hybrid + Reranker pipeline
        print("\nBuilding retrieval pipeline (this may take a moment)...")
        app_state["retriever"] = build_advanced_retriever(chunks, persist_directory="vector_store")
        print("✅ Pipeline ready and loaded into memory!")
        
    yield # The server runs while paused here
    
    # This runs when you shut the server down
    print("\n--- 🛑 Shutting down ---")
    app_state.clear()

# Initialize the FastAPI application
app = FastAPI(
    title="Agentic RAG Backend",
    description="Hybrid Search + Cross-Encoder Reranking API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock this to your actual frontend domain before going live
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root_redirect():
    """Redirects the base URL automatically to the Swagger API documentation."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

# --- Pydantic Data Models ---
class QueryRequest(BaseModel):
    query: str

class DocumentInfo(BaseModel):
    content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    context_used: List[DocumentInfo]

# --- API Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "online", "message": "API is operational"}

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """
    Receives a query, runs it through the RAG pipeline, and returns the answer.
    """
    retriever = app_state.get("retriever")
    if not retriever:
        raise HTTPException(status_code=500, detail="RAG Pipeline is not initialized. Please check server logs.")
    
    print(f"\n[Incoming Query]: {request.query}")
    
    # 1. Retrieve the highly filtered chunks
    retrieved_docs = retriever.invoke(request.query)
    
    # 2. Generate the final answer using OpenAI
    answer = generate_answer(request.query, retrieved_docs)
    
    # 3. Format the exact context used so the frontend can display "Sources"
    # sanitize_metadata ensures numpy types (e.g. float32 from FlashRank) are
    # converted to native Python types before Pydantic tries to serialize them.
    context_used = [
        DocumentInfo(content=doc.page_content, metadata=sanitize_metadata(doc.metadata)) 
        for doc in retrieved_docs
    ]
    
    return QueryResponse(answer=answer, context_used=context_used)