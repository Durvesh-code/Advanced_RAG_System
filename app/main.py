from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.services.generator import generate_answer
from app.services.reranker import Reranker
from app.services.retrieval import HybridIndex


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        index = HybridIndex(settings.index_dir, settings.embedding_model)
        index.load()
        reranker = Reranker(settings.reranker_model)
    except (FileNotFoundError, RuntimeError):
        index = None
        reranker = None

    app.state.index = index
    app.state.reranker = reranker
    yield


app = FastAPI(
    title="Advanced RAG System",
    version="2.0.0",
    description="Hybrid dense+sparse retrieval with RRF fusion, cross-encoder reranking, and grounded generation.",
)


@app.get("/health")
def health() -> dict:
    index = app.state.index
    return {"status": "ok", "indexed": bool(index and index.is_ready)}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    index = app.state.index
    reranker = app.state.reranker

    if not index or not index.is_ready or reranker is None:
        raise HTTPException(
            status_code=503,
            detail="Index is not ready. Run: python -m scripts.ingest",
        )

    candidates = index.hybrid_search(
        request.question,
        settings.top_k_dense,
        settings.top_k_sparse,
        settings.top_k_fused,
    )

    reranked = reranker.rerank(
        request.question,
        candidates,
        request.top_k or settings.top_k_reranked,
    )
    answer = generate_answer(request.question, reranked)

    return AskResponse(
        answer=answer,
        sources=[x.to_dict() for x in reranked],
    )
