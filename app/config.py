from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    index_dir: Path = Path(os.getenv("INDEX_DIR", "index"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    top_k_dense: int = int(os.getenv("TOP_K_DENSE", "8"))
    top_k_sparse: int = int(os.getenv("TOP_K_SPARSE", "8"))
    top_k_fused: int = int(os.getenv("TOP_K_FUSED", "10"))
    top_k_reranked: int = int(os.getenv("TOP_K_RERANKED", "4"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "140"))
    data_dir: Path = Path("data")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None


settings = Settings()
