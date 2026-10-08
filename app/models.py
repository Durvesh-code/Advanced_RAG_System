from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    source: str
    page: int | None
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_score: float | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "source": self.chunk.source,
            "page": self.chunk.page,
            "text": self.chunk.text,
            "score": self.score,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "fused_score": self.fused_score,
            "rerank_score": self.rerank_score,
        }
