from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from app.models import Chunk, RetrievedChunk

_TOKEN_RE = re.compile(r"(?u)\b\w+\b")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def reciprocal_rank_fusion(
    *ranked_lists: list[str], k: int = 60
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridIndex:
    def __init__(self, index_dir: Path, embedding_model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt before using dense retrieval") from exc

        self.index_dir = index_dir
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.bm25 = None
        self.by_id: dict[str, Chunk] = {}

    @property
    def is_ready(self) -> bool:
        return bool(self.chunks) and self.embeddings is not None and self.bm25 is not None

    def build(self, chunks: Iterable[Chunk]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("No documents found to index")

        texts = [c.text for c in self.chunks]
        self.embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.by_id = {c.chunk_id: c for c in self.chunks}

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        (self.index_dir / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        assert self.embeddings is not None
        np.save(self.index_dir / "embeddings.npy", self.embeddings)

    def load(self) -> None:
        chunks_path = self.index_dir / "chunks.json"
        embeddings_path = self.index_dir / "embeddings.npy"
        if not chunks_path.exists() or not embeddings_path.exists():
            raise FileNotFoundError("Index missing. Run: python -m scripts.ingest")

        self.chunks = [
            Chunk(**item)
            for item in json.loads(chunks_path.read_text(encoding="utf-8"))
        ]
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.embeddings = np.load(embeddings_path)

        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    def dense_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self.embeddings is None:
            raise RuntimeError("Index not loaded")
        q = self.embedding_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        scores = self.embeddings @ q
        order = np.argsort(-scores)[:top_k]
        return [
            RetrievedChunk(self.chunks[i], float(scores[i]), dense_rank=rank)
            for rank, i in enumerate(order, start=1)
        ]

    def sparse_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self.bm25 is None:
            raise RuntimeError("Index not loaded")
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-np.asarray(scores))[:top_k]
        return [
            RetrievedChunk(self.chunks[i], float(scores[i]), sparse_rank=rank)
            for rank, i in enumerate(order, start=1)
        ]

    def hybrid_search(
        self, query: str, dense_k: int, sparse_k: int, fused_k: int
    ) -> list[RetrievedChunk]:
        dense = self.dense_search(query, dense_k)
        sparse = self.sparse_search(query, sparse_k)

        fused = reciprocal_rank_fusion(
            [x.chunk.chunk_id for x in dense],
            [x.chunk.chunk_id for x in sparse],
        )

        dense_rank = {x.chunk.chunk_id: x.dense_rank for x in dense}
        sparse_rank = {x.chunk.chunk_id: x.sparse_rank for x in sparse}
        dense_score = {x.chunk.chunk_id: x.score for x in dense}
        sparse_score = {x.chunk.chunk_id: x.score for x in sparse}

        results: list[RetrievedChunk] = []
        for chunk_id, fused_score in fused[:fused_k]:
            score = max(
                dense_score.get(chunk_id, float("-inf")),
                sparse_score.get(chunk_id, float("-inf")),
            )
            results.append(
                RetrievedChunk(
                    self.by_id[chunk_id],
                    float(score),
                    dense_rank=dense_rank.get(chunk_id),
                    sparse_rank=sparse_rank.get(chunk_id),
                    fused_score=float(fused_score),
                )
            )
        return results
