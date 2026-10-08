from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.services.retrieval import HybridIndex


@dataclass(frozen=True)
class QueryCase:
    query: str
    expected_source: str


CASES = [
    QueryCase("How does RRF combine dense and sparse retrieval?", "retrieval_notes.md"),
    QueryCase("Why is a cross-encoder used after candidate retrieval?", "retrieval_notes.md"),
    QueryCase("What should a grounded generator do when evidence is missing?", "generation_policy.md"),
    QueryCase("What metadata is retained for PDF chunks?", "engineering_notes.md"),
]


def main() -> None:
    index = HybridIndex(settings.index_dir, settings.embedding_model)
    index.load()

    hits = 0
    reciprocal_rank = 0.0

    for case in CASES:
        results = index.hybrid_search(
            case.query,
            settings.top_k_dense,
            settings.top_k_sparse,
            settings.top_k_fused,
        )
        source_rank = next(
            (i for i, item in enumerate(results, start=1) if item.chunk.source == case.expected_source),
            None,
        )

        if source_rank is not None:
            hits += 1
            reciprocal_rank += 1.0 / source_rank

        print(f"{case.query}\n  expected={case.expected_source} rank={source_rank}\n")

    print(f"Recall@{settings.top_k_fused}: {hits / len(CASES):.3f}")
    print(f"MRR@{settings.top_k_fused}: {reciprocal_rank / len(CASES):.3f}")


if __name__ == "__main__":
    main()
