from __future__ import annotations

from app.config import settings
from app.services.ingestion import read_documents
from app.services.retrieval import HybridIndex


def main() -> None:
    chunks = list(
        read_documents(
            settings.data_dir,
            settings.chunk_size,
            settings.chunk_overlap,
        )
    )

    index = HybridIndex(settings.index_dir, settings.embedding_model)
    index.build(chunks)
    index.save()

    print(f"Indexed {len(chunks)} chunks into {settings.index_dir}/")


if __name__ == "__main__":
    main()
