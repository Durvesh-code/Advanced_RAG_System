from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from app.models import Chunk


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 140) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = para
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size].strip())
            current = current[chunk_size - overlap :].strip()

    if current:
        chunks.append(current)
    return [c for c in chunks if c]


def read_documents(data_dir: Path, chunk_size: int, overlap: int) -> Iterator[Chunk]:
    try:
        import fitz
    except ImportError:
        fitz = None

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()

        if suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8")
            for i, part in enumerate(chunk_text(text, chunk_size, overlap)):
                yield Chunk(f"{path.name}:c{i}", path.name, None, part)

        elif suffix == ".pdf" and fitz is not None:
            doc = fitz.open(path)
            try:
                for page_num, page in enumerate(doc, start=1):
                    for i, part in enumerate(
                        chunk_text(page.get_text("text"), chunk_size, overlap)
                    ):
                        yield Chunk(
                            f"{path.name}:p{page_num}:c{i}",
                            path.name,
                            page_num,
                            part,
                        )
            finally:
                doc.close()
