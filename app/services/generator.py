from __future__ import annotations

from openai import OpenAI

from app.config import settings
from app.models import RetrievedChunk

SYSTEM_PROMPT = """You are a grounded question-answering system.
Answer only from the supplied context. Do not invent facts, citations, or numbers.
When the context is insufficient, say that the answer is not available in the indexed documents.
Cite supporting chunks inline using [1], [2], etc.
"""


def generate_answer(query: str, sources: list[RetrievedChunk]) -> str:
    if not sources:
        return "I could not find relevant evidence in the indexed documents."
    if not settings.openai_api_key:
        return (
            "LLM generation is disabled because OPENAI_API_KEY is not configured. "
            "See the returned sources for grounded evidence."
        )

    context = "\n\n".join(
        f"[{i}] {item.chunk.text}\nSource: {item.chunk.source}"
        + (f", page {item.chunk.page}" if item.chunk.page else "")
        for i, item in enumerate(sources, start=1)
    )

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.responses.create(
        model=settings.openai_model,
        instructions=SYSTEM_PROMPT,
        input=f"Question: {query}\n\nContext:\n{context}",
    )
    return response.output_text
