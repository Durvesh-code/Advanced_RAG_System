# Generation Policy

The generator receives only the top reranked passages. It must answer from those passages and must not invent facts, citations, or numbers.

When evidence is insufficient, the generator should explicitly say that the answer is unavailable in the indexed documents.

The API returns the generated answer together with the supporting chunks so a user can inspect the evidence behind the response.
