# Retrieval Notes

Hybrid retrieval combines semantic search with lexical search. Dense retrieval captures meaning and paraphrases, while BM25 is useful for exact identifiers, names, and rare terms.

Reciprocal Rank Fusion (RRF) combines ranked dense and sparse candidate lists without assuming their raw scores share a common scale. A candidate appearing near the top of both lists receives a stronger fused score.

A cross-encoder reranker runs after candidate generation. It jointly reads the query and passage, which allows richer interaction than independent embeddings, but is more expensive. Keeping the reranking shortlist small makes that trade-off practical.

Separating dense retrieval, sparse retrieval, fusion, and reranking also makes failures easier to diagnose. If a useful passage never enters the candidate set, the reranker cannot recover it.
