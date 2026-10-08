# Advanced RAG System

A production-minded Retrieval-Augmented Generation prototype that focuses on **retrieval recall, candidate fusion, reranking, and evidence traceability**.

## Architecture

```text
Documents -> Ingestion + metadata
                 |
        +--------+--------+
        |                 |
   Dense Retrieval   Sparse Retrieval
   embeddings        BM25
        |                 |
        +--------+--------+
                 |
            RRF Fusion
                 |
        Cross-Encoder Reranker
                 |
          Grounded LLM
          + citations
```

## Why it is challenging

A naive RAG app sends the top vector-search results directly to an LLM. Two failures follow: the correct passage may never be retrieved, or a superficially similar passage may outrank the useful one.

This implementation separates those problems into measurable stages. Dense retrieval captures semantic similarity, BM25 protects exact terms and identifiers, Reciprocal Rank Fusion combines the ranked lists without requiring comparable raw score scales, and a cross-encoder reranks only the small fused shortlist.

Every passage preserves its source filename and, for PDFs, page number. The API returns both the generated answer and the supporting evidence.

## Stack

- Python + FastAPI
- Sentence Transformers dense embeddings
- BM25 sparse retrieval
- Reciprocal Rank Fusion
- Cross-encoder reranking
- OpenAI Responses API for grounded generation
- PyMuPDF for PDFs
- NumPy local vector store
- Pytest

## Run locally

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux

python -m scripts.ingest
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` and call `POST /ask`.

Put your own `.pdf`, `.md`, or `.txt` documents in `data/` and rerun ingestion.

## Benchmark

```bash
python -m scripts.benchmark
```

The benchmark reports Recall@K and MRR@K for a small hand-labeled query set. It is intended to make retrieval changes measurable rather than to claim a universal benchmark score.

## Project structure

```text
app/
  main.py
  config.py
  models.py
  services/
    ingestion.py
    retrieval.py
    reranker.py
    generator.py
scripts/
  ingest.py
  benchmark.py
tests/
  test_ingestion.py
  test_retrieval.py
data/
  *.md
```

## Design decisions

**Dense + sparse retrieval:** semantic embeddings handle paraphrases while lexical search protects rare keywords and exact identifiers.

**RRF fusion:** dense and sparse raw scores are not assumed to be comparable, so the fusion step uses rank.

**Cross-encoder reranking:** the reranker jointly reads the query and candidate passage. It is applied after recall-oriented candidate generation so the expensive stage sees only a small shortlist.

**Evidence traceability:** source metadata survives the entire pipeline and is exposed by the API.

## Future work

- Replace brute-force dense search with FAISS/ANN for large corpora.
- Add query rewriting and multi-query retrieval.
- Add parent-document retrieval.
- Add held-out evaluation for precision, recall, latency, and answer faithfulness.
- Add stage-level latency tracing and retrieval diagnostics.

## Project story

The central engineering lesson is to treat RAG as a **retrieval system first and an LLM application second**. A reranker cannot recover a passage that candidate generation missed, so retrieval recall and stage-by-stage evaluation matter more than simply swapping in a larger generator.
