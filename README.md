# 🔍 Advanced RAG Pipeline

> A **Retrieval-Augmented Generation (RAG)** system built with **FastAPI**, featuring Hybrid Search (Dense + Sparse), Cross-Encoder Reranking, and GPU-accelerated embeddings — designed for intelligent, context-grounded Q&A over PDF documents.



## 🧠 Overview

This project implements a **multi-stage Retrieval-Augmented Generation (RAG)** pipeline that goes beyond basic vector search. Instead of relying on a single retrieval strategy, it fuses **semantic (dense)** and **keyword (sparse)** search, then applies a **neural cross-encoder reranker** to select only the most relevant context before passing it to the LLM.

The result: fewer hallucinations, more precise answers, and citation-backed responses drawn strictly from your own documents.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│               FastAPI Server                │
│  POST /ask  →  retriever  →  generator      │
└───────────────┬─────────────────────────────┘
                │
    ┌───────────▼────────────┐
    │   Hybrid Retriever     │
    │  ┌────────┐ ┌────────┐ │
    │  │ Dense  │ │ Sparse │ │   (Ensemble: 50/50 weight)
    │  │ Chroma │ │  BM25  │ │
    │  └────────┘ └────────┘ │
    └───────────┬────────────┘
                │  Top 20 candidates
                ▼
    ┌───────────────────────┐
    │  FlashRank Reranker   │   (Cross-Encoder — picks Top 3)
    └───────────┬───────────┘
                │  Top 3 reranked chunks
                ▼
    ┌───────────────────────┐
    │     OpenAI GPT-4o     │   (Generates grounded answer)
    └───────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI + Uvicorn |
| **LLM** | OpenAI GPT-4o (`gpt-4o`) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (GPU/CUDA) |
| **Vector Store** | ChromaDB (persistent) |
| **Sparse Search** | BM25 (`rank_bm25`) |
| **Reranker** | FlashRank (Cross-Encoder, lightning fast) |
| **PDF Parsing** | PyMuPDF (`pymupdf`) |
| **Orchestration** | LangChain |
| **Config** | `python-dotenv` |

---

## 📁 Project Structure

```
Rag/
├── api/
│   ├── main.py          # FastAPI app, lifespan, endpoints, CORS
│   └── route.py         # (Additional routes)
├── core/
│   ├── ingestion.py     # PDF loading & recursive text chunking
│   ├── retrieval.py     # Hybrid retriever + FlashRank reranker
│   └── generation.py    # Prompt template + GPT-4o answer generation
├── data/                # 📂 Drop your PDF files here
├── vector_store/        # ChromaDB persisted embeddings (auto-created)
├── requirements.txt
├── .env                 # Your API keys (never commit this!)
└── README.md
```

---


## 🔐 Environment Variables

Create a `.env` file in the project root (copy from the example below):

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
```

## ⚙️ How It Works

### Stage 1 — Ingestion (`core/ingestion.py`)
- Scans the `data/` folder for all `.pdf` files
- Parses pages using **PyMuPDF**
- Splits text into 500-token chunks with 50-token overlap using `RecursiveCharacterTextSplitter`

### Stage 2 — Hybrid Retrieval (`core/retrieval.py`)
- **Dense Retriever**: Embeds chunks using `all-MiniLM-L6-v2` → stored in **ChromaDB** → fetches top-10 semantic matches
- **Sparse Retriever**: Indexes chunks with **BM25** → fetches top-10 keyword matches
- **Fusion**: `EnsembleRetriever` merges both result sets with equal 50/50 weighting

### Stage 3 — Reranking (`core/retrieval.py`)
- **FlashRank** cross-encoder scores all 20 combined candidates
- Only the **top 3 highest-relevance chunks** are passed forward

### Stage 4 — Generation (`core/generation.py`)
- A strict prompt template instructs the LLM to answer **only from the provided context**
- **GPT-4o** generates the final answer with `temperature=0.1` to minimize hallucinations

---

## 🔧 Configuration

You can tune the pipeline behavior by modifying these values directly in the source files:

| Parameter | File | Default | Description |
|---|---|---|---|
| `chunk_size` | `ingestion.py` | `500` | Token size per chunk |
| `chunk_overlap` | `ingestion.py` | `50` | Overlap between chunks |
| `dense k` | `retrieval.py` | `10` | Semantic results fetched |
| `sparse k` | `retrieval.py` | `10` | Keyword results fetched |
| `top_n` | `retrieval.py` | `3` | Final reranked chunks sent to LLM |
| `model` | `generation.py` | `gpt-4o` | OpenAI model to use |
| `temperature` | `generation.py` | `0.1` | LLM creativity (lower = more factual) |
| `device` | `retrieval.py` | `cuda` | Embedding device (`cuda` or `cpu`) |

---



<div align="center">
  Built with ❤️ using LangChain, FastAPI, and OpenAI
</div>
