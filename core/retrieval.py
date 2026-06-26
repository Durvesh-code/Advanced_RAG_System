import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever

def get_embedding_model():
    """
    Initializes the embedding model.
    Configured to use CUDA to fully leverage GPU acceleration for high-speed local processing.
    """
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'} # Forces GPU usage 
    )
    return embeddings

def build_advanced_retriever(chunks, persist_directory="../vector_store"):
    """
    Builds the Hybrid + Reranker retrieval pipeline.
    """
    embeddings = get_embedding_model()
    
    # ----------------------------------------------------------------
    # 1. DENSE RETRIEVER (Vector/Semantic Search)
    # ----------------------------------------------------------------
    print("Initializing Vector Database (Chroma)...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    # Fetch top 10 semantic matches
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # ----------------------------------------------------------------
    # 2. SPARSE RETRIEVER (Keyword/Lexical Search)
    # ----------------------------------------------------------------
    print("Initializing BM25 Keyword Search...")
    sparse_retriever = BM25Retriever.from_documents(chunks)
    sparse_retriever.k = 10 # Fetch top 10 keyword matches

    # ----------------------------------------------------------------
    # 3. HYBRID FUSION
    # ----------------------------------------------------------------
    print("Fusing retrievers...")
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.5, 0.5] # Give equal importance to meaning and exact keywords
    )

    # ----------------------------------------------------------------
    # 4. CROSS-ENCODER RERANKER
    # ----------------------------------------------------------------
    print("Adding Cross-Encoder Reranker (Flashrank)...")
    compressor = FlashrankRerank(top_n=3) # Only pass the final absolute best 3 chunks to the LLM
    
    advanced_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid_retriever
    )
    
    print("Advanced Retrieval Pipeline Ready.")
    return advanced_retriever

# --- Quick Test Block ---
if __name__ == "__main__":
    from ingestion import load_and_chunk_pdfs
    
    # 1. Load the data using the script we wrote in Step 2
    test_chunks = load_and_chunk_pdfs("../data")
    
    if test_chunks:
        # 2. Build the retrieval pipeline
        retriever = build_advanced_retriever(test_chunks)
        
        # 3. Test it with a query
        test_query = "What is the main objective discussed in the document?"
        print(f"\nTesting Query: '{test_query}'\n")
        
        results = retriever.invoke(test_query)
        
        print(f"--- Top {len(results)} Reranked Results ---")
        for i, doc in enumerate(results):
            print(f"\nResult {i+1} (Relevance Score: {doc.metadata.get('relevance_score', 'N/A')}):")
            print(doc.page_content[:200] + "...\n")