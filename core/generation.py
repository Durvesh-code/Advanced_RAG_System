import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Load environment variables from the .env file
load_dotenv()

def get_llm():
    """
    Initializes the OpenAI LLM. 
    Using gpt-4o or gpt-3.5-turbo.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

    print("Initializing LLM...")
    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o",              # You can change to gpt-3.5-turbo or another openai model
        temperature=0.1,             # Low temperature to prevent hallucinations
        max_tokens=1024
    )
    return llm

def generate_answer(query: str, retrieved_docs: list):
    """
    Takes the user query and the reranked documents, formats them, and generates an answer.
    """
    llm = get_llm()
    
    # 1. Extract the text from the document objects
    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    
    if not context_text:
        return "I couldn't find any relevant information in the provided documents to answer your question."

    # 2. Define a strict, professional prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template="""You are an expert AI assistant. Use ONLY the following retrieved context to answer the user's question. 
        If the answer is not contained within the context, explicitly state that you do not know. Do not hallucinate or use outside knowledge.
        
        Context:
        {context}
        
        Question: 
        {query}
        
        Answer:"""
    )
    
    # 3. Format the prompt and invoke the model
    print("Generating response...")
    chain = prompt_template | llm
    
    response = chain.invoke({"context": context_text, "query": query})
    
    return response.content

# --- Quick Test Block ---
if __name__ == "__main__":
    from langchain_core.documents import Document
    
    # Mock documents so we can test the LLM without running the heavy GPU retrieval pipeline
    mock_docs = [
        Document(page_content="The advanced hybrid RAG system utilizes both dense and sparse retrieval methods, fused together with Reciprocal Rank Fusion, before passing through a cross-encoder reranker."),
        Document(page_content="Using an RTX series GPU for local embedding generation drastically reduces the ingestion time compared to CPU-bound processing.")
    ]
    
    test_query = "How does the hybrid RAG system work?"
    
    print(f"\nTesting Query: '{test_query}'\n")
    final_answer = generate_answer(test_query, mock_docs)
    
    print("\n--- Final Generated Answer ---")
    print(final_answer)