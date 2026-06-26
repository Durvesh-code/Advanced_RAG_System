import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdfs(data_folder: str = "data", chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Loads all PDFs from the specified folder and splits them into manageable chunks.
    """
    all_docs = []
    
    # 1. Ensure the data directory exists
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"Created '{data_folder}' directory. Please drop some PDFs in there.")
        return []

    # 2. Find all PDF files
    pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDFs found in the '{data_folder}' folder.")
        return []

    print(f"Found {len(pdf_files)} PDFs. Loading...")
    
    # 3. Load documents using PyMuPDF
    for filename in pdf_files:
        file_path = os.path.join(data_folder, filename)
        loader = PyMuPDFLoader(file_path)
        all_docs.extend(loader.load())
        
    print(f"Loaded {len(all_docs)} total pages.")

    # 4. Split into semantic chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(all_docs)
    print(f"Successfully split documents into {len(chunks)} chunks.")
    
    return chunks

# --- Quick Test Block ---
# This block ONLY runs if you execute this specific file directly from the terminal.
if __name__ == "__main__":
    test_chunks = load_and_chunk_pdfs("../data") # Adjust path if running from inside the core folder
    
    if test_chunks:
        print("\n--- Sample Chunk Preview ---")
        print(f"{test_chunks[0].page_content[:250]}...")