"""
Document loading and vector database creation script.
Loads PDF and TXT files from the docs/ folder, splits them, and saves them to ChromaDB.
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def load_documents(docs_dir: str = "docs"):
    """Loads all PDF and TXT files from the docs/ folder."""
    documents = []
    docs_path = Path(docs_dir)
    
    if not docs_path.exists():
        raise FileNotFoundError(f"Folder {docs_dir} not found!")
    
    # Load PDF files
    pdf_files = list(docs_path.glob("*.pdf"))
    for pdf_file in pdf_files:
        print(f"Loading PDF: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())
    
    # Load TXT files
    txt_files = list(docs_path.glob("*.txt"))
    for txt_file in txt_files:
        print(f"Loading TXT: {txt_file.name}")
        loader = TextLoader(str(txt_file), encoding='utf-8')
        documents.extend(loader.load())
    
    if not documents:
        raise ValueError(f"No PDF or TXT files found in {docs_dir} folder!")
    
    print(f"Total {len(documents)} documents loaded.")
    return documents

def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Splits documents into chunks of specified size."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Documents split into {len(chunks)} chunks.")
    return chunks

def create_vectorstore(chunks, persist_directory: str = "./chroma_db", api_key: str = None):
    """Creates embeddings and saves them to ChromaDB."""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API Key required! Please set the OPENAI_API_KEY environment variable or provide the api_key parameter.")
    
    # Create OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Save to ChromaDB
    print(f"Creating embeddings and saving to {persist_directory} folder...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Vector database successfully created: {persist_directory}")
    return vectorstore

def main():
    """Main function: Executes all operations in sequence."""
    print("=" * 50)
    print("Document Loading and Vector Database Creation")
    print("=" * 50)
    
    # --- ADDED SECTION: Request API Key from User ---
    import getpass
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  API Key not found in environment variables.")
        # getpass prevents the password from being displayed on screen (not even asterisks, secure)
        # If getpass causes issues, you can also use plain input().
        try:
            api_key = getpass.getpass("👉 Please paste your OpenAI API Key and press Enter: ")
        except:
            api_key = input("👉 Please paste your OpenAI API Key and press Enter: ")
            
    if not api_key or not api_key.startswith("sk-"):
        print("\n❌ Invalid or empty API Key! Terminating program.")
        return
    # ----------------------------------------------------

    try:
        # 1. Load documents
        documents = load_documents("docs")
        
        # 2. Split documents into chunks
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
        
        # 3. Create vector database (sending API Key here)
        vectorstore = create_vectorstore(chunks, persist_directory="./chroma_db", api_key=api_key)
        
        print("\n✅ Operation completed successfully!")
        print(f"📁 Vector database created: ./chroma_db")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()

