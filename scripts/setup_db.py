import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# 1. Folder containing your uploaded PDFs
# Ensure you save your 80+ pages of PDFs in the 'data' folder
DATA_PATH = "./data"
PERSIST_DIR = "./vector_db"

def create_vector_db():
    documents = []
    
    print("Reading PDFs from /data...")
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())

    # 2. Split text into chunks with overlap for better context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Load FREE local embedding model
    # Downloads the model once; subsequent runs are offline
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Create and persist the vector database
    print(f"Creating vector database at {PERSIST_DIR}...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    
    print("✅ Local vector database created successfully!")

if __name__ == "__main__":
    create_vector_db()