import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_PATH = "./data"
PERSIST_DIR = "./vector_db"

def create_vector_db():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created {DATA_PATH} folder. Please put your PDFs there.")
        return

    documents = []
    print("Reading PDFs from /data...")
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            try:
                loader = PyPDFLoader(os.path.join(DATA_PATH, file))
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file}: {e}")

    # Optimal splitting for regulatory documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # 100% FREE Local Embeddings
    print("Loading embedding model (BGE-small)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )

    print(f"Creating vector database at {PERSIST_DIR}...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print("✅ Local vector database created successfully!")

if __name__ == "__main__":
    create_vector_db()