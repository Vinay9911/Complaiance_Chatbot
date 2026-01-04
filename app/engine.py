import json
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class ComplianceEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_db = Chroma(persist_directory="./vector_db", embedding_function=self.embeddings)
        with open("app/master_map.json", "r") as f:
            self.master_map = json.load(f)

    def search(self, product_name):
        # 1. Check Master Map (Structured)
        product_info = None
        for p in self.master_map["products"]:
            if any(k in product_name.lower() for k in p["keywords"]):
                product_info = p
                break
        
        # 2. Query Vector DB (Unstructured)
        query_text = f"Registration procedure and fees for {product_info['standard'] if product_info else product_name}"
        docs = self.vector_db.similarity_search(query_text, k=2)
        details = [doc.page_content for doc in docs]

        return {
            "is_known": product_info is not None,
            "rule": product_info,
            "manual_details": details
        }