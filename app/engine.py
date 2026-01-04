import json
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage 
from google import genai
from google.genai import types

load_dotenv()

class ComplianceEngine:
    def __init__(self):
        # 1. Setup Local Embeddings and Groq (Llama 3.3)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            temperature=0, 
            model_name="llama-3.3-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # 2. Setup New Gemini Client (google-genai)
        self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # 3. Load Databases
        if os.path.exists("./vector_db"):
            self.vector_db = Chroma(persist_directory="./vector_db", embedding_function=self.embeddings)
        else:
            self.vector_db = None

        with open("app/master_map.json", "r") as f:
            self.master_map = json.load(f)

    
        
    def classify_product(self, query):
        """Uses domain reasoning to identify ANY electronic/electrical product."""
        
        # Get your specific categories for direct mapping
        known_categories = [p["keywords"][0] for p in self.master_map["products"]]
        
        system_prompt = f"""
        You are an expert in International Trade and Product Classification (HS Codes).
        Your task is to classify the user's product into one of three statuses:
        
        1. [CATEGORY_NAME]: If the product clearly belongs to one of these specific local categories: {', '.join(known_categories)}.
        
        2. 'unknown_electronic': If the product is NOT in the list above, but IS an electronic, electrical, or electromechanical device. 
           This includes:
           - Development boards (Arduino, Raspberry Pi, ESP32)
           - IoT Devices and Sensors (Smart keys, Zigbee hubs, PIR sensors)
           - Industrial/Lab equipment (Oscilloscopes, PLC modules)
           - PC Components (Graphics cards, RAM, Motherboards)
           - Any item that requires a battery, plug, or carries an electric signal.
        
        3. 'out_of_scope': If the item is purely mechanical, biological, or chemical (e.g., 'wooden chair', 'apple', 'paper book', 'cotton shirt').

        RESPONSE FORMAT: Only return the status string. No explanation.
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User Query: {query}")
            ])
            
            category = response.content.strip().lower()
            
            # Case 1: Out of Scope
            if "out_of_scope" in category:
                return "out_of_scope", None
            
            # Case 2: Known locally
            for p in self.master_map["products"]:
                if any(k.lower() in category for k in p["keywords"]):
                    return "known", p
            
            # Case 3: Recognized as electronic but missing from local JSON
            return "unknown_electronic", None
            
        except Exception as e:
            print(f"Classification error: {e}")
            return "out_of_scope", None

    def web_search_compliance(self, query):
        """Uses Gemini with live Google Search grounding to find real-time Indian compliance data."""
        prompt = f"Provide a detailed Indian compliance roadmap (BIS, WPC, E-Waste) for: {query}."
        
        # Enabling Google Search tool in the new SDK
        response = self.gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        return response.text

    def search(self, user_query):
        status, product_info = self.classify_product(user_query)
        
        # Case 1: Non-electronic items
        if status == "out_of_scope":
            return {
                "is_known": False, 
                "answer": "I only provide compliance roadmaps for electronic and electrical products in India. This item appears to be out of scope."
            }

        # Case 2: Known electronic item (Standard RAG)
        if status == "known":
            search_term = f"Requirements and fees for {product_info['standard']}"
            docs = self.vector_db.similarity_search(search_term, k=5) if self.vector_db else []
            context = "\n---\n".join([d.page_content for d in docs])
            
            rag_prompt = f"""
            Expert Indian Compliance response for {user_query}. 
            Standard: {product_info['standard']}
            Authority: {product_info['authority']}
            Context: {context}
            """
            ai_response = self.llm.invoke(rag_prompt)
            return {"is_known": True, "rule": product_info, "answer": ai_response.content}

        # Case 3: Unknown electronic item (Fallback to Gemini Web Search)
        if status == "unknown_electronic":
            fallback_intro = "I do not have specific information about this product in my local database. I'll try searching online for you...\n\n"
            online_data = self.web_search_compliance(user_query)
            return {"is_known": False, "answer": fallback_intro + online_data}