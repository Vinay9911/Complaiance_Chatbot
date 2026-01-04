import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .engine import ComplianceEngine

app = FastAPI()
engine = ComplianceEngine()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class ProductQuery(BaseModel):
    name: str

@app.get("/", response_class=FileResponse)
async def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h2>Error: static/index.html not found.</h2>")

@app.post("/api/check")
async def check_compliance(query: ProductQuery):
    try:
        result = engine.search(query.name)
        return result
    except Exception as e:
        return {"is_known": False, "answer": f"System Error: {str(e)}"}