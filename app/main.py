from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .engine import ComplianceEngine

app = FastAPI()
engine = ComplianceEngine()
app.mount("/static", StaticFiles(directory="static"), name="static")

class ProductQuery(BaseModel):
    name: str

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html") as f:
        return f.read()

@app.post("/api/check")
async def check_compliance(query: ProductQuery):
    result = engine.search(query.name)
    return result