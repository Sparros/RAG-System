from fastapi import FastAPI
from app.api.router import api_router

app = FastAPI(
    title="RAG System",
    version="1.0.0"
)

app.include_router(api_router)

@app.get("/")
def root():
    return {"message": "RAG API is running"}
