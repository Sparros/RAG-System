# app/api/router.py
from fastapi import APIRouter
from app.api.endpoints import health
from app.api.endpoints import query
from app.api.endpoints import ingest

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
api_router.include_router(ingest.router, prefix="/documents", tags=["Ingest"])

