from fastapi import APIRouter
from app.api.v1.endpoints import query, system

api_router = APIRouter()

# Đăng ký các endpoints v1
api_router.include_router(query.router, prefix="/query", tags=["RAG Querying"])
api_router.include_router(system.router, prefix="/system", tags=["System Status"])
