import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.v1.router import api_router
from app.core.dependencies import set_rag_pipeline
from src.pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

def _cors_origins() -> list[str]:
    raw_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

def _cors_allow_credentials(origins: list[str]) -> bool:
    requested = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() in ("true", "1", "yes")
    return requested and origins != ["*"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler của FastAPI:
    Khởi động RAG Pipeline một lần duy nhất lúc startup, nạp sẵn các mô hình 
    Embedding & Reranker lên CPU/GPU để tối ưu hóa thời gian phản hồi (latency).
    """
    logger.info("Đang khởi tạo thực thể RAGPipeline trung tâm lúc khởi động máy chủ...")
    try:
        pipeline = RAGPipeline()
        set_rag_pipeline(pipeline)
        logger.info("[SUCCESS] Đã khởi tạo và nạp thành công RAGPipeline vào bộ nhớ!")
    except Exception as e:
        logger.error(f"[CRITICAL] Thất bại khi khởi tạo RAGPipeline: {e}")
        
    yield
    
    logger.info("Đang giải phóng các tài nguyên máy chủ RAG...")

# Khởi tạo FastAPI với Lifespan
app = FastAPI(
    title="ViLaborRAG API Server",
    description="REST API Server phục vụ Hỏi đáp Bộ luật Lao động Việt Nam 2019 có trích dẫn nguồn chuẩn xác.",
    version="1.0.0",
    lifespan=lifespan
)

cors_origins = _cors_origins()

# Cấu hình CORS cho phép gọi API từ các domain khác (nếu tích hợp với frontend riêng)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=_cors_allow_credentials(cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'"
    )
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    return response

# Đăng ký routes API
app.include_router(api_router, prefix="/api/v1")

# Mount thư mục chứa mã nguồn tĩnh (HTML/CSS/JS) của Frontend ở root
# Giúp người dùng truy cập trực tiếp URL của server (ví dụ: http://localhost:8000) để vào giao diện
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
