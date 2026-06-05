from fastapi import APIRouter
from app.schemas.system_schema import SystemStatusResponse
from src.config import (
    APP_ENV,
    CORPUS_JSONL_PATH,
    GEMINI_API_KEY,
    OFFLINE_EMBEDDING_DIR,
    OFFLINE_RERANKER_DIR,
    PROJECT_ROOT,
    SETTINGS,
    VECTOR_DB_DIR,
)

router = APIRouter()

@router.get("/status", response_model=SystemStatusResponse, summary="Lấy trạng thái và tham số hệ thống RAG")
def get_system_status():
    """
    Trả về toàn bộ thông số cấu hình hoạt động hiện có của hệ thống ViLaborRAG:
    - Trạng thái API Key.
    - Cấu hình mô hình ngôn ngữ & embedding.
    - Các ngưỡng điểm tương đồng của 4 chiến lược tìm kiếm.
    - Câu tuyên bố miễn trừ trách nhiệm pháp lý mặc định.
    """
    return {
        "status": "ready",
        "api_key_configured": bool(GEMINI_API_KEY),
        "models": SETTINGS.get("models", {}),
        "thresholds": SETTINGS.get("retrieval", {}).get("thresholds", {}),
        "disclaimer": SETTINGS.get("generation", {}).get("default_disclaimer", "")
    }

@router.get("/healthz", summary="Process health check nhẹ")
def get_healthz():
    return {"status": "ok"}

@router.get("/readyz", summary="Kiểm tra readiness của các artifact RAG bắt buộc")
def get_readyz():
    legacy_index = PROJECT_ROOT / "data" / "processed" / "faiss_index.bin"
    legacy_mapping = PROJECT_ROOT / "data" / "processed" / "faiss_mapping.json"
    configured_index = VECTOR_DB_DIR / "faiss_index.bin"
    configured_mapping = VECTOR_DB_DIR / "faiss_mapping.json"

    checks = {
        "corpus": CORPUS_JSONL_PATH.exists(),
        "vector_index": (
            configured_index.exists() and configured_mapping.exists()
        ) or (
            legacy_index.exists() and legacy_mapping.exists()
        ),
        "embedding_model": OFFLINE_EMBEDDING_DIR.exists() or bool(SETTINGS.get("models", {}).get("embedding")),
        "reranker_model": OFFLINE_RERANKER_DIR.exists() or bool(SETTINGS.get("models", {}).get("reranker")),
        "gemini_api_key": bool(GEMINI_API_KEY) or APP_ENV not in ("production", "prod"),
    }

    ready = all(checks.values())
    return {
        "status": "ready" if ready else "degraded",
        "checks": checks,
    }
