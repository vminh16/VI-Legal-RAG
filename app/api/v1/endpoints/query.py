import logging
import os
import time
from fastapi import APIRouter, Depends, HTTPException, Request
from app.schemas.query_schema import QueryRequest, QueryResponse
from app.core.dependencies import get_rag_pipeline
from src.pipeline.rag_pipeline import RAGPipeline

router = APIRouter()
logger = logging.getLogger(__name__)
_rate_limit_buckets: dict[str, list[float]] = {}

class BypassRefusalDetector:
    """Detector giả lập để bỏ qua các tầng từ chối phục vụ mục đích debug và kiểm thử."""
    class MockRefusalJudgment:
        def __init__(self):
            self.refuse = False
            self.reason = "Bypassed."
            self.category = "in_scope"

    def detect_query_refusal(self, query: str):
        return self.MockRefusalJudgment()

    def detect_retrieval_refusal(self, retrieved_chunks: list, strategy: str):
        return {"refuse": False, "category": "in_scope"}

    def detect_output_refusal(self, response, verification_report: dict):
        return {"refuse": False, "category": "in_scope"}

def _env_flag(name: str) -> bool:
    return os.getenv(name, "False").lower() in ("true", "1", "yes")

def _debug_bypass_enabled() -> bool:
    app_env = os.getenv("APP_ENV", "development").lower().strip()
    return app_env not in ("production", "prod") and _env_flag("ENABLE_DEBUG_ENDPOINTS")

def _enforce_auth(request: Request) -> None:
    expected_token = os.getenv("API_AUTH_TOKEN", "")
    if not expected_token:
        return

    auth_header = request.headers.get("Authorization", "")
    expected_header = f"Bearer {expected_token}"
    if auth_header != expected_header:
        raise HTTPException(status_code=401, detail="Thiếu hoặc sai Bearer token truy cập API.")

def _enforce_rate_limit(request: Request) -> None:
    app_env = os.getenv("APP_ENV", "development").lower().strip()
    default_limit = "60" if app_env in ("production", "prod") else "0"
    limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", default_limit))
    if limit <= 0:
        return

    client_host = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window_start = now - 60.0
    bucket = [stamp for stamp in _rate_limit_buckets.get(client_host, []) if stamp >= window_start]
    if len(bucket) >= limit:
        _rate_limit_buckets[client_host] = bucket
        raise HTTPException(status_code=429, detail="Vượt quá số lượng truy vấn cho phép trong 60 giây.")

    bucket.append(now)
    _rate_limit_buckets[client_host] = bucket

@router.post("", response_model=QueryResponse, summary="Thực hiện hỏi đáp RAG có trích dẫn và hậu kiểm")
def query_rag(request: QueryRequest, http_request: Request, pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    """
    Tiếp nhận câu hỏi pháp lý tiếng Việt và xử lý qua 10 Thành phần của RAG Pipeline:
    1. Tiền kiểm ý định (Tầng 1 Refusal).
    2. Truy hồi đa chiến lược (BM25, Dense, Hybrid, Rerank).
    3. Ngưỡng tương đồng truy hồi (Tầng 2 Refusal).
    4. Gộp nhóm ngữ cảnh & Format nguồn.
    5. Sinh câu trả lời & Disclaimer.
    6. Kiểm định tĩnh & động (Citation & Faithfulness Checkers).
    7. Quyết định lai Option C (Tầng 3 Refusal).
    """
    try:
        _enforce_auth(http_request)
        _enforce_rate_limit(http_request)

        if request.bypass_refusal:
            if not _debug_bypass_enabled():
                raise HTTPException(
                    status_code=403,
                    detail="Chế độ bypass refusal chỉ được phép trong môi trường debug nội bộ."
                )
            # Tạm thời ghi đè refusal detector bằng bộ phát hiện bỏ qua (bypass)
            orig_detector = getattr(pipeline, "refusal_detector", None)
            pipeline.refusal_detector = BypassRefusalDetector()
            try:
                result = pipeline.answer_question(
                    query=request.query,
                    strategy=request.strategy,
                    top_k=request.top_k
                )
            finally:
                # Đảm bảo luôn khôi phục detector ban đầu sau khi chạy xong
                pipeline.refusal_detector = orig_detector
        else:
            result = pipeline.answer_question(
                query=request.query,
                strategy=request.strategy,
                top_k=request.top_k
            )
        logger.info("RAG query processed: strategy=%s top_k=%s refused=%s", request.strategy, request.top_k, result.get("refused"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi hệ thống khi xử lý câu hỏi qua RAG Pipeline: {str(e)}"
        )
