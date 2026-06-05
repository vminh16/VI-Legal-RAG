import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.core.dependencies import get_rag_pipeline
from src.pipeline.rag_pipeline import RAGPipeline

client = TestClient(app)

@pytest.fixture
def mock_pipeline():
    mock = MagicMock(spec=RAGPipeline)
    app.dependency_overrides[get_rag_pipeline] = lambda: mock
    yield mock
    app.dependency_overrides.clear()

def test_system_status_endpoint():
    """Kiểm tra GET /api/v1/system/status trả về 200 và thông tin hệ thống."""
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ready"
    assert "api_key_configured" in data
    assert "models" in data
    assert "thresholds" in data
    assert "disclaimer" in data

def test_health_and_readiness_endpoints():
    """Kiểm tra health/readiness tách biệt trạng thái process và phụ thuộc runtime."""
    health_response = client.get("/api/v1/system/healthz")
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"

    ready_response = client.get("/api/v1/system/readyz")
    assert ready_response.status_code == 200
    data = ready_response.json()
    assert "status" in data
    assert "checks" in data
    assert "corpus" in data["checks"]
    assert "vector_index" in data["checks"]

def test_cors_wildcard_does_not_allow_credentials():
    """Production CORS không được kết hợp wildcard origin với credentials."""
    cors_middleware = next(
        middleware for middleware in app.user_middleware
        if middleware.cls.__name__ == "CORSMiddleware"
    )
    allow_origins = cors_middleware.kwargs.get("allow_origins")
    allow_credentials = cors_middleware.kwargs.get("allow_credentials")

    assert not (allow_origins == ["*"] and allow_credentials is True)

def test_static_response_includes_content_security_policy():
    """Static UI phải có CSP tối thiểu để giảm rủi ro XSS từ output LLM/corpus."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Content-Security-Policy" in response.headers

def test_query_rag_endpoint_success(mock_pipeline):
    """Kiểm tra POST /api/v1/query trả về kết quả truy vấn thành công từ pipeline."""
    mock_pipeline.answer_question.return_value = {
        "answer": "Theo Bộ luật Lao động 2019...",
        "citations": [
            {
                "citation_id": 1,
                "article": "Điều 90",
                "clause": "Khoản 1",
                "title": "Lương",
                "source_url": "https://example.com",
                "evidence": "Lương thử việc tối thiểu..."
            }
        ],
        "confidence": 0.95,
        "refused": False,
        "category": "in_scope",
        "retrieved_chunks": [
            {
                "chunk_id": "c1",
                "text": "Nội dung Điều 90...",
                "score": 0.85,
                "article": "Điều 90",
                "clause": "Khoản 1",
                "chapter": "Chương VI",
                "title": "Tiền lương"
            }
        ]
    }

    payload = {
        "query": "Lương thử việc bao nhiêu?",
        "strategy": "hybrid_rerank",
        "top_k": 5,
        "bypass_refusal": False
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Theo Bộ luật Lao động 2019..."
    assert data["refused"] is False
    assert len(data["citations"]) == 1
    assert data["citations"][0]["article"] == "Điều 90"
    assert data["confidence"] == 0.95

def test_query_rag_endpoint_bypass_refusal(monkeypatch, mock_pipeline):
    """Kiểm tra POST /api/v1/query có bypass_refusal=True sẽ tạm thời override detector."""
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("ENABLE_DEBUG_ENDPOINTS", "true")
    original_detector = MagicMock()
    mock_pipeline.refusal_detector = original_detector

    def side_effect(*args, **kwargs):
        # Kiểm tra xem detector có bị đổi sang BypassRefusalDetector trong quá trình xử lý không
        assert mock_pipeline.refusal_detector is not original_detector
        return {
            "answer": "Trả lời khi đã bypass",
            "citations": [],
            "confidence": 0.8,
            "refused": False,
            "category": "in_scope",
            "retrieved_chunks": []
        }
    mock_pipeline.answer_question.side_effect = side_effect

    payload = {
        "query": "Làm thế nào để nấu phở bò?",
        "strategy": "hybrid_rerank",
        "top_k": 5,
        "bypass_refusal": True
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    
    # Xác nhận mock_pipeline.answer_question được gọi
    mock_pipeline.answer_question.assert_called_once_with(
        query="Làm thế nào để nấu phở bò?",
        strategy="hybrid_rerank",
        top_k=5
    )
    
    # Xác nhận sau cuộc gọi, detector gốc được khôi phục
    assert mock_pipeline.refusal_detector is original_detector

def test_query_bypass_refusal_rejected_in_production(monkeypatch, mock_pipeline):
    """Không cho phép bypass refusal ở production dù frontend gửi cờ debug."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("ENABLE_DEBUG_ENDPOINTS", raising=False)

    payload = {
        "query": "Làm thế nào để nấu phở bò?",
        "strategy": "hybrid_rerank",
        "top_k": 5,
        "bypass_refusal": True
    }

    response = client.post("/api/v1/query", json=payload)

    assert response.status_code == 403
    mock_pipeline.answer_question.assert_not_called()

def test_query_requires_bearer_token_when_configured(monkeypatch, mock_pipeline):
    """Khi API_AUTH_TOKEN được cấu hình, query endpoint bắt buộc xác thực Bearer token."""
    monkeypatch.setenv("API_AUTH_TOKEN", "secret-token")
    mock_pipeline.answer_question.return_value = {
        "answer": "ok",
        "citations": [],
        "confidence": 0.0,
        "refused": True,
        "category": "test",
        "retrieved_chunks": []
    }
    payload = {
        "query": "Thử việc có được trả lương không?",
        "strategy": "hybrid_rerank",
        "top_k": 5,
        "bypass_refusal": False
    }

    missing_auth = client.post("/api/v1/query", json=payload)
    assert missing_auth.status_code == 401

    valid_auth = client.post(
        "/api/v1/query",
        json=payload,
        headers={"Authorization": "Bearer secret-token"}
    )
    assert valid_auth.status_code == 200
