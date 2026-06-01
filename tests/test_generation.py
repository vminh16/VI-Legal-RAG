import os
import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel
from src.config import GEMINI_API_KEY

# Import classes mong muốn (chưa tồn tại)
# Sử dụng try/except để pytest có thể thu thập lỗi import một cách chuẩn xác theo quy trình TDD
try:
    from src.generation.generator import GeminiGenerator, RAGResponse, Citation
except ImportError:
    GeminiGenerator = None
    RAGResponse = None
    Citation = None

@pytest.fixture
def sample_retrieved_chunks():
    """Danh sách chunk giả lập trả ra từ retriever."""
    return [
        {
            "chunk_id": "bll2019_dieu_24_khoan_1",
            "document_title": "Bộ luật Lao động 2019",
            "law_number": "45/2019/QH14",
            "chapter": "Chương III",
            "section": "Mục 1",
            "article": "Điều 24",
            "clause": "1",
            "point": None,
            "title": "Thử việc",
            "text": "Người sử dụng lao động và người lao động có thỏa thuận về việc thử việc, quyền và nghĩa vụ của hai bên trong thời gian thử việc.",
            "source_url": "https://test.url"
        }
    ]

def test_classes_exist():
    """Kiểm tra xem các lớp phục vụ Generation đã được tạo chưa (Bước 1 TDD mong đợi Fail)."""
    assert GeminiGenerator is not None, "Chưa lập trình lớp GeminiGenerator!"
    assert RAGResponse is not None, "Chưa lập trình lớp RAGResponse!"
    assert Citation is not None, "Chưa lập trình lớp Citation!"

def test_generator_offline_mocking(sample_retrieved_chunks):
    """Kiểm tra hoạt động sinh câu trả lời trong chế độ Mocking (môi trường Offline/CI)."""
    if GeminiGenerator is None:
        pytest.fail("Chưa lập trình lớp GeminiGenerator!")
        
    generator = GeminiGenerator(api_key="mock_key")
    
    # Dữ liệu giả lập khớp chính xác tuyệt đối với Pydantic RAGResponse schema
    mock_json_response = {
        "answer": "Theo quy định tại Điều 24 Bộ luật Lao động 2019, người lao động và người sử dụng lao động có thỏa thuận về việc thử việc [1].",
        "citations": [
            {
                "citation_id": 1,
                "article": "Điều 24",
                "clause": "1",
                "title": "Thử việc",
                "source_url": "https://test.url",
                "evidence": "Người sử dụng lao động và người lao động có thỏa thuận về việc thử việc"
            }
        ],
        "confidence": 0.95
    }
    
    # Mocking SDK call của google-genai
    # SDK genai gọi: client.models.generate_content(...)
    # Kết quả trả về là một object có trường .text chứa chuỗi JSON
    mock_response_obj = MagicMock()
    mock_response_obj.text = json_str = "".join(str(mock_json_response).replace("'", '"'))
    
    with patch("google.genai.Client") as MockClient:
        # Giả lập client.models.generate_content trả về mock_response_obj
        mock_client_instance = MockClient.return_value
        mock_client_instance.models.generate_content.return_value = mock_response_obj
        
        # Khởi tạo lại generator với Mock Client
        generator.client = mock_client_instance
        
        response = generator.generate_answer(
            query="thử việc là gì",
            retrieved_chunks=sample_retrieved_chunks
        )
        
        # Xác minh kết quả
        assert isinstance(response, RAGResponse)
        assert "Điều 24" in response.answer
        assert len(response.citations) == 1
        assert response.citations[0].article == "Điều 24"
        assert response.confidence == 0.95

@pytest.mark.skipif(not GEMINI_API_KEY, reason="Không tìm thấy GEMINI_API_KEY trong môi trường (.env) để chạy test online thực tế.")
def test_generator_online_actual(sample_retrieved_chunks):
    """Kiểm tra gọi API thực tế tới Gemini 2.5 Flash khi có API Key online."""
    if GeminiGenerator is None:
        pytest.fail("Chưa lập trình lớp GeminiGenerator!")
        
    generator = GeminiGenerator()
    response = generator.generate_answer(
        query="thử việc là gì theo Bộ luật Lao động 2019?",
        retrieved_chunks=sample_retrieved_chunks
    )
    
    assert isinstance(response, RAGResponse)
    assert response.confidence > 0.5
    assert len(response.citations) > 0
    assert "Điều 24" in response.answer or "thử việc" in response.answer.lower()
