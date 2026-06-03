import pytest
from unittest.mock import MagicMock, patch
from src.generation.generator import RAGResponse, Citation

try:
    from src.verification.refusal_detector import RefusalDetector, RefusalJudgment
except ImportError:
    RefusalDetector = None
    RefusalJudgment = None

def test_refusal_classes_exist():
    """Xác nhận lớp RefusalDetector và RefusalJudgment đã được định nghĩa."""
    assert RefusalDetector is not None, "Chưa lập trình lớp RefusalDetector!"
    assert RefusalJudgment is not None, "Chưa lập trình lớp RefusalJudgment!"

def test_query_refusal_out_of_scope_cooking():
    """Tầng 1: Kiểm tra từ chối câu hỏi nấu ăn (out_of_scope)."""
    if RefusalDetector is None:
        pytest.fail("Chưa lập trình lớp RefusalDetector!")
        
    detector = RefusalDetector(api_key="mock_key")
    
    mock_response_obj = MagicMock()
    mock_response_obj.text = '{"refuse": true, "reason": "Rất tiếc, câu hỏi về nấu ăn nằm ngoài phạm vi Bộ luật Lao động 2019.", "category": "out_of_scope"}'
    
    with patch("google.genai.Client") as MockClient:
        mock_client_instance = MockClient.return_value
        mock_client_instance.models.generate_content.return_value = mock_response_obj
        detector.client = mock_client_instance
        
        judgment = detector.detect_query_refusal("Làm thế nào để nấu phở bò ngon?")
        assert judgment.refuse is True
        assert judgment.category == "out_of_scope"
        assert "nấu ăn" in judgment.reason

def test_query_refusal_administrative_fine():
    """Tầng 1: Kiểm tra từ chối câu hỏi mức phạt hành chính."""
    if RefusalDetector is None:
        pytest.fail("Chưa lập trình lớp RefusalDetector!")
        
    detector = RefusalDetector(api_key="mock_key")
    
    mock_response_obj = MagicMock()
    mock_response_obj.text = '{"refuse": true, "reason": "Bộ luật Lao động 2019 chỉ quy định khung hành vi, mức xử phạt hành chính cụ thể nằm trong Nghị định 12/2022/NĐ-CP.", "category": "administrative_fine"}'
    
    with patch("google.genai.Client") as MockClient:
        mock_client_instance = MockClient.return_value
        mock_client_instance.models.generate_content.return_value = mock_response_obj
        detector.client = mock_client_instance
        
        judgment = detector.detect_query_refusal("Công ty không đóng bảo hiểm bị phạt bao nhiêu tiền?")
        assert judgment.refuse is True
        assert judgment.category == "administrative_fine"
        assert "Nghị định 12/2022" in judgment.reason

def test_query_refusal_civil_law():
    """Tầng 1: Kiểm tra từ chối câu hỏi thuộc luật khác (Dân sự/Ly hôn)."""
    if RefusalDetector is None:
        pytest.fail("Chưa lập trình lớp RefusalDetector!")
        
    detector = RefusalDetector(api_key="mock_key")
    
    mock_response_obj = MagicMock()
    mock_response_obj.text = '{"refuse": true, "reason": "Câu hỏi thuộc lĩnh vực hôn nhân gia đình (Luật Hôn nhân và Gia đình), nằm ngoài phạm vi Bộ luật Lao động 2019.", "category": "out_of_scope"}'
    
    with patch("google.genai.Client") as MockClient:
        mock_client_instance = MockClient.return_value
        mock_client_instance.models.generate_content.return_value = mock_response_obj
        detector.client = mock_client_instance
        
        judgment = detector.detect_query_refusal("Làm thủ tục ly hôn đơn phương như thế nào?")
        assert judgment.refuse is True
        assert judgment.category == "out_of_scope"

def test_query_refusal_in_scope():
    """Tầng 1: Kiểm tra chấp nhận câu hỏi hợp lệ (Hỏi về thời giờ làm việc)."""
    if RefusalDetector is None:
        pytest.fail("Chưa lập trình lớp RefusalDetector!")
        
    detector = RefusalDetector(api_key="mock_key")
    
    mock_response_obj = MagicMock()
    mock_response_obj.text = '{"refuse": false, "reason": "Câu hỏi hoàn toàn thuộc phạm vi quy định của Bộ luật Lao động 2019.", "category": "in_scope"}'
    
    with patch("google.genai.Client") as MockClient:
        mock_client_instance = MockClient.return_value
        mock_client_instance.models.generate_content.return_value = mock_response_obj
        detector.client = mock_client_instance
        
        judgment = detector.detect_query_refusal("Thời giờ làm việc thử việc của lao động chuyên môn tối đa bao nhiêu ngày?")
        assert judgment.refuse is False
        assert judgment.category == "in_scope"

def test_retrieval_refusal_empty():
    """Tầng 2: Kiểm tra từ chối khi danh sách chunks rỗng."""
    if RefusalDetector is None:
        pytest.fail("Chưa lập trình lớp RefusalDetector!")
        
    detector = RefusalDetector(api_key="mock_key")
    
    judgment = detector.detect_retrieval_refusal([], strategy="dense")
    assert judgment["refuse"] is True
    assert judgment["category"] == "no_relevant_context"

def test_retrieval_refusal_low_score_dense():
    """Tầng 2: Kiểm tra từ chối khi điểm tương đồng quá thấp đối với Dense retrieval (< 0.35)."""
    if RefusalDetector is None:
        pytest.fail("Chưa lập trình lớp RefusalDetector!")
        
    detector = RefusalDetector(api_key="mock_key")
    
    low_score_chunks = [
        {"chunk_id": "test_chunk", "text": "Văn bản rác", "score": 0.25}  # Dưới 0.35
    ]
    
    judgment = detector.detect_retrieval_refusal(low_score_chunks, strategy="dense")
    assert judgment["refuse"] is True
    assert judgment["category"] == "no_relevant_context"
    assert "tương đồng" in judgment["reason"]

def test_retrieval_refusal_high_score_dense():
    """Tầng 2: Chấp nhận khi điểm tương đồng hợp lệ đối với Dense retrieval (>= 0.35)."""
    if RefusalDetector is None:
        pytest.fail("Chưa lập trình lớp RefusalDetector!")
        
    detector = RefusalDetector(api_key="mock_key")
    
    high_score_chunks = [
        {"chunk_id": "test_chunk", "text": "Điều 24 quy định thử việc...", "score": 0.65}
    ]
    
    judgment = detector.detect_retrieval_refusal(high_score_chunks, strategy="dense")
    assert judgment["refuse"] is False

def test_output_refusal_low_confidence():
    """Tầng 3: Kiểm tra từ chối khi độ tự tin generator hoặc báo cáo hậu kiểm có lỗi nặng."""
    if RefusalDetector is None:
        pytest.fail("Chưa lập trình lớp RefusalDetector!")
        
    detector = RefusalDetector(api_key="mock_key")
    
    # RAG response có độ tự tin 0.0 (chủ động nhận diện thiếu nguồn)
    low_conf_response = RAGResponse(
        answer="Dựa trên Bộ luật...",
        citations=[],
        confidence=0.0
    )
    
    judgment = detector.detect_output_refusal(low_conf_response, verification_report={"is_valid": True, "errors": []})
    assert judgment["refuse"] is True
    assert judgment["category"] == "generation_unconfident"
    
    # RAG response có lỗi bịa luật nghiêm trọng từ bộ checker tĩnh
    error_response = RAGResponse(
        answer="Tôi chắc chắn bạn sẽ thắng...",
        citations=[],
        confidence=0.90
    )
    report_with_errors = {
        "is_valid": False,
        "errors": ["citation_not_found_in_corpus"]  # Lỗi nghiêm trọng
    }
    
    judgment = detector.detect_output_refusal(error_response, verification_report=report_with_errors)
    assert judgment["refuse"] is True
    assert judgment["category"] == "critical_validation_error"
