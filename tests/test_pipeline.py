import pytest
from unittest.mock import MagicMock
from src.generation.generator import RAGResponse, Citation

try:
    from src.pipeline.rag_pipeline import RAGPipeline
except ImportError:
    RAGPipeline = None

def test_pipeline_classes_exist():
    """Kiểm tra lớp RAGPipeline có tồn tại (TDD Bước 1 mong đợi Thất bại)."""
    assert RAGPipeline is not None, "Chưa lập trình lớp RAGPipeline!"

def test_pipeline_pre_retrieval_refusal():
    """Tầng 1 Refusal: Ngắt luồng sớm khi phát hiện câu hỏi vi phạm tiền kiểm ý định."""
    if RAGPipeline is None:
        pytest.fail("Chưa lập trình lớp RAGPipeline!")
        
    mock_detector = MagicMock()
    mock_detector.detect_query_refusal.return_value.refuse = True
    mock_detector.detect_query_refusal.return_value.reason = "Không trả lời câu hỏi nấu ăn."
    mock_detector.detect_query_refusal.return_value.category = "out_of_scope"
    
    pipeline = RAGPipeline(
        api_key="mock_key",
        retrieval_pipeline=MagicMock(),
        generator=MagicMock(),
        refusal_detector=mock_detector
    )
    
    result = pipeline.answer_question("Làm thế nào để nấu phở bò?", strategy="dense")
    
    assert result["refused"] is True
    assert result["category"] == "out_of_scope"
    assert "nấu phở" in result["answer"] or "nấu ăn" in result["answer"]
    pipeline.retrieval_pipeline.retrieve.assert_not_called()
    pipeline.generator.generate_answer.assert_not_called()

def test_pipeline_post_retrieval_refusal():
    """Tầng 2 Refusal: Ngắt luồng khi điểm tương đồng của chunks thu về quá thấp."""
    if RAGPipeline is None:
        pytest.fail("Chưa lập trình lớp RAGPipeline!")
        
    # Mock RefusalDetector Tầng 1 cho qua (in_scope)
    mock_detector = MagicMock()
    mock_detector.detect_query_refusal.return_value.refuse = False
    mock_detector.detect_query_refusal.return_value.category = "in_scope"
    
    # Mock Retrieval thu về kết quả điểm tương đồng quá thấp
    mock_chunks = [{"chunk_id": "c1", "text": "Rác", "score": 0.15}]
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = mock_chunks
    
    # Mock RefusalDetector Tầng 2 kích hoạt từ chối
    mock_detector.detect_retrieval_refusal.return_value = {
        "refuse": True,
        "reason": "Điểm quá thấp.",
        "category": "no_relevant_context"
    }
    
    pipeline = RAGPipeline(
        api_key="mock_key",
        retrieval_pipeline=mock_retrieval,
        generator=MagicMock(),
        refusal_detector=mock_detector
    )
    
    result = pipeline.answer_question("Hợp đồng lao động là gì?", strategy="dense")
    
    assert result["refused"] is True
    assert result["category"] == "no_relevant_context"
    pipeline.generator.generate_answer.assert_not_called()

def test_pipeline_post_generation_critical_error_refusal():
    """Tầng 3 Refusal: Từ chối lai Option C khi hậu kiểm Checker phát hiện lỗi nặng."""
    if RAGPipeline is None:
        pytest.fail("Chưa lập trình lớp RAGPipeline!")
        
    # Cho qua Tầng 1 và Tầng 2
    mock_detector = MagicMock()
    mock_detector.detect_query_refusal.return_value.refuse = False
    mock_detector.detect_retrieval_refusal.return_value = {"refuse": False}
    
    # Mock Retrieval thu về chunks hợp lệ
    mock_chunks = [{"chunk_id": "c1", "text": "Điều 24 quy định thử việc...", "score": 0.85}]
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = mock_chunks
    
    # Mock Generator sinh câu trả lời có lỗi bịa luật nghiêm trọng
    mock_response = RAGResponse(
        answer="Bạn chắc chắn thắng kiện Điều 300 [1].",
        citations=[Citation(citation_id=1, article="Điều 300", clause=None, title="Bịa", source_url="https://test", evidence="bịa")],
        confidence=0.90
    )
    mock_generator = MagicMock()
    mock_generator.generate_answer.return_value = mock_response
    
    # Mock các Checkers phát hiện lỗi bịa luật tĩnh (fabricated citation)
    mock_citation_checker = MagicMock()
    mock_citation_checker.check_citations.return_value = {
        "is_valid": False,
        "errors": ["citation_not_found_in_corpus"]
    }
    mock_faithfulness_checker = MagicMock()
    mock_faithfulness_checker.check_faithfulness.return_value = {"is_faithful": True, "conflicts": []}
    mock_faithfulness_checker.check_disclaimer.return_value = {"has_disclaimer": True, "errors": []}
    
    # Mock RefusalDetector Tầng 3 nhận diện lỗi và kích hoạt từ chối lai
    mock_detector.detect_output_refusal.return_value = {
        "refuse": True,
        "reason": "Dựa trên Bộ luật Lao động 2019 và dữ liệu ngữ cảnh..., tôi xin phép từ chối trả lời.",
        "category": "critical_validation_error"
    }
    
    pipeline = RAGPipeline(
        api_key="mock_key",
        retrieval_pipeline=mock_retrieval,
        generator=mock_generator,
        citation_checker=mock_citation_checker,
        faithfulness_checker=mock_faithfulness_checker,
        refusal_detector=mock_detector
    )
    
    result = pipeline.answer_question("Thử việc bao lâu?", strategy="dense")
    
    assert result["refused"] is True
    assert result["category"] == "critical_validation_error"
    assert "từ chối" in result["answer"]

def test_pipeline_successful_flow():
    """Kiểm tra luồng RAG thành công hoàn mỹ (Option C không phát hiện lỗi nào)."""
    if RAGPipeline is None:
        pytest.fail("Chưa lập trình lớp RAGPipeline!")
        
    # Cho qua Tầng 1 và Tầng 2
    mock_detector = MagicMock()
    mock_detector.detect_query_refusal.return_value.refuse = False
    mock_detector.detect_retrieval_refusal.return_value = {"refuse": False}
    mock_detector.detect_output_refusal.return_value = {"refuse": False}
    
    # Mock Retrieval thu về chunks hợp lệ
    mock_chunks = [{"chunk_id": "c1", "text": "Điều 24 quy định thử việc...", "score": 0.85}]
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = mock_chunks
    
    # Mock Generator sinh câu trả lời hoàn hảo
    mock_response = RAGResponse(
        answer="Hai bên thỏa thuận về thử việc theo Điều 24 [1]. Lưu ý: Ý kiến tư vấn trên chỉ mang tính chất tham khảo dựa trên quy định của Bộ luật Lao động 2019 và dữ liệu ngữ cảnh hiện có tại thời điểm tra cứu.",
        citations=[Citation(citation_id=1, article="Điều 24", clause="1", title="Thử việc", source_url="https://test", evidence="thỏa thuận về thử việc")],
        confidence=0.95
    )
    mock_generator = MagicMock()
    mock_generator.generate_answer.return_value = mock_response
    
    # Mock các Checkers xác nhận xanh lá toàn bộ
    mock_citation_checker = MagicMock()
    mock_citation_checker.check_citations.return_value = {"is_valid": True, "errors": []}
    mock_faithfulness_checker = MagicMock()
    mock_faithfulness_checker.check_faithfulness.return_value = {"is_faithful": True, "conflicts": []}
    mock_faithfulness_checker.check_disclaimer.return_value = {"has_disclaimer": True, "errors": []}
    
    pipeline = RAGPipeline(
        api_key="mock_key",
        retrieval_pipeline=mock_retrieval,
        generator=mock_generator,
        citation_checker=mock_citation_checker,
        faithfulness_checker=mock_faithfulness_checker,
        refusal_detector=mock_detector
    )
    
    result = pipeline.answer_question("Thử việc bao lâu?", strategy="dense")
    
    assert result["refused"] is False
    assert "Điều 24" in result["answer"]
    assert len(result["citations"]) == 1
    assert result["confidence"] == 0.95

def test_pipeline_handles_empty_citations_without_crashing():
    """Regression: Generator có thể trả citations rỗng; pipeline không được nổ IndexError."""
    if RAGPipeline is None:
        pytest.fail("Chưa lập trình lớp RAGPipeline!")

    mock_detector = MagicMock()
    mock_detector.detect_query_refusal.return_value.refuse = False
    mock_detector.detect_retrieval_refusal.return_value = {"refuse": False}
    mock_detector.detect_output_refusal.return_value = {"refuse": False}

    mock_chunks = [{"chunk_id": "c1", "text": "Điều 24 quy định thử việc...", "score": 0.85}]
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = mock_chunks

    mock_response = RAGResponse(
        answer="Câu trả lời thiếu trích dẫn nhưng generator vẫn tự tin.",
        citations=[],
        confidence=0.90
    )
    mock_generator = MagicMock()
    mock_generator.generate_answer.return_value = mock_response

    mock_citation_checker = MagicMock()
    mock_citation_checker.check_citations.return_value = {"is_valid": True, "errors": []}
    mock_faithfulness_checker = MagicMock()
    mock_faithfulness_checker.check_faithfulness.return_value = {"is_faithful": True, "conflicts": []}
    mock_faithfulness_checker.check_disclaimer.return_value = {"has_disclaimer": True, "errors": []}

    pipeline = RAGPipeline(
        api_key="mock_key",
        retrieval_pipeline=mock_retrieval,
        generator=mock_generator,
        citation_checker=mock_citation_checker,
        faithfulness_checker=mock_faithfulness_checker,
        refusal_detector=mock_detector
    )

    result = pipeline.answer_question("Thử việc là gì?", strategy="dense")

    assert result["refused"] is False
    assert result["citations"] == []
    assert result["confidence"] == 0.90
