import re
import pytest
from unittest.mock import MagicMock, patch
from src.generation.generator import RAGResponse, Citation

try:
    from src.verification.citation_checker import CitationChecker
    from src.verification.faithfulness_checker import FaithfulnessChecker
except ImportError:
    CitationChecker = None
    FaithfulnessChecker = None

@pytest.fixture
def sample_corpus_chunks():
    """CSDL Corpus giả lập chứa 3 Điều thực tế để đối chiếu."""
    return [
        {
            "chunk_id": "bll2019_dieu_24_khoan_1",
            "article": "Điều 24",
            "clause": "1",
            "title": "Thử việc",
            "text": "Người sử dụng lao động và người lao động có thỏa thuận về việc thử việc, quyền và nghĩa vụ của hai bên trong thời gian thử việc.",
            "source_url": "https://luatvietnam.vn/dieu-24"
        },
        {
            "chunk_id": "bll2019_dieu_85_khoan_1",
            "article": "Điều 85",
            "clause": "1",
            "title": "Thời gian nghỉ ngơi",
            "text": "Người lao động làm việc thử việc tối đa không quá 30 ngày đối với lao động chuyên môn nghiệp vụ trung cấp.",
            "source_url": "https://luatvietnam.vn/dieu-85"
        },
        {
            "chunk_id": "bll2019_dieu_36_khoan_1",
            "article": "Điều 36",
            "clause": "1",
            "title": "Quyền đơn phương chấm dứt hợp đồng lao động của người lao động",
            "text": "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động nhưng phải báo trước cho người sử dụng lao động ít nhất 45 ngày đối với hợp đồng không xác định thời hạn.",
            "source_url": "https://luatvietnam.vn/dieu-36"
        }
    ]

@pytest.fixture
def sample_retrieved_chunks(sample_corpus_chunks):
    """Context truy hồi thực tế (chỉ chứa Điều 24 và Điều 85). Điều 36 tồn tại trong Corpus nhưng không được truy hồi."""
    return [sample_corpus_chunks[0], sample_corpus_chunks[1]]

def test_classes_exist():
    """Xác nhận các lớp của Pha 4 đã tồn tại (TDD Bước 1 mong đợi Thất bại)."""
    assert CitationChecker is not None, "Chưa lập trình lớp CitationChecker!"
    assert FaithfulnessChecker is not None, "Chưa lập trình lớp FaithfulnessChecker!"

def test_citation_checker_valid_case(sample_corpus_chunks, sample_retrieved_chunks):
    """Kiểm tra trường hợp trích dẫn hoàn hảo không có bất kỳ lỗi nào."""
    if CitationChecker is None:
        pytest.fail("Chưa lập trình lớp CitationChecker!")

    # Khởi tạo checker trực tiếp với corpus chunks giả lập
    checker = CitationChecker(chunks=sample_corpus_chunks)
    
    valid_response = RAGResponse(
        answer="Theo quy định, hai bên thỏa thuận về việc thử việc [1]. Thời gian thử việc là không quá 30 ngày [2]. Lưu ý: Ý kiến tư vấn trên chỉ mang tính chất tham khảo dựa trên quy định của Bộ luật Lao động 2019 và dữ liệu ngữ cảnh hiện có tại thời điểm tra cứu.",
        citations=[
            Citation(
                citation_id=1,
                article="Điều 24",
                clause="1",
                title="Thử việc",
                source_url="https://luatvietnam.vn/dieu-24",
                evidence="thỏa thuận về việc thử việc"
            ),
            Citation(
                citation_id=2,
                article="Điều 85",
                clause="1",
                title="Thời gian nghỉ ngơi",
                source_url="https://luatvietnam.vn/dieu-85",
                evidence="tối đa không quá 30 ngày"
            )
        ],
        confidence=0.95
    )
    
    report = checker.check_citations(valid_response, sample_retrieved_chunks)
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0

def test_citation_checker_fabricated_citation(sample_corpus_chunks, sample_retrieved_chunks):
    """Edge case cực đoan 1: Bịa trích dẫn Điều luật không tồn tại trong Corpus (Ví dụ: Điều 300)."""
    if CitationChecker is None:
        pytest.fail("Chưa lập trình lớp CitationChecker!")

    checker = CitationChecker(chunks=sample_corpus_chunks)
    
    fabricated_response = RAGResponse(
        answer="Theo Điều 300, người sử dụng lao động có quyền xử phạt hành chính [1].",
        citations=[
            Citation(
                citation_id=1,
                article="Điều 300",  # Không tồn tại trong sample_corpus_chunks
                clause="1",
                title="Quyền xử phạt",
                source_url="https://test.url",
                evidence="xử phạt hành chính"
            )
        ],
        confidence=0.90
    )
    
    report = checker.check_citations(fabricated_response, sample_retrieved_chunks)
    assert report["is_valid"] is False
    assert "citation_not_found_in_corpus" in report["errors"]

def test_citation_checker_out_of_context(sample_corpus_chunks, sample_retrieved_chunks):
    """Edge case cực đoan 2: Trích dẫn Điều luật có thực trong Corpus nhưng không có trong Ngữ cảnh được truy hồi."""
    if CitationChecker is None:
        pytest.fail("Chưa lập trình lớp CitationChecker!")

    checker = CitationChecker(chunks=sample_corpus_chunks)
    
    out_of_context_response = RAGResponse(
        answer="Người lao động được đơn phương chấm dứt hợp đồng nhưng phải báo trước 45 ngày [1].",
        citations=[
            Citation(
                citation_id=1,
                article="Điều 36",  # Có trong corpus nhưng không có trong sample_retrieved_chunks
                clause="1",
                title="Quyền đơn phương chấm dứt hợp đồng lao động của người lao động",
                source_url="https://luatvietnam.vn/dieu-36",
                evidence="phải báo trước cho người sử dụng lao động ít nhất 45 ngày"
            )
        ],
        confidence=0.90
    )
    
    report = checker.check_citations(out_of_context_response, sample_retrieved_chunks)
    assert report["is_valid"] is False
    assert "citation_not_in_retrieved_context" in report["errors"]

def test_citation_checker_fabricated_evidence(sample_corpus_chunks, sample_retrieved_chunks):
    """Edge case cực đoan 3: Trích dẫn Điều luật đúng nhưng đoạn evidence làm bằng chứng bị mô hình bịa ra."""
    if CitationChecker is None:
        pytest.fail("Chưa lập trình lớp CitationChecker!")

    checker = CitationChecker(chunks=sample_corpus_chunks)
    
    fake_evidence_response = RAGResponse(
        answer="Thời gian thử việc là 3 tháng đối với lao động trình độ cao [1].",
        citations=[
            Citation(
                citation_id=1,
                article="Điều 24",
                clause="1",
                title="Thử việc",
                source_url="https://luatvietnam.vn/dieu-24",
                evidence="lao động trình độ cao phải thử việc 3 tháng"  # Chuỗi này không có trong text gốc của Điều 24
            )
        ],
        confidence=0.85
    )
    
    report = checker.check_citations(fake_evidence_response, sample_retrieved_chunks)
    assert report["is_valid"] is False
    assert "fabricated_evidence" in report["errors"]

def test_citation_checker_malformed_bracket(sample_corpus_chunks, sample_retrieved_chunks):
    """Edge case cực đoan 4: Nhãn trích dẫn trong answer lệch với số lượng citations trong danh sách."""
    if CitationChecker is None:
        pytest.fail("Chưa lập trình lớp CitationChecker!")

    checker = CitationChecker(chunks=sample_corpus_chunks)
    
    malformed_response = RAGResponse(
        answer="Hai bên thỏa thuận về việc thử việc [1]. Tuy nhiên, có một nhãn bịa [2] ở đây.",
        citations=[
            Citation(
                citation_id=1,
                article="Điều 24",
                clause="1",
                title="Thử việc",
                source_url="https://luatvietnam.vn/dieu-24",
                evidence="thỏa thuận về việc thử việc"
            )
            # Thiếu citation_id = 2 trong danh sách
        ],
        confidence=0.95
    )
    
    report = checker.check_citations(malformed_response, sample_retrieved_chunks)
    assert report["is_valid"] is False
    assert "malformed_citation" in report["errors"]

def test_faithfulness_checker_valid_case():
    """Kiểm tra trường hợp bám nguồn hoàn hảo (Rule-based khớp số ngày và LLM thẩm định True)."""
    if FaithfulnessChecker is None:
        pytest.fail("Chưa lập trình lớp FaithfulnessChecker!")

    checker = FaithfulnessChecker(api_key="mock_key")
    
    response = RAGResponse(
        answer="Thời gian thử việc tối đa là 30 ngày [1]. Lưu ý: Ý kiến tư vấn trên chỉ mang tính chất tham khảo dựa trên quy định của Bộ luật Lao động 2019 và dữ liệu ngữ cảnh hiện có tại thời điểm tra cứu.",
        citations=[
            Citation(
                citation_id=1,
                article="Điều 85",
                clause="1",
                title="Thời gian nghỉ ngơi",
                source_url="https://luatvietnam.vn/dieu-85",
                evidence="thử việc tối đa không quá 30 ngày đối với lao động"
            )
        ],
        confidence=0.90
    )
    
    # Mock LLM-as-judge kết quả thẩm định True
    mock_response_obj = MagicMock()
    mock_response_obj.text = '{"supported": true, "reason": "Bằng chứng nêu rõ thời gian thử việc tối đa là 30 ngày."}'
    
    with patch("google.genai.Client") as MockClient:
        mock_client_instance = MockClient.return_value
        mock_client_instance.models.generate_content.return_value = mock_response_obj
        checker.client = mock_client_instance
        
        report = checker.check_faithfulness(response)
        assert report["is_faithful"] is True
        assert len(report["conflicts"]) == 0

def test_faithfulness_checker_numeric_conflict():
    """Edge case cực đoan 5: Lập luận trong câu trả lời có số liệu/thời hạn sai lệch trực tiếp với Bằng chứng gốc."""
    if FaithfulnessChecker is None:
        pytest.fail("Chưa lập trình lớp FaithfulnessChecker!")

    checker = FaithfulnessChecker(api_key="mock_key")
    
    # Mâu thuẫn số ngày: Câu trả lời nói 60 ngày, nhưng evidence ghi 30 ngày
    conflict_response = RAGResponse(
        answer="Thời gian thử việc tối đa phải là 60 ngày [1].",
        citations=[
            Citation(
                citation_id=1,
                article="Điều 85",
                clause="1",
                title="Thời gian nghỉ ngơi",
                source_url="https://luatvietnam.vn/dieu-85",
                evidence="thử việc tối đa không quá 30 ngày đối với lao động"
            )
        ],
        confidence=0.90
    )
    
    # Ở đây bộ rule-based thô phải tự phát hiện lệch số (60 vs 30) mà không cần tốn chi phí gọi LLM
    report = checker.check_faithfulness(conflict_response)
    assert report["is_faithful"] is False
    assert "faithfulness_conflict" in report["conflicts"][0]["error_type"]
    assert "60" in report["conflicts"][0]["claim"]
    assert "30" in report["conflicts"][0]["evidence"]

def test_faithfulness_checker_semantic_unfaithful():
    """Edge case cực đoan 6: Không bám nguồn về mặt ngữ nghĩa sâu (LLM-as-judge thẩm định False)."""
    if FaithfulnessChecker is None:
        pytest.fail("Chưa lập trình lớp FaithfulnessChecker!")

    checker = FaithfulnessChecker(api_key="mock_key")
    
    # Câu trả lời suy diễn bừa bãi: Người thử việc được hưởng 100% lương
    unfaithful_response = RAGResponse(
        answer="Người lao động thử việc được hưởng toàn bộ lương chính thức [1].",
        citations=[
            Citation(
                citation_id=1,
                article="Điều 24",
                clause="1",
                title="Thử việc",
                source_url="https://luatvietnam.vn/dieu-24",
                evidence="thỏa thuận về việc thử việc, quyền và nghĩa vụ của hai bên"
            )
        ],
        confidence=0.80
    )
    
    # Mock LLM-as-judge kết quả thẩm định False
    mock_response_obj = MagicMock()
    mock_response_obj.text = '{"supported": false, "reason": "Bằng chứng chỉ nêu về việc thỏa thuận quyền và nghĩa vụ thử việc, không có bất kỳ thông tin nào chứng minh mức lương hưởng là 100%."}'
    
    with patch("google.genai.Client") as MockClient:
        mock_client_instance = MockClient.return_value
        mock_client_instance.models.generate_content.return_value = mock_response_obj
        checker.client = mock_client_instance
        
        report = checker.check_faithfulness(unfaithful_response)
        assert report["is_faithful"] is False
        assert "severe_unfaithfulness" in report["conflicts"][0]["error_type"]
        assert report["conflicts"][0]["supported"] is False

def test_faithfulness_checker_missing_disclaimer():
    """Edge case cực đoan 7: Phát hiện thiếu tuyên bố miễn trừ trách nhiệm pháp lý mặc định ở cuối."""
    if FaithfulnessChecker is None:
        pytest.fail("Chưa lập trình lớp FaithfulnessChecker!")

    checker = FaithfulnessChecker(api_key="mock_key")
    
    # Câu trả lời không hề có dòng miễn trừ trách nhiệm ở cuối
    no_disclaimer_response = RAGResponse(
        answer="Hai bên có thỏa thuận thử việc theo quy định [1].",
        citations=[
            Citation(
                citation_id=1,
                article="Điều 24",
                clause="1",
                title="Thử việc",
                source_url="https://luatvietnam.vn/dieu-24",
                evidence="thỏa thuận về việc thử việc"
            )
        ],
        confidence=0.90
    )
    
    report = checker.check_disclaimer(no_disclaimer_response)
    assert report["has_disclaimer"] is False
    assert "missing_disclaimer" in report["errors"]

def test_arithmetic_verifier_bypass_valid():
    """Kiểm định Arithmetic-aware Verification: Bỏ qua số trong câu hỏi người dùng và phép nhân phần trăm hợp lệ."""
    from src.verification.faithfulness_checker import check_numeric_discrepancy
    
    # Số 500000 trong query, 300% trong evidence. Kết quả 1500000 tính ra hợp lệ.
    query = "Lương ngày thường của tôi là 500.000 đồng, đi làm Tết được bao nhiêu?"
    evidence = "Người lao động làm việc vào ngày nghỉ lễ, tết được trả ít nhất bằng 300% lương ngày thường."
    claim = "Tiền lương làm việc vào ngày lễ Tết là: 500.000 đồng * 300% = 1.500.000 đồng."
    
    err = check_numeric_discrepancy(claim, evidence, article="Điều 98", clause="1", query=query)
    assert err is None, f"Lẽ ra phép tính hợp lệ phải được thông qua, nhưng báo lỗi: {err}"

def test_arithmetic_verifier_mismatch():
    """Kiểm định Arithmetic-aware Verification: Phép tính sai bị chặn bởi bộ đối chiếu số."""
    from src.verification.faithfulness_checker import check_numeric_discrepancy
    
    # 500.000 * 300% = 18.000.000 là phép tính sai lệch
    query = "Lương ngày thường của tôi là 500.000 đồng, đi làm Tết được bao nhiêu?"
    evidence = "Người lao động làm việc vào ngày nghỉ lễ, tết được trả ít nhất bằng 300% lương ngày thường."
    claim = "Tiền lương làm việc vào ngày lễ Tết là: 500.000 đồng * 300% = 18.000.000 đồng."
    
    err = check_numeric_discrepancy(claim, evidence, article="Điều 98", clause="1", query=query)
    assert err is not None, "Phép tính sai lệch lẽ ra phải bị chặn lại."
    assert "mâu thuẫn số liệu" in err
