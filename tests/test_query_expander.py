import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.query_expander import QueryExpander, ExpandedQueries
from src.retrieval.retrieval_pipeline import RetrievalPipeline

def test_query_expander_mock_mode():
    """Kiểm tra QueryExpander tự động trả về [] ở chế độ Mock (truyền api_key rỗng)."""
    expander = QueryExpander(api_key="")
    assert expander.client is None
    res = expander.expand_query("Thử việc có lương không?")
    assert res == []

def test_query_expander_exception_handling():
    """Kiểm tra QueryExpander xử lý ngoại lệ và fallback về [] an toàn."""
    expander = QueryExpander(api_key="mock_key")
    
    with patch.object(expander, "client") as mock_client:
        mock_client.models.generate_content.side_effect = Exception("API Error")
        res = expander.expand_query("Thử việc có lương không?")
        assert res == []

def test_query_expander_success_mode():
    """Kiểm tra QueryExpander sinh câu hỏi phụ thành công khi có kết quả trả về."""
    expander = QueryExpander(api_key="mock_key")
    
    mock_response = MagicMock()
    mock_response.text = '{"queries": ["tiền lương thử việc", "mức lương thử việc tối thiểu"]}'
    
    with patch.object(expander, "client") as mock_client:
        mock_client.models.generate_content.return_value = mock_response
        res = expander.expand_query("Thử việc có lương không?")
        assert len(res) == 2
        assert "tiền lương thử việc" in res
        assert "mức lương thử việc tối thiểu" in res

def test_retrieval_pipeline_multi_query_integration():
    """Kiểm tra RetrievalPipeline gọi QueryExpander và gộp kết quả thành công khi chạy hybrid_rerank."""
    # Khởi tạo mock cho retrievers và reranker với ít nhất 1 chunk để tránh lỗi chia cho 0 trong rank-bm25
    sample_chunks = [{"chunk_id": "chunk_mock", "text": "văn bản pháp luật lao động", "article": "Điều 1"}]
    pipeline = RetrievalPipeline(chunks=sample_chunks)
    
    # Mock expand_query trả ra 2 câu hỏi phụ
    pipeline.query_expander.expand_query = MagicMock(return_value=["tiền lương lễ", "nghỉ Tết hưởng lương"])
    
    # Thay vì mock pipeline.retrieve, ta mock các retriever con để chạy được phương thức retrieve thật
    pipeline.bm25_retriever.retrieve = MagicMock(return_value=[
        {"chunk_id": "chunk_1", "text": "lương lễ", "score": 10.0, "article": "Điều 98"}
    ])
    pipeline.dense_retriever.retrieve = MagicMock(return_value=[
        {"chunk_id": "chunk_1", "text": "lương lễ", "score": 0.8, "article": "Điều 98"}
    ])
    
    # Mock reranker
    pipeline.reranker = MagicMock()
    pipeline.reranker.rerank = MagicMock(return_value=[{"chunk_id": "chunk_1", "score": 1.0}])
    
    # Chạy hybrid_rerank thực tế
    res = pipeline.retrieve("Lương Tết", strategy="hybrid_rerank", top_k=5)
    
    # Xác minh QueryExpander được gọi đúng
    pipeline.query_expander.expand_query.assert_called_once_with("Lương Tết")
    
    # Xác minh Reranker được gọi
    assert pipeline.reranker.rerank.called
    called_args = pipeline.reranker.rerank.call_args[0]
    assert called_args[0] == "Lương Tết"
    
    # Kiểm tra danh sách candidates gộp vào reranker
    merged_chunks = called_args[1]
    assert len(merged_chunks) > 0
