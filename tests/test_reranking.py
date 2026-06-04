import pytest
from src.reranking.reranker import Reranker

def test_reranker_cross_encoder():
    """Kiểm tra hoạt động tái xếp hạng của Reranker Cross-Encoder từ module mới."""
    sample_chunks = [
        {"chunk_id": "chunk_1", "text": "Người lao động được quyền đơn phương chấm dứt hợp đồng lao động bất kỳ lúc nào."},
        {"chunk_id": "chunk_2", "text": "Thời gian thử việc đối với công nhân kỹ thuật tối đa không quá 30 ngày."}
    ]
    
    # Dùng mô hình CrossEncoder siêu nhẹ cho unit test
    test_reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker = Reranker(model_name=test_reranker_model)
    
    # Tái xếp hạng câu hỏi về "thử việc"
    reranked = reranker.rerank(
        query="thử việc công nhân kỹ thuật",
        chunks=sample_chunks,
        top_k=2
    )
    
    assert len(reranked) == 2
    # Kỳ vọng chunk_2 sẽ được đẩy lên số 1 vì chứa thông tin thử việc công nhân sát nghĩa nhất
    assert reranked[0]["chunk_id"] == "chunk_2"
    assert "score" in reranked[0]
