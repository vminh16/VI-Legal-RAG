import pytest
from src.evaluation.evaluator import RAGEvaluator

def test_evaluator_normalization():
    """Kiểm tra tính đúng đắn của các hàm chuẩn hóa Điều và Khoản."""
    evaluator = RAGEvaluator(corpus_path="non_existent_file.jsonl")
    
    # Chuẩn hóa Điều
    assert evaluator.normalize_article("Điều 24.") == "Điều 24"
    assert evaluator.normalize_article("dieu 24") == "Điều 24"
    assert evaluator.normalize_article("Điều   107") == "Điều 107"
    assert evaluator.normalize_article("Không khớp điều") == "Không khớp điều"
    
    # Chuẩn hóa Khoản
    assert evaluator.normalize_clause("Khoản 1") == "1"
    assert evaluator.normalize_clause("2") == "2"
    assert evaluator.normalize_clause("") == ""

def test_evaluate_retrieval_perfect_match():
    """Kiểm tra tính toán truy hồi khớp hoàn hảo ở rank 1."""
    evaluator = RAGEvaluator(corpus_path="non_existent_file.jsonl")
    
    retrieved = [
        {"chunk_id": "c1", "article": "Điều 24", "clause": "1", "score": 0.9},
        {"chunk_id": "c2", "article": "Điều 26", "clause": None, "score": 0.8}
    ]
    
    gold_sources = [
        {"article": "Điều 24", "clause": "1"}
    ]
    
    metrics = evaluator.evaluate_retrieval(retrieved, gold_sources, k=5)
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr_at_k"] == 1.0
    assert metrics["hit_rate_at_k"] == 1.0

def test_evaluate_retrieval_partial_match():
    """Kiểm tra truy hồi khớp một phần (gold source nằm ở rank 2)."""
    evaluator = RAGEvaluator(corpus_path="non_existent_file.jsonl")
    
    retrieved = [
        {"chunk_id": "c1", "article": "Điều 10", "clause": None, "score": 0.9},
        {"chunk_id": "c2", "article": "Điều 24", "clause": "2", "score": 0.8}, # Trùng Điều 24 nhưng lệch Khoản (gold là khoản 1)
        {"chunk_id": "c3", "article": "Điều 26", "clause": None, "score": 0.7}
    ]
    
    # Gold source chỉ định cả Điều và Khoản
    gold_sources = [
        {"article": "Điều 24", "clause": "1"}, # Không khớp hoàn toàn vì lệch Khoản
        {"article": "Điều 26", "clause": None}  # Khớp với rank 3
    ]
    
    metrics = evaluator.evaluate_retrieval(retrieved, gold_sources, k=5)
    # Chỉ khớp được Điều 26 (1/2 gold sources) -> recall = 0.5
    assert metrics["recall_at_k"] == 0.5
    # Gold source khớp đầu tiên là Điều 26 tại rank 3 -> MRR = 1/3
    assert metrics["mrr_at_k"] == pytest.approx(1/3)
    assert metrics["hit_rate_at_k"] == 1.0

def test_evaluate_citations_metrics():
    """Kiểm tra tính toán tỷ lệ trích dẫn bịa và lệch nguồn."""
    evaluator = RAGEvaluator(corpus_path="non_existent_file.jsonl")
    
    # Giả lập tập các Điều luật hợp lệ trong corpus
    evaluator.corpus_articles = {"Điều 24", "Điều 25", "Điều 26"}
    
    retrieved = [
        {"article": "Điều 24", "clause": None},
        {"article": "Điều 25", "clause": None}
    ]
    
    citations = [
        {"article": "Điều 24", "clause": "1"}, # Hợp lệ & Trùng retrieved
        {"article": "Điều 26", "clause": None}, # Hợp lệ nhưng không nằm trong retrieved (lệch nguồn)
        {"article": "Điều 300", "clause": None} # Bịa luật (không nằm trong corpus_articles)
    ]
    
    metrics = evaluator.evaluate_citations(citations, retrieved)
    # Tỷ lệ bịa luật: 1/3 (Điều 300)
    assert metrics["fabricated_citation_rate"] == pytest.approx(1/3)
    # Tỷ lệ lệch nguồn: 2/3 (Điều 26 và Điều 300 đều không nằm trong retrieved)
    assert metrics["unsupported_citation_rate"] == pytest.approx(2/3)
    assert metrics["citation_count"] == 3

def test_evaluate_refusal_metrics():
    """Kiểm tra chỉ số đo lường từ chối (Refusal Accuracy)."""
    evaluator = RAGEvaluator(corpus_path="non_existent_file.jsonl")
    
    # Từ chối đúng: must_refuse=True và refused=True
    metrics1 = evaluator.evaluate_refusal(refused=True, must_refuse=True)
    assert metrics1["refusal_correct"] == 1.0
    
    # Cho qua đúng: must_refuse=False và refused=False
    metrics2 = evaluator.evaluate_refusal(refused=False, must_refuse=False)
    assert metrics2["refusal_correct"] == 1.0
    
    # Sai sót (Bỏ lọt): must_refuse=True nhưng refused=False
    metrics3 = evaluator.evaluate_refusal(refused=False, must_refuse=True)
    assert metrics3["refusal_correct"] == 0.0
    
    # Sai sót (Từ chối nhầm): must_refuse=False nhưng refused=True
    metrics4 = evaluator.evaluate_refusal(refused=True, must_refuse=False)
    assert metrics4["refusal_correct"] == 0.0
