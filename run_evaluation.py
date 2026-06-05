import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Cấu hình mã hóa UTF-8 cho console trên Windows để tránh UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.config import (
    SETTINGS, 
    PROJECT_ROOT, 
    EMBEDDING_MODEL_NAME, 
    RERANKER_MODEL_NAME, 
    GEMINI_API_KEY
)
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.pipeline.rag_pipeline import RAGPipeline
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.rate_limiter import wrap_pipeline_with_rate_limiter

# Cấu hình Logging hiển thị thông tin trực quan sạch sẽ
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("RAGEvaluatorCLI")

# --- LỚP MOCK ĐỂ PHỤC VỤ CHẠY THỬ NGHIỆM KHÔNG CẦN API KEY ---

class MockGenerator:
    """Mock Generator giả lập phản hồi của Gemini khi chạy chế độ Mock hoặc thiếu API Key."""
    def generate_answer(self, query: str, merged_chunks: list, context_text: str):
        from src.generation.generator import RAGResponse, Citation
        
        # Thử tìm xem câu hỏi có Điều luật nào liên quan trong context không
        detected_article = "Điều 26"
        for chunk in merged_chunks:
            if "article" in chunk and chunk["article"]:
                detected_article = chunk["article"]
                break
                
        # Giả lập câu trả lời dựa trên từ khóa câu hỏi
        clean_q = query.lower()
        if "giữ bản chính" in clean_q or "giấy tờ tùy thân" in clean_q:
            answer = "Người sử dụng lao động không được giữ bản chính giấy tờ tùy thân, văn bằng, chứng chỉ của người lao động khi giao kết, thực hiện hợp đồng lao động theo quy định tại Điều 17 [1]. Lưu ý: Ý kiến tư vấn trên chỉ mang tính chất tham khảo dựa trên quy định của Bộ luật Lao động 2019."
            citations = [
                Citation(
                    citation_id=1, 
                    article="Điều 17", 
                    clause="1", 
                    title="Hành vi người sử dụng lao động không được làm khi giao kết, thực hiện hợp đồng lao động", 
                    source_url="https://example.com", 
                    evidence="Không được giữ bản chính giấy tờ tùy thân"
                )
            ]
        elif "đơn phương" in clean_q and "không cần báo trước" in clean_q:
            answer = "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động không cần báo trước trong một số trường hợp như bị ngược đãi, cưỡng bức lao động, hoặc không được trả đủ lương theo quy định tại Điều 35 [1]. Lưu ý: Ý kiến tư vấn trên chỉ mang tính chất tham khảo dựa trên quy định của Bộ luật Lao động 2019."
            citations = [
                Citation(
                    citation_id=1, 
                    article="Điều 35", 
                    clause="2", 
                    title="Quyền đơn phương chấm dứt hợp đồng lao động của người lao động", 
                    source_url="https://example.com", 
                    evidence="không cần báo trước trong các trường hợp"
                )
            ]
        else:
            answer = f"Người lao động và người sử dụng lao động thực hiện các quyền và nghĩa vụ theo quy định tại {detected_article} [1]. Lưu ý: Ý kiến tư vấn trên chỉ mang tính chất tham khảo dựa trên quy định của Bộ luật Lao động 2019."
            citations = [
                Citation(
                    citation_id=1, 
                    article=detected_article, 
                    clause=None, 
                    title="Quy định liên quan", 
                    source_url="https://example.com", 
                    evidence="quy định liên quan"
                )
            ]
            
        return RAGResponse(
            answer=answer,
            citations=citations,
            confidence=0.95
        )

class MockFaithfulnessChecker:
    """Mock FaithfulnessChecker giả lập kết quả bám nguồn xanh lá."""
    def check_faithfulness(self, response):
        return {"is_faithful": True, "conflicts": []}
        
    def check_disclaimer(self, response):
        return {"has_disclaimer": True, "errors": []}

class MockRefusalDetector:
    """Mock RefusalDetector phòng thủ 3 tầng dùng để từ chối các câu hỏi mẫu."""
    class MockRefusalJudgment:
        def __init__(self, refuse=False, reason="Hợp lệ", category="in_scope"):
            self.refuse = refuse
            self.reason = reason
            self.category = category

    def detect_query_refusal(self, query: str):
        query_clean = query.lower()
        if "nấu" in query_clean or "phở" in query_clean or "cơm" in query_clean:
            return self.MockRefusalJudgment(
                refuse=True, 
                reason="Rất tiếc, câu hỏi về nấu ăn nằm ngoài phạm vi Bộ luật Lao động 2019.", 
                category="out_of_scope"
            )
        if "phạt" in query_clean and ("tiền" in query_clean or "nhiêu" in query_clean or "mức" in query_clean):
            return self.MockRefusalJudgment(
                refuse=True,
                reason="Bộ luật Lao động 2019 chỉ quy định khung hành vi, mức xử phạt hành chính cụ thể nằm trong Nghị định 12/2022/NĐ-CP.",
                category="administrative_fine"
            )
        return self.MockRefusalJudgment()

    def detect_retrieval_refusal(self, retrieved_chunks: list, strategy: str):
        if not retrieved_chunks:
            return {
                "refuse": True,
                "reason": "Dựa trên Bộ luật Lao động 2019, tôi không tìm thấy căn cứ pháp lý liên quan.",
                "category": "no_relevant_context"
            }
        return {"refuse": False, "category": "in_scope"}

    def detect_output_refusal(self, response, verification_report: dict):
        return {"refuse": False, "category": "in_scope"}

class BypassRefusalDetector:
    """Bypass Refusal Detector dùng cho các cấu hình tắt từ chối (A đến E)."""
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


def setup_ablation_pipeline(config_id: str, api_key: str, use_mock_llm: bool):
    """
    Dựng RAGPipeline tương ứng với từng cấu hình thí nghiệm (A đến G) trong Đặc tả.
    """
    config_id = config_id.upper().strip()
    
    # 1. Xác định corpus
    if config_id in ("A", "B"):
        corpus_path = Path(PROJECT_ROOT) / "data/processed/corpus_fixed.jsonl"
        index_dir = Path(PROJECT_ROOT) / "data/processed"
    else:
        corpus_path = Path(PROJECT_ROOT) / "data/processed/corpus_structured.jsonl"
        index_dir = Path(PROJECT_ROOT) / "data/processed"

    # Strategy truy hồi
    if config_id in ("A", "C"):
        strategy = "bm25"
    elif config_id == "B":
        strategy = "dense"
    elif config_id == "D":
        strategy = "hybrid"
    else:
        strategy = "hybrid_rerank"

    # Hệ thống từ chối
    has_refusal = (config_id == "G")

    logger.info(f"--- Thiết lập cấu hình {config_id} ---")
    logger.info(f"  + Corpus: {corpus_path.name}")
    logger.info(f"  + Chiến lược retrieval: {strategy}")
    logger.info(f"  + Hệ thống từ chối: {'Kích hoạt' if has_refusal else 'Bỏ qua (Bypass)'}")

    # 2. Khởi tạo Retrieval Pipeline con
    retrieval_pipeline = RetrievalPipeline(
        corpus_path=corpus_path,
        embedding_model=EMBEDDING_MODEL_NAME,
        reranker_model=RERANKER_MODEL_NAME,
        index_dir=index_dir
    )

    # 3. Khởi tạo Pipeline chính
    if use_mock_llm:
        logger.info("  [MOCK LLM MODE] Pipeline đang chạy với Mock LLM Components (Không tốn chi phí API).")
        refusal_detector = MockRefusalDetector() if has_refusal else BypassRefusalDetector()
        pipeline = RAGPipeline(
            api_key="mock_key",
            retrieval_pipeline=retrieval_pipeline,
            generator=MockGenerator(),
            faithfulness_checker=MockFaithfulnessChecker(),
            refusal_detector=refusal_detector
        )
    else:
        refusal_detector = None if has_refusal else BypassRefusalDetector()
        pipeline = RAGPipeline(
            api_key=api_key,
            retrieval_pipeline=retrieval_pipeline,
            refusal_detector=refusal_detector
        )

    return pipeline, strategy, corpus_path

def run_configuration(config_id: str, benchmark_data: list, api_key: str, delay: float, use_mock_llm: bool, limit: int = None) -> list:
    """
    Chạy toàn bộ tập câu hỏi benchmark trên một cấu hình RAG cụ thể.
    """
    pipeline, strategy, corpus_path = setup_ablation_pipeline(config_id, api_key, use_mock_llm)
    
    # Chỉ bọc rate limiter nếu không phải chế độ Mock LLM
    if not use_mock_llm:
        pipeline = wrap_pipeline_with_rate_limiter(pipeline, delay_between_calls=delay)
    
    results = []
    subset = benchmark_data[:limit] if limit else benchmark_data
    
    # Nạp corpus của cấu hình để evaluator so khớp Điều luật tĩnh
    evaluator = RAGEvaluator(corpus_path=corpus_path)

    for idx, item in enumerate(subset):
        q_id = item.get("question_id", f"q_{idx}")
        question = item.get("question")
        gold_sources = item.get("gold_sources", [])
        must_refuse = item.get("must_refuse", False)
        
        logger.info(f"[{config_id}] Đang chạy câu hỏi {idx+1}/{len(subset)} ID: {q_id}")
        
        start_time = time.time()
        try:
            # Gọi Pipeline xử lý
            output = pipeline.answer_question(question, strategy=strategy)
            latency = time.time() - start_time
            
            # Tính toán các chỉ số đo lường
            retrieved_chunks = output.get("retrieved_chunks", [])
            citations = output.get("citations", [])
            refused = output.get("refused", False)
            
            eval_retrieval_5 = evaluator.evaluate_retrieval(retrieved_chunks, gold_sources, k=5)
            eval_retrieval_3 = evaluator.evaluate_retrieval(retrieved_chunks, gold_sources, k=3)
            eval_citations = evaluator.evaluate_citations(citations, retrieved_chunks)
            eval_refusal = evaluator.evaluate_refusal(refused, must_refuse)
            
            # Đóng gói kết quả câu hỏi
            record = {
                "question_id": q_id,
                "question": question,
                "gold_sources": gold_sources,
                "must_refuse": must_refuse,
                "refused": refused,
                "category": output.get("category", ""),
                "answer": output.get("answer", ""),
                "confidence": output.get("confidence", 0.0),
                "latency_seconds": latency,
                "recall_at_5": eval_retrieval_5["recall_at_k"],
                "mrr_at_5": eval_retrieval_5["mrr_at_k"],
                "hit_rate_at_5": eval_retrieval_5["hit_rate_at_k"],
                "recall_at_3": eval_retrieval_3["recall_at_k"],
                "mrr_at_3": eval_retrieval_3["mrr_at_k"],
                "hit_rate_at_3": eval_retrieval_3["hit_rate_at_k"],
                "fabricated_citation_rate": eval_citations["fabricated_citation_rate"],
                "unsupported_citation_rate": eval_citations["unsupported_citation_rate"],
                "citation_count": eval_citations["citation_count"],
                "refusal_correct": eval_refusal["refusal_correct"]
            }
            results.append(record)
            
        except Exception as e:
            logger.error(f"Lỗi khi chạy câu hỏi ID {q_id}: {e}")
            results.append({
                "question_id": q_id,
                "question": question,
                "error": str(e),
                "recall_at_5": 0.0,
                "mrr_at_5": 0.0,
                "refusal_correct": 0.0
            })
            
    return results

def aggregate_metrics(results: list) -> dict:
    """Tổng hợp điểm trung bình của tất cả chỉ số trong một lượt chạy."""
    total = len(results)
    if total == 0:
        return {}
        
    valid_results = [r for r in results if "error" not in r]
    v_total = len(valid_results)
    if v_total == 0:
        return {"status": "all_failed"}

    # Tính trung bình các chỉ số
    avg_recall_5 = sum(r["recall_at_5"] for r in valid_results) / v_total
    avg_mrr_5 = sum(r["mrr_at_5"] for r in valid_results) / v_total
    avg_hit_rate_5 = sum(r["hit_rate_at_5"] for r in valid_results) / v_total
    avg_recall_3 = sum(r["recall_at_3"] for r in valid_results) / v_total
    avg_mrr_3 = sum(r["mrr_at_3"] for r in valid_results) / v_total
    avg_hit_rate_3 = sum(r["hit_rate_at_3"] for r in valid_results) / v_total
    
    avg_fab_cite = sum(r["fabricated_citation_rate"] for r in valid_results) / v_total
    avg_unsupp_cite = sum(r["unsupported_citation_rate"] for r in valid_results) / v_total
    avg_refusal_acc = sum(r["refusal_correct"] for r in valid_results) / v_total
    avg_latency = sum(r["latency_seconds"] for r in valid_results) / v_total
    
    return {
        "total_questions": total,
        "successful_runs": v_total,
        "recall_at_5": avg_recall_5,
        "mrr_at_5": avg_mrr_5,
        "hit_rate_at_5": avg_hit_rate_5,
        "recall_at_3": avg_recall_3,
        "mrr_at_3": avg_mrr_3,
        "hit_rate_at_3": avg_hit_rate_3,
        "fabricated_citation_rate": avg_fab_cite,
        "unsupported_citation_rate": avg_unsupp_cite,
        "refusal_accuracy": avg_refusal_acc,
        "avg_latency_seconds": avg_latency
    }

def print_markdown_table(summary_data: dict):
    """In bảng so sánh kết quả Ablation Study dưới dạng Markdown chuyên nghiệp."""
    headers = [
        "Cấu hình", "Chunking", "Retrieval", "Refusal", 
        "Recall@5", "MRR@5", "Recall@3", "Fabricated Cit.", "Refusal Acc.", "Latency (s)"
    ]
    
    rows = []
    # Bản đồ đặc tả cấu hình
    spec_map = {
        "A": ["Fixed-size", "BM25", "Không"],
        "B": ["Fixed-size", "Dense", "Không"],
        "C": ["Structure-aware", "BM25", "Không"],
        "D": ["Structure-aware", "Hybrid", "Không"],
        "E": ["Structure-aware", "Hybrid + Rerank", "Không"],
        "G": ["Structure-aware", "Hybrid + Rerank", "Có"]
    }
    
    for cfg_id, metrics in summary_data.items():
        spec = spec_map.get(cfg_id, ["Unknown", "Unknown", "Unknown"])
        rows.append([
            f"**Config {cfg_id}**",
            spec[0],
            spec[1],
            spec[2],
            f"{metrics.get('recall_at_5', 0.0):.3f}",
            f"{metrics.get('mrr_at_5', 0.0):.3f}",
            f"{metrics.get('recall_at_3', 0.0):.3f}",
            f"{metrics.get('fabricated_citation_rate', 0.0):.3f}",
            f"{metrics.get('refusal_accuracy', 0.0):.3f}",
            f"{metrics.get('avg_latency_seconds', 0.0):.2f}s"
        ])
        
    # Render bảng
    header_line = " | ".join(headers)
    divider_line = " | ".join(["---"] * len(headers))
    
    print("\n### BẢNG SO SÁNH THỬ NGHIỆM THỰC NGHIỆM (ABLATION STUDY)")
    print(f"| {header_line} |")
    print(f"| {divider_line} |")
    for row in rows:
        row_str = " | ".join(row)
        print(f"| {row_str} |")
    print("\n")

def main():
    parser = argparse.ArgumentParser(description="Hệ thống chạy Thử nghiệm và Đánh giá RAG ViLaborRAG.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="G", 
        help="Lựa chọn cấu hình chạy: A, B, C, D, E, G hoặc ALL (để chạy tất cả các cấu hình). Mặc định là G."
    )
    parser.add_argument(
        "--benchmark_path", 
        type=str, 
        default="data/benchmark/benchmark.json", 
        help="Đường dẫn đến file câu hỏi benchmark JSON."
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Giới hạn số câu hỏi để chạy thử nhanh (Smoke Test)."
    )
    parser.add_argument(
        "--delay", 
        type=float, 
        default=4.0, 
        help="Khoảng nghỉ (giây) giữa các yêu cầu gọi API Gemini để chống Rate Limit. Mặc định 4.0s."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Kích hoạt chế độ Mock LLM (không cần khóa API Gemini, tự động giả lập phản hồi của mô hình)."
    )
    args = parser.parse_args()

    # 1. Xác minh API Key và quyết định chế độ chạy
    api_key = GEMINI_API_KEY
    use_mock_llm = args.mock
    
    if not api_key:
        logger.warning("Không tìm thấy GEMINI_API_KEY trong môi trường/file .env.")
        logger.warning("Hệ thống tự động kích hoạt [MOCK LLM MODE] để tiếp tục chạy đánh giá giả lập.")
        use_mock_llm = True

    # 2. Đọc file Benchmark
    bench_path = Path(args.benchmark_path)
    if not bench_path.exists():
        # Fallback về benchmark_sample.json nếu benchmark.json chưa được tạo
        fallback_path = Path(PROJECT_ROOT) / "data/benchmark/benchmark_sample.json"
        if fallback_path.exists():
            logger.info(f"Không tìm thấy {bench_path.name}. Sử dụng tệp mẫu dự phòng: {fallback_path.name}")
            bench_path = fallback_path
        else:
            logger.error(f"Không tìm thấy tệp câu hỏi benchmark tại: {bench_path}")
            sys.exit(1)

    try:
        with open(bench_path, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
        logger.info(f"Đã nạp thành công {len(benchmark_data)} câu hỏi từ tệp {bench_path.name}")
    except Exception as e:
        logger.error(f"Không thể đọc tệp benchmark JSON: {e}")
        sys.exit(1)

    # 3. Phân loại cấu hình cần chạy
    target_config = args.config.upper().strip()
    configs_to_run = ["A", "B", "C", "D", "E", "G"] if target_config == "ALL" else [target_config]
    
    # 4. Vòng lặp thực thi
    summary_reports = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Tạo thư mục đầu ra nếu chưa có
    output_dir = Path(PROJECT_ROOT) / "data/benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    for cfg in configs_to_run:
        logger.info(f"\n=======================================================")
        logger.info(f"BẮT ĐẦU CHẠY THỬ NGHIỆM CẤU HÌNH: Config {cfg}")
        logger.info(f"=======================================================")
        
        # Chạy thí nghiệm
        run_results = run_configuration(cfg, benchmark_data, api_key, args.delay, use_mock_llm, args.limit)
        
        # Lưu kết quả chi tiết của cấu hình
        detail_file = output_dir / f"evaluation_detail_{cfg}_{timestamp}.json"
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump(run_results, f, ensure_ascii=False, indent=2)
        logger.info(f"[SUCCESS] Đã xuất kết quả chi tiết của Config {cfg} sang: {detail_file.name}")
        
        # Tính toán chỉ số tổng hợp
        cfg_summary = aggregate_metrics(run_results)
        summary_reports[cfg] = cfg_summary
        
        # In tóm tắt kết quả
        logger.info(f"--- KẾT QUẢ ĐO LƯỜNG SƠ BỘ CONFIG {cfg} ---")
        logger.info(f"  + Số câu hỏi chạy thành công: {cfg_summary.get('successful_runs')}/{cfg_summary.get('total_questions')}")
        logger.info(f"  + Recall@5: {cfg_summary.get('recall_at_5', 0.0):.4f}")
        logger.info(f"  + MRR@5: {cfg_summary.get('mrr_at_5', 0.0):.4f}")
        logger.info(f"  + Recall@3: {cfg_summary.get('recall_at_3', 0.0):.4f}")
        logger.info(f"  + Tỷ lệ trích dẫn bịa (Fabricated): {cfg_summary.get('fabricated_citation_rate', 0.0):.4f}")
        logger.info(f"  + Độ chính xác từ chối (Refusal Acc): {cfg_summary.get('refusal_accuracy', 0.0):.4f}")
        logger.info(f"  + Thời gian chạy trung bình: {cfg_summary.get('avg_latency_seconds', 0.0):.2f} giây")

    # 5. Lưu và In bảng tổng hợp
    summary_file = output_dir / f"evaluation_summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_reports, f, ensure_ascii=False, indent=2)
    logger.info(f"\n[SUCCESS] Đã xuất báo cáo tổng hợp Ablation Study sang: {summary_file.name}")
    
    # In bảng Markdown
    print_markdown_table(summary_reports)

if __name__ == "__main__":
    main()
