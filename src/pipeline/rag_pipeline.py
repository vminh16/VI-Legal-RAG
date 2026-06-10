import logging
from src.config import SETTINGS
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.retrieval.context_builder import ContextBuilder
from src.generation.generator import GeminiGenerator
from src.verification.citation_checker import CitationChecker
from src.verification.faithfulness_checker import FaithfulnessChecker
from src.verification.refusal_detector import RefusalDetector

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Bộ Điều phối RAG Pipeline trung tâm (RAG Pipeline Coordinator).
    Đóng vai trò là Điểm tiếp nhận duy nhất (Single Entrypoint) của hệ thống ViLaborRAG,
    phối hợp và điều phối hoạt động của 10 Thành phần chuẩn Đặc tả.
    """
    def __init__(
        self,
        api_key: str = None,
        retrieval_pipeline = None,
        context_builder = None,
        generator = None,
        citation_checker = None,
        faithfulness_checker = None,
        refusal_detector = None
    ):
        # 1. Tải cấu hình từ settings.yml
        retrieval_cfg = SETTINGS.get("retrieval", {})
        top_k = retrieval_cfg.get("top_k", 5)
        
        # 2. Khởi tạo toàn bộ 10 Thành phần đơn nhiệm con
        self.retrieval_pipeline = retrieval_pipeline if retrieval_pipeline is not None else RetrievalPipeline()
        self.context_builder = context_builder if context_builder is not None else ContextBuilder(max_chunks=top_k)
        self.generator = generator if generator is not None else GeminiGenerator(api_key=api_key)
        self.citation_checker = citation_checker if citation_checker is not None else CitationChecker()
        self.faithfulness_checker = faithfulness_checker if faithfulness_checker is not None else FaithfulnessChecker(api_key=api_key)
        self.refusal_detector = refusal_detector if refusal_detector is not None else RefusalDetector(api_key=api_key)

    def answer_question(self, query: str, strategy: str = "hybrid_rerank", top_k: int = 5) -> dict:
        """
        Orchestrator trung tâm xử lý trọn vẹn luồng RAG 3 Tầng Từ chối Lai (Option C):
        
        1. Tầng 1 Refusal: Tiền kiểm Ý định (Query Intent Classification).
        2. Retrieval: BM25 / Dense / Hybrid / Cross-Encoder Reranker.
        3. Tầng 2 Refusal: Trung kiểm Điểm tương đồng Top-1 so với Ngưỡng sàn.
        4. Context Builder: Gộp các chunk cùng Điều & Định dạng chuẩn [Nguồn X].
        5. Generation: Sinh câu trả lời có cấu trúc RAGResponse.
        6. Tầng 3 Refusal (Part 1): Kiểm tra Generator tự tin thấp (confidence = 0.0).
        7. Verification: Citation check tĩnh + Faithfulness check thô số + LLM-as-judge ngữ nghĩa.
        8. Tầng 3 Refusal (Part 2): Hậu kiểm Lai Option C (Nếu lỗi Checker nặng -> Từ chối; Nếu không -> Trả về câu trả lời).
        """
        if not query or not query.strip():
            return {
                "answer": "Vui lòng nhập câu hỏi để tôi có thể hỗ trợ tra cứu.",
                "citations": [],
                "confidence": 0.0,
                "refused": True,
                "category": "empty_query",
                "retrieved_chunks": []
            }

        # ---------------------------------------------------------------------
        # TẦNG 1 REFUSAL: TIỀN KIỂM Ý ĐỊNH CÂU HỎI
        # ---------------------------------------------------------------------
        intent = self.refusal_detector.detect_query_refusal(query)
        if intent.refuse:
            logger.info(f"Từ chối Tầng 1 (Query Intent) câu hỏi: '{query}'. Lý do: {intent.reason}")
            return {
                "answer": intent.reason,
                "citations": [],
                "confidence": 0.0,
                "refused": True,
                "category": intent.category,
                "retrieved_chunks": []
            }

        # ---------------------------------------------------------------------
        # TRUY HỒI THÔNG TIN (RETRIEVAL & RERANKING)
        # ---------------------------------------------------------------------
        retrieved_chunks = self.retrieval_pipeline.retrieve(query, strategy=strategy, top_k=top_k)

        # ---------------------------------------------------------------------
        # TẦNG 2 REFUSAL: TRUNG KIỂM ĐIỂM SỐ TƯƠNG ĐỒNG TOP-1
        # ---------------------------------------------------------------------
        ret_refusal = self.refusal_detector.detect_retrieval_refusal(retrieved_chunks, strategy=strategy)
        if ret_refusal["refuse"]:
            logger.info(f"Từ chối Tầng 2 (Retrieval Score) câu hỏi: '{query}'. Lý do: {ret_refusal['reason']}")
            return {
                "answer": ret_refusal["reason"],
                "citations": [],
                "confidence": 0.0,
                "refused": True,
                "category": ret_refusal["category"],
                "retrieved_chunks": retrieved_chunks
            }

        # ---------------------------------------------------------------------
        # THÀNH PHẦN 6: CONTEXT BUILDER (GỘP CHUNKS & ĐỊNH DẠNG CHUẨN)
        # ---------------------------------------------------------------------
        context_text, merged_chunks = self.context_builder.build_context(retrieved_chunks)

        # ---------------------------------------------------------------------
        # THÀNH PHẦN 7: GENERATOR (GEMINI 2.5 FLASH SINH RAGRESPONSE)
        # ---------------------------------------------------------------------
        response = self.generator.generate_answer(query, merged_chunks, context_text)

        # ---------------------------------------------------------------------
        # TẦNG 3 REFUSAL (PART 1): KIỂM TRA GENERATOR KHÔNG TỰ TIN
        # ---------------------------------------------------------------------
        unconfident_refusal = self.refusal_detector.detect_output_refusal(response, {"is_valid": True, "errors": []})
        if unconfident_refusal["refuse"]:
            logger.info(f"Từ chối Tầng 3 (Unconfident Generation) câu hỏi: '{query}'.")
            return {
                "answer": unconfident_refusal["reason"],
                "citations": [],
                "confidence": 0.0,
                "refused": True,
                "category": unconfident_refusal["category"],
                "retrieved_chunks": retrieved_chunks
            }

        # ---------------------------------------------------------------------
        # THÀNH PHẦN 8 & 9: HẬU KIỂM TĨNH VÀ ĐỘNG (CHECKERS)
        # ---------------------------------------------------------------------
        citation_report = self.citation_checker.check_citations(response, merged_chunks)
        faithfulness_report = self.faithfulness_checker.check_faithfulness(response, query=query)
        disclaimer_report = self.faithfulness_checker.check_disclaimer(response)

        # Hợp nhất toàn bộ lỗi kiểm định của bộ đôi Checkers
        report_errors = []
        report_errors.extend(citation_report.get("errors", []))
        
        if not faithfulness_report.get("is_faithful", True):
            for conflict in faithfulness_report.get("conflicts", []):
                report_errors.append(conflict.get("error_type", "severe_unfaithfulness"))
                
        report_errors.extend(disclaimer_report.get("errors", []))
        
        unified_report = {
            "is_valid": citation_report.get("is_valid", True) and \
                        faithfulness_report.get("is_faithful", True) and \
                        disclaimer_report.get("has_disclaimer", True),
            "errors": list(set(report_errors))
        }

        # ---------------------------------------------------------------------
        # TẦNG 3 REFUSAL (PART 2): HẬU KIỂM LAI OPTION C (Từ chối khi Checker lỗi nặng)
        # ---------------------------------------------------------------------
        final_refusal = self.refusal_detector.detect_output_refusal(response, unified_report)
        if final_refusal["refuse"]:
            logger.warning(f"Từ chối Tầng 3 (Critical Validation Errors) câu hỏi: '{query}'. Lỗi phát hiện: {unified_report['errors']}")
            return {
                "answer": final_refusal["reason"],
                "citations": [],
                "confidence": 0.0,
                "refused": True,
                "category": final_refusal["category"],
                "retrieved_chunks": retrieved_chunks
            }

        # ---------------------------------------------------------------------
        # PHẢN HỒI RAG THÀNH CÔNG HOÀN MỸ
        # ---------------------------------------------------------------------
        citations = [
            c.model_dump() if hasattr(c, "model_dump") else c.dict()
            for c in (response.citations or [])
        ]

        return {
            "answer": response.answer,
            "citations": citations,
            "confidence": response.confidence,
            "refused": False,
            "category": "in_scope",
            "retrieved_chunks": retrieved_chunks
        }
