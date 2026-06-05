import json
import logging
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, LLM_MODEL_NAME
from src.generation.generator import RAGResponse

logger = logging.getLogger(__name__)

class RefusalJudgment(BaseModel):
    refuse: bool = Field(
        ..., 
        description="True nếu câu hỏi ngoài phạm vi Bộ luật Lao động 2019 hoặc chứa yêu cầu bị cấm, ngược lại False."
    )
    reason: str = Field(
        ..., 
        description="Lý do từ chối cụ thể hoặc lý do đồng ý."
    )
    category: str = Field(
        ..., 
        description="Phân loại: 'out_of_scope', 'administrative_fine', 'needs_decree', 'personal_verdict', 'in_scope'."
    )


class RefusalDetector:
    """
    Module phòng thủ Từ chối thông minh 3 tầng (3-Layer Refusal Defense System).
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Thất bại khi khởi tạo genai.Client trong RefusalDetector: {e}")
                self.client = None

    def detect_query_refusal(self, query: str) -> RefusalJudgment:
        """
        Tầng 1: Tiền kiểm Ý định Câu hỏi trước khi truy hồi.
        Phát hiện câu hỏi ngoài phạm vi, mức phạt tiền hành chính, luật khác hoặc phán thắng thua.
        """
        if not query or not query.strip():
            return RefusalJudgment(
                refuse=True,
                reason="Vui lòng nhập câu hỏi để tôi hỗ trợ tra cứu.",
                category="empty_query"
            )

        # Prompt phân loại ý định chi tiết
        prompt = f"""Bạn là một chuyên gia phân loại ý định câu hỏi về Bộ luật Lao động Việt Nam 2019.
Hãy phân tích câu hỏi sau đây để xác định xem nó có hợp lệ để hệ thống RAG tra cứu trả lời hay không.

CÂU HỎI CỦA NGƯỜI DÙNG:
"{query}"

CÁC TIÊU CHÍ PHÂN LOẠI & TỪ CHỐI BẮT BUỘC:
1. TRƯỜNG HỢP HỢP LỆ (in_scope):
   - Câu hỏi hỏi trực tiếp về quyền, nghĩa vụ, thời giờ làm việc, thời giờ nghỉ ngơi, thử việc, chấm dứt hợp đồng, tiền lương (các nguyên tắc chung) thuộc Bộ luật Lao động 2019.
   - Trả về: refuse=False, category="in_scope".
2. NGOÀI PHẠM VI (out_of_scope):
   - Hỏi về nấu ăn, công nghệ, visa du học, hoặc các bộ luật khác như Luật Hình sự, Luật Dân sự (thủ tục ly hôn), Luật Đất đai...
   - Trả về: refuse=True, category="out_of_scope" kèm lý do giải thích lịch sự.
3. MỨC PHẠT HÀNH CHÍNH CỤ THỂ (administrative_fine):
   - Hỏi về số tiền bị phạt cụ thể (ví dụ: 'phạt bao nhiêu tiền', 'phạt bao nhiêu triệu') khi vi phạm đóng bảo hiểm, vi phạm giờ làm việc. Bộ luật Lao động 2019 chỉ quy định khung hành vi cấm, còn mức phạt hành chính cụ thể thuộc về Nghị định 12/2022/NĐ-CP.
   - Trả về: refuse=True, category="administrative_fine" kèm giải thích rõ rằng Bộ luật Lao động không quy định số tiền phạt, mà phải tra cứu Nghị định 12/2022/NĐ-CP.
4. YÊU CẦU NGHỊ ĐỊNH CHI TIẾT SÂU (needs_decree):
   - Hỏi về các biểu mẫu, quy trình kỹ thuật rất sâu không quy định ở cấp Luật mà thuộc thẩm quyền của Nghị định/Thông tư hướng dẫn chi tiết.
   - Trả về: refuse=True, category="needs_decree".
5. YÊU CẦU PHÁN QUYẾT CÁ NHÂN (personal_verdict):
   - Hỏi bắt buộc khẳng định bên nào thắng kiện 100%, hoặc phán quyết tranh chấp cụ thể của họ.
   - Trả về: refuse=True, category="personal_verdict".

Hãy trả về cấu trúc JSON khớp chính xác tuyệt đối với Pydantic schema được định nghĩa."""

        try:
            if not self.client:
                # Kích hoạt trực tiếp khối xử lý ngoại lệ (fallback offline)
                raise ValueError("API Client chưa được kết nối")

            config = types.GenerateContentConfig(
                system_instruction="Bạn là Trợ lý phân loại ý định câu hỏi tư vấn pháp lý chuyên nghiệp.",
                response_mime_type="application/json",
                response_schema=RefusalJudgment,
                temperature=0.0
            )
            
            response = self.client.models.generate_content(
                model=LLM_MODEL_NAME,
                contents=prompt,
                config=config
            )
            
            if not response.text:
                raise ValueError("Gemini API trả về kết quả phân loại rỗng.")
                
            data = json.loads(response.text)
            return RefusalJudgment(**data)

        except Exception as e:
            logger.error(f"Lỗi khi phân loại ý định câu hỏi qua Gemini API: {e}")
            # --- CƠ CHẾ BẢO VỆ OFFLINE (FALLBACK) ---
            # Tự phân loại thô các từ khóa phổ biến trong test và đời thực
            query_clean = query.lower()
            if "nấu" in query_clean or "phở" in query_clean or "cơm" in query_clean:
                return RefusalJudgment(
                    refuse=True,
                    reason="Rất tiếc, câu hỏi về nấu ăn nằm ngoài phạm vi Bộ luật Lao động 2019.",
                    category="out_of_scope"
                )
            if "phạt" in query_clean and ("tiền" in query_clean or "nhiêu" in query_clean or "mức" in query_clean):
                return RefusalJudgment(
                    refuse=True,
                    reason="Bộ luật Lao động 2019 chỉ quy định khung hành vi, mức xử phạt hành chính cụ thể nằm trong Nghị định 12/2022/NĐ-CP.",
                    category="administrative_fine"
                )
            if "ly hôn" in query_clean or "dân sự" in query_clean:
                return RefusalJudgment(
                    refuse=True,
                    reason="Câu hỏi thuộc lĩnh vực hôn nhân gia đình (Luật Hôn nhân và Gia đình), nằm ngoài phạm vi Bộ luật Lao động 2019.",
                    category="out_of_scope"
                )
            
            # Mặc định chấp nhận đi tiếp nếu lỗi kỹ thuật không phân loại được
            return RefusalJudgment(
                refuse=False,
                reason="Lỗi kết nối phân loại, tiếp tục đi vào RAG pipeline để tra cứu.",
                category="in_scope"
            )

    def detect_retrieval_refusal(self, retrieved_chunks: list[dict], strategy: str = "hybrid") -> dict:
        """
        Tầng 2: Trung kiểm Điểm số tương đồng truy hồi.
        Từ chối khi CSDL không chứa bài viết liên quan hoặc điểm tương đồng quá thấp.
        """
        if not retrieved_chunks:
            return {
                "refuse": True,
                "reason": "Dựa trên Bộ luật Lao động 2019 và dữ liệu ngữ cảnh được cung cấp, tôi không tìm thấy căn cứ pháp lý liên quan để trả lời câu hỏi của bạn.",
                "category": "no_relevant_context"
            }

        # Lấy điểm số của chunk cao nhất
        top_chunk = retrieved_chunks[0]
        score = top_chunk.get("score", 0.0)
        
        strategy = strategy.lower().strip()
        refuse = False
        reason = ""
        
        # Áp dụng ngưỡng điểm sàn riêng biệt cho mỗi chiến lược tìm kiếm
        if strategy == "dense":
            if score < 0.35:
                refuse = True
                reason = f"Độ tương đồng ngữ nghĩa của ngữ cảnh tìm thấy quá thấp ({score:.3f} < 0.350), không đủ tin cậy để trả lời pháp lý."
        elif strategy == "bm25":
            if score < 1.0:
                refuse = True
                reason = f"Điểm số liên quan từ khóa của ngữ cảnh quá thấp ({score:.3f} < 1.000), không đủ căn cứ để đối chiếu."
        elif strategy == "hybrid":
            if score < 0.012:
                refuse = True
                reason = f"Điểm số gộp RRF của ngữ cảnh quá thấp ({score:.4f} < 0.0120), không đủ căn cứ vững chắc."
        elif strategy == "hybrid_rerank":
            if score < -2.0:
                refuse = True
                reason = f"Điểm tái xếp hạng Cross-Encoder quá thấp ({score:.3f} < -2.000), không đủ tin cậy."

        if refuse:
            return {
                "refuse": True,
                "reason": f"Dựa trên Bộ luật Lao động 2019 và dữ liệu được cung cấp, tôi không tìm thấy căn cứ pháp lý liên quan đủ độ tin cậy để trả lời câu hỏi của bạn. (Lý do: {reason})",
                "category": "no_relevant_context"
            }
            
        return {
            "refuse": False,
            "category": "in_scope"
        }

    def detect_output_refusal(self, response: RAGResponse, verification_report: dict) -> dict:
        """
        Tầng 3: Hậu kiểm Lỗi Sinh & Bám nguồn sau khi sinh câu trả lời.
        Chặn hiển thị câu trả lời nếu generator tự tin thấp hoặc checker phát hiện lỗi nặng.
        """
        # 1. Phát hiện generator tự tin bằng 0.0
        if response.confidence == 0.0:
            return {
                "refuse": True,
                "reason": response.answer,  # Sử dụng trực tiếp thông điệp từ chối của generator
                "category": "generation_unconfident"
            }

        # 2. Phát hiện lỗi nghiêm trọng từ bộ Citation & Faithfulness checker
        report_errors = verification_report.get("errors", [])
        critical_errors = [
            "citation_not_found_in_corpus",
            "fabricated_evidence",
            "faithfulness_conflict",
            "severe_unfaithfulness"
        ]
        
        has_critical_error = any(err in report_errors for err in critical_errors)
        if not verification_report.get("is_valid", True) and has_critical_error:
            logger.warning(f"Phát hiện lỗi kiểm định nghiêm trọng ở đầu ra: {report_errors}. Kích hoạt từ chối an toàn.")
            return {
                "refuse": True,
                "reason": "Dựa trên Bộ luật Lao động 2019 và dữ liệu ngữ cảnh được cung cấp, tôi không tìm thấy căn cứ pháp lý liên quan hoặc phát hiện sự mâu thuẫn để trả lời câu hỏi của bạn. Vui lòng đặt lại câu hỏi rõ ràng hơn.",
                "category": "critical_validation_error"
            }
            
        return {
            "refuse": False,
            "category": "in_scope"
        }
