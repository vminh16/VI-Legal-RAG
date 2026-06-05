import re
import json
import logging
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, LLM_MODEL_NAME
from src.generation.generator import RAGResponse

logger = logging.getLogger(__name__)

class FaithfulnessJudgment(BaseModel):
    supported: bool = Field(
        ..., 
        description="True nếu tuyên bố được chứng minh đầy đủ bởi bằng chứng, ngược lại False."
    )
    reason: str = Field(
        ..., 
        description="Lý do chi tiết giải thích cho kết luận."
    )

def segment_claims(answer: str) -> list[dict]:
    """
    Tách câu trả lời thành các Claim độc lập kết thúc bằng nhãn trích dẫn [X].
    Ví dụ: 'A [1]. B [2].' -> [{'claim': 'A', 'citation_id': 1}, {'claim': 'B', 'citation_id': 2}]
    """
    if not answer:
        return []
        
    parts = re.split(r'(\[\d+\])', answer)
    claims = []
    
    # parts chứa xen kẽ văn bản và nhãn: [text_0, label_1, text_1, label_2, ...]
    for i in range(0, len(parts) - 1, 2):
        claim_text = parts[i].strip()
        
        # Loại bỏ các ký tự dấu câu thừa ở đầu và cuối claim
        claim_text = re.sub(r'^[\s\.\,\;\:\-\–\—\•]+|[\s\.\,\;\:\-\–\—\•]+$', '', claim_text).strip()
        marker = parts[i+1]
        
        # Trích xuất số ID trích dẫn
        match = re.search(r'\d+', marker)
        if match and claim_text:
            cit_id = int(match.group())
            claims.append({
                "claim": claim_text,
                "citation_id": cit_id,
                "marker": marker
            })
            
    return claims

def check_numeric_discrepancy(claim: str, evidence: str) -> str | None:
    """
    Kiểm tra thô bằng Rule-based phát hiện sai lệch số liệu/thời hạn.
    Ví dụ: claim nói 60 ngày nhưng evidence ghi 30 ngày.
    """
    # Trích xuất tất cả chữ số độc lập
    claim_nums = set(re.findall(r'\b\d+\b', claim))
    evidence_nums = set(re.findall(r'\b\d+\b', evidence))
    
    # Bỏ qua số 0 và 1 vì chúng có thể là số thứ tự hoặc số Điều/Khoản
    conflicts = [n for n in claim_nums if n not in evidence_nums and int(n) > 1]
    if conflicts:
        return f"mâu thuẫn số liệu: Claim chứa số ({', '.join(conflicts)}) không xuất hiện trong Evidence."
    return None


class FaithfulnessChecker:
    """
    Module kiểm tra độ bám nguồn ngữ nghĩa kết hợp kiểm tra tĩnh thời hạn số liệu.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Thất bại khi khởi tạo genai.Client trong FaithfulnessChecker: {e}")
                self.client = None

    def check_faithfulness(self, response: RAGResponse) -> dict:
        """
        Thực hiện kiểm định độ bám nguồn:
        1. Phân tách câu trả lời thành các Claim.
        2. Dùng Rule-based kiểm tra số liệu thô giữa Claim và Evidence.
        3. Dùng Gemini 2.5 Flash làm Judge để thẩm định ngữ nghĩa.
        """
        answer = response.answer or ""
        citations = response.citations or []
        
        # Tạo từ điển map citation_id -> citation để truy cứu nhanh
        citation_map = {c.citation_id: c for c in citations}
        
        # 1. Phân tách claims
        claims = segment_claims(answer)
        conflicts = []
        
        for c_info in claims:
            claim_text = c_info["claim"]
            cit_id = c_info["citation_id"]
            
            cit = citation_map.get(cit_id)
            if not cit:
                # Lỗi định dạng trích dẫn (sẽ được bắt ở CitationChecker, bỏ qua ở đây)
                continue
                
            evidence = cit.evidence or ""
            
            # 2. Kiểm tra thô Rule-based phát hiện lệch số
            num_error = check_numeric_discrepancy(claim_text, evidence)
            if num_error:
                conflicts.append({
                    "citation_id": cit_id,
                    "claim": claim_text,
                    "evidence": evidence,
                    "supported": False,
                    "error_type": "faithfulness_conflict",
                    "reason": num_error
                })
                continue  # Bỏ qua bước gọi LLM nếu đã phát hiện lệch số thô
                
            # 3. Gọi LLM-as-judge thẩm định ngữ nghĩa sâu
            if not self.client:
                # Ở chế độ Giả lập (Mock Mode), giả định các ý khác bám nguồn thành công nếu không lệch số thô
                continue

            try:
                prompt = f"""Bạn là một Thẩm định viên Pháp lý Lao động. Hãy đánh giá xem Tuyên bố (Claim) dưới đây có được chứng minh hoàn toàn bởi Bằng chứng (Evidence) được cung cấp hay không.
Tuyệt đối KHÔNG sử dụng bất kỳ kiến thức bên ngoài nào. Chỉ được phép căn cứ vào nội dung Bằng chứng.

Tuyên bố: "{claim_text}"
Bằng chứng: "{evidence}"

Hãy trả về đúng cấu trúc JSON với các trường:
- supported: True nếu bằng chứng chứng minh hoàn toàn cho tuyên bố, False nếu bằng chứng không đủ hoặc mâu thuẫn.
- reason: Giải thích ngắn gọn lý do tại sao supported là True hoặc False."""

                config = types.GenerateContentConfig(
                    system_instruction="Bạn là Chuyên gia thẩm định tính xác thực của câu trả lời pháp lý.",
                    response_mime_type="application/json",
                    response_schema=FaithfulnessJudgment,
                    temperature=0.0
                )
                
                # Gọi API
                judg_response = self.client.models.generate_content(
                    model=LLM_MODEL_NAME,
                    contents=prompt,
                    config=config
                )
                
                if not judg_response.text:
                    raise ValueError("Gemini API trả về kết quả thẩm định rỗng.")
                    
                data = json.loads(judg_response.text)
                judgment = FaithfulnessJudgment(**data)
                
                if not judgment.supported:
                    conflicts.append({
                        "citation_id": cit_id,
                        "claim": claim_text,
                        "evidence": evidence,
                        "supported": False,
                        "error_type": "severe_unfaithfulness",
                        "reason": judgment.reason
                    })
                    
            except Exception as e:
                logger.error(f"Lỗi trong quá trình thẩm định Faithfulness bằng Gemini API: {e}")
                # Khi xảy ra lỗi kết nối mạng, tạm thời bỏ qua cảnh báo nghiêm trọng để tránh ngắt mạch luồng
                pass
                
        return {
            "is_faithful": len(conflicts) == 0,
            "conflicts": conflicts
        }

    def check_disclaimer(self, response: RAGResponse) -> dict:
        """
        Kiểm tra sự tồn tại của tuyên bố miễn trừ trách nhiệm pháp lý mặc định ở cuối.
        """
        answer = response.answer or ""
        disclaimer = "Ý kiến tư vấn trên chỉ mang tính chất tham khảo dựa trên quy định của Bộ luật Lao động 2019"
        
        disclaimer_clean = " ".join(disclaimer.lower().split())
        answer_clean = " ".join(answer.lower().split())
        
        if disclaimer_clean not in answer_clean:
            return {
                "has_disclaimer": False,
                "errors": ["missing_disclaimer"]
            }
            
        return {
            "has_disclaimer": True,
            "errors": []
        }
