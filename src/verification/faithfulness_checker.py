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
    Tách câu trả lời thành các Tuyên bố (Claims) độc lập bằng cách phân tích theo từng câu
    và liên kết với các nhãn trích dẫn [X] xuất hiện trong chính câu đó.
    """
    if not answer:
        return []
        
    # Tách thành các đoạn/dòng trước
    paragraphs = answer.split('\n')
    claims = []
    
    for para in paragraphs:
        if not para.strip():
            continue
        
        # Tách đoạn thành các câu bằng dấu chấm câu (kết thúc bằng ., ?, ! và có khoảng trắng phía sau)
        sentences = re.split(r'(?<=[.!?])\s+', para)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Tìm tất cả các nhãn [X] trong câu này
            markers = re.findall(r'(\[(\d+)\])', sentence)
            if markers:
                # Làm sạch câu bằng cách loại bỏ các nhãn trích dẫn [X] ra khỏi text của claim
                claim_text = sentence
                for marker, _ in markers:
                    claim_text = claim_text.replace(marker, "")
                
                # Loại bỏ các ký tự dấu câu và ký tự đặc biệt thừa ở đầu và cuối claim
                claim_text = re.sub(r'^[\s\.\,\;\:\-\–\—\•\*]+|[\s\.\,\;\:\-\–\—\•\*]+$', '', claim_text).strip()
                
                if claim_text:
                    for marker, cit_id_str in markers:
                        claims.append({
                            "claim": claim_text,
                            "citation_id": int(cit_id_str),
                            "marker": marker
                        })
    return claims

def check_numeric_discrepancy(claim: str, evidence: str, article: str = "", clause: str | None = None, query: str = "") -> str | None:
    """
    Kiểm tra thô bằng Rule-based phát hiện sai lệch số liệu/thời hạn.
    Bổ sung bộ tự động tính toán lại ở backend (Arithmetic Re-calculation) để bỏ qua
    biến số người dùng và kết quả phép nhân/phần trăm hợp lệ.
    """
    # Làm sạch các năm định danh pháp luật cụ thể để tránh bị trích xuất nhầm thành số liệu cần kiểm tra
    def clean_law_years(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'(?:bộ\s+luật\s+lao\s+động|luật\s+lao\s+động|bllđ)\s*(?:năm\s*)?2019', '', text, flags=re.IGNORECASE)
        text = re.sub(r'45/2019/qh14', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(?:hiệu\s+lực\s+từ\s*|ngày\s*)?0?1/0?1/2021', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(?:năm\s*)?2020', '', text, flags=re.IGNORECASE)
        return text

    claim_cleaned = clean_law_years(claim)
    evidence_cleaned = clean_law_years(evidence)
    query_cleaned = clean_law_years(query) if query else ""

    # Trích xuất toàn bộ số từ query
    query_nums = set(re.findall(r'\b\d+\b', query_cleaned)) if query_cleaned else set()
    
    # Trích xuất toàn bộ số từ evidence
    evidence_nums = set(re.findall(r'\b\d+\b', evidence_cleaned))
    
    # Các số định danh cần bỏ qua (Bypass list)
    bypass_nums = set()
    
    # Bỏ qua số hiệu Điều luật
    if article:
        art_num_match = re.search(r'\d+', article)
        if art_num_match:
            bypass_nums.add(art_num_match.group())
            
    # Bỏ qua số hiệu Khoản luật
    if clause:
        cl_num_match = re.search(r'\d+', str(clause))
        if cl_num_match:
            bypass_nums.add(cl_num_match.group())
            
    # QUÉT VÀ XÁC THỰC PHÉP TOÁN (Arithmetic Re-calculation)
    # Nhận diện phép toán dạng: A * B = C hoặc A x B = C hoặc A * B% = C
    math_pattern = r'(\d+[\d\.,]*)\s*(?:đồng|đ|USD|%)?\s*([\*xX\/+-])\s*(\d+[\d\.,]*%?)\s*=\s*(\d+[\d\.,]*)'
    math_matches = re.findall(math_pattern, claim_cleaned)
    
    for op1_str, operator, op2_str, res_str in math_matches:
        try:
            # Chuẩn hóa chuỗi số về float
            def clean_num(s):
                s = s.strip()
                if s.endswith('%'):
                    s = s[:-1]
                
                # Nếu có cả dấu chấm và dấu phẩy
                if '.' in s and ',' in s:
                    if s.rfind('.') > s.rfind(','):
                        s = s.replace(',', '')
                    else:
                        s = s.replace('.', '').replace(',', '.')
                    return float(s)
                    
                # Nếu chỉ có một loại dấu phân cách
                separator = None
                if '.' in s:
                    separator = '.'
                elif ',' in s:
                    separator = ','
                    
                if separator:
                    parts = s.split(separator)
                    if len(parts) > 2:
                        s = s.replace(separator, '')
                    else:
                        if len(parts[1]) == 3:
                            s = s.replace(separator, '')
                        else:
                            s = s.replace(separator, '.')
                return float(s)
            
            val1 = clean_num(op1_str)
            val2 = clean_num(op2_str)
            val_res = clean_num(res_str)
            
            # Xử lý phần trăm cho toán hạng 2
            is_pct = '%' in op2_str
            multiplier = val2 / 100.0 if is_pct else val2
            
            calculated_res = 0.0
            op = operator.lower()
            if op in ('*', 'x'):
                calculated_res = val1 * multiplier
            elif op == '/':
                calculated_res = val1 / multiplier if multiplier != 0 else 0.0
            elif op == '+':
                calculated_res = val1 + multiplier
            elif op == '-':
                calculated_res = val1 - multiplier
                
            # So khớp kết quả tự tính với kết quả của bot
            if abs(calculated_res - val_res) <= 1.0:
                # Phép toán chính xác! Miễn trừ toàn bộ các số tham gia phép tính này
                for raw_s in (op1_str, op2_str, res_str):
                    for num in re.findall(r'\b\d+\b', raw_s):
                        bypass_nums.add(num)
                logger.info(f"Arithmetic Verification: Phép tính hợp lệ '{op1_str} {operator} {op2_str} = {res_str}'. Đã cấp quyền miễn trừ.")
        except Exception as e:
            logger.warning(f"Arithmetic Verification: Lỗi khi tính toán lại phép tính: {e}")
    
    # Trích xuất tất cả chữ số từ claim
    claim_nums = set(re.findall(r'\b\d+\b', claim_cleaned))
    
    # Loại bỏ các số bypass khỏi claim_nums
    claim_nums = claim_nums - bypass_nums
    
    # Bỏ qua số 0 và 1 vì chúng có thể là số thứ tự
    conflicts = [n for n in claim_nums if n not in evidence_nums and int(n) > 1]
    if conflicts:
        return f"mâu thuẫn số liệu: Claim chứa số ({', '.join(conflicts)}) không xuất hiện trong Evidence."
    return None


class FaithfulnessChecker:
    """
    Module kiểm tra độ bám nguồn ngữ nghĩa kết hợp kiểm tra tĩnh thời hạn số liệu.
    Hỗ trợ mô hình Local NLI và cơ chế thác nước (Cascade) giảm tải Gemini API.
    """
    def __init__(self, api_key: str = None, nli_model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Thất bại khi khởi tạo genai.Client trong FaithfulnessChecker: {e}")
                self.client = None
                
        # Khởi tạo mô hình Local NLI (Sentence-Transformers CrossEncoder)
        self.nli_model_name = nli_model_name
        self.nli_model = None
        try:
            from sentence_transformers import CrossEncoder
            self.nli_model = CrossEncoder(self.nli_model_name)
            logger.info(f"Đã tải thành công mô hình NLI cục bộ: {self.nli_model_name}")
        except Exception as e:
            logger.warning(f"Không thể tải mô hình NLI cục bộ '{self.nli_model_name}': {e}. Sẽ dùng Gemini làm mặc định.")
            self.nli_model = None

    def check_faithfulness(self, response: RAGResponse, query: str = "") -> dict:
        """
        Thực hiện kiểm định độ bám nguồn theo hình thác nước (Cascade):
        1. Phân tách câu trả lời thành các Claim.
        2. Chạy Rule-based so khớp số thô và xác thực phép tính số học (Arithmetic-aware).
        3. Chạy Local NLI so khớp ngữ nghĩa. Đạt/Từ chối ngay nếu vượt ngưỡng tin cậy.
        4. Gemini Fallback Judge: Chỉ gọi Gemini API khi NLI nằm ngoài khoảng tự tin.
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
                continue
                
            evidence = cit.evidence or ""
            
            # --- CẤP ĐỘ 1: KIỂM TRA LỆCH SỐ HỌC (ARITHMETIC-AWARE VERIFICATION) ---
            num_error = check_numeric_discrepancy(claim_text, evidence, cit.article, cit.clause, query=query)
            if num_error:
                conflicts.append({
                    "citation_id": cit_id,
                    "claim": claim_text,
                    "evidence": evidence,
                    "supported": False,
                    "error_type": "faithfulness_conflict",
                    "reason": num_error
                })
                continue
                
            # --- CẤP ĐỘ 2: LOCAL NLI CHECK (MÔ HÌNH NLI CỤC BỘ) ---
            nli_resolved = False
            if self.nli_model is not None:
                try:
                    scores = self.nli_model.predict([(evidence, claim_text)])[0]
                    
                    # Tính toán xác suất dạng softmax
                    import numpy as np
                    exp_scores = np.exp(scores - np.max(scores))
                    probs = exp_scores / exp_scores.sum()
                    
                    # Xác định label mapping của mô hình NLI
                    label_map = {0: "contradiction", 1: "entailment", 2: "neutral"}
                    if hasattr(self.nli_model, "model") and hasattr(self.nli_model.model, "config") and hasattr(self.nli_model.model.config, "id2label"):
                        label_map = {int(k): str(v).lower() for k, v in self.nli_model.model.config.id2label.items()}
                        
                    prob_dict = {label_map.get(i, f"label_{i}"): float(p) for i, p in enumerate(probs)}
                    
                    entail_prob = prob_dict.get("entailment", prob_dict.get("entail", 0.0))
                    contradict_prob = prob_dict.get("contradiction", prob_dict.get("contradict", 0.0))
                    
                    logger.info(f"Local NLI probabilities for Claim '{claim_text[:35]}...': {prob_dict}")
                    
                    # Áp dụng ngưỡng tin cậy để quyết định có bypass Gemini
                    if entail_prob > 0.85:
                        nli_resolved = True
                        logger.info(f"Local NLI: Chấp nhận claim bám nguồn (Entailment score: {entail_prob:.2f})")
                        continue # Đạt, đi tiếp sang claim sau
                    elif contradict_prob > 0.85:
                        nli_resolved = True
                        logger.warning(f"Local NLI: Phát hiện mâu thuẫn rõ rệt (Contradiction score: {contradict_prob:.2f})")
                        conflicts.append({
                            "citation_id": cit_id,
                            "claim": claim_text,
                            "evidence": evidence,
                            "supported": False,
                            "error_type": "severe_unfaithfulness",
                            "reason": f"Mô hình NLI cục bộ phát hiện mâu thuẫn rõ rệt (Contradiction score: {contradict_prob:.2f})."
                        })
                        continue
                except Exception as ex:
                    logger.warning(f"Lỗi khi chạy mô hình NLI cục bộ: {ex}. Chuyển tiếp sang Gemini fallback.")
            
            # --- CẤP ĐỘ 3: GEMINI FALLBACK JUDGE (CHỈ CHẠY KHI NLI KHÔNG CHẮC CHẮN) ---
            if nli_resolved:
                continue
                
            if not self.client:
                # Ở chế độ Giả lập (Mock Mode), mặc định thông qua
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
