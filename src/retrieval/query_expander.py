import json
import logging
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, LLM_MODEL_NAME

logger = logging.getLogger(__name__)

class ExpandedQueries(BaseModel):
    queries: list[str] = Field(
        ...,
        description="Danh sách gồm tối đa 2 câu truy vấn tìm kiếm phụ ngắn gọn bằng tiếng Việt, tập trung vào các từ khóa pháp lý."
    )

class QueryExpander:
    """
    Module sinh các truy vấn tìm kiếm bổ trợ (Query Expansion / Multi-query).
    Giúp tăng diện tích bao phủ tìm kiếm sang các Điều luật bổ trợ có liên quan ngữ nghĩa.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key if api_key is not None else GEMINI_API_KEY
        if not self.api_key:
            logger.warning(
                "Không tìm thấy GEMINI_API_KEY. QueryExpander sẽ chạy ở chế độ Giả lập (Mock Mode)."
            )
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Thất bại khi khởi tạo genai.Client trong QueryExpander: {e}")
                self.client = None

    def expand_query(self, query: str) -> list[str]:
        """
        Sinh 2 truy vấn phụ từ câu hỏi gốc của người dùng.
        Trả về danh sách các truy vấn phụ. Nếu xảy ra lỗi hoặc chạy Mock Mode, trả về danh sách rỗng [].
        """
        if not query or not query.strip():
            return []

        if not self.client:
            # Chế độ Giả lập (Mock Mode) khi không có API Key
            return []

        prompt = f"""Bạn là trợ lý mở rộng câu hỏi tìm kiếm pháp lý. Hãy phân tích câu hỏi sau đây của người dùng và sinh ra tối đa 2 câu truy vấn tìm kiếm (search queries) độc lập dưới dạng danh sách văn bản ngắn gọn, tập trung vào các từ khóa pháp luật chính.
        
        Ví dụ: "Lương đi làm ngày Tết tính thế nào" ->
        - tiền lương làm thêm giờ ngày lễ tết
        - đi làm ngày nghỉ lễ hưởng lương thế nào
        
        CÂU HỎI CỦA NGƯỜI DÙNG:
        "{query}"
        
        Hãy trả về kết quả định dạng JSON khớp chính xác tuyệt đối với cấu trúc Pydantic schema được định nghĩa."""

        try:
            config = types.GenerateContentConfig(
                system_instruction="Bạn là Chuyên gia tối ưu hóa và sinh từ khóa tìm kiếm pháp luật Việt Nam.",
                response_mime_type="application/json",
                response_schema=ExpandedQueries,
                temperature=0.0,
                max_output_tokens=150
            )

            response = self.client.models.generate_content(
                model=LLM_MODEL_NAME,
                contents=prompt,
                config=config
            )

            if not response.text:
                raise ValueError("Gemini API trả về kết quả rỗng.")

            data = json.loads(response.text)
            expanded = ExpandedQueries(**data)
            
            # Làm sạch các chuỗi trả về
            result = [q.strip() for q in expanded.queries if q.strip()]
            logger.info(f"Query '{query}' được mở rộng thành: {result}")
            return result[:2]

        except Exception as e:
            logger.error(f"Lỗi khi mở rộng truy vấn qua Gemini API: {e}")
            return []
