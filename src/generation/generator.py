import json
import logging
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, LLM_MODEL_NAME

# Cấu hình logging
logger = logging.getLogger(__name__)

class Citation(BaseModel):
    citation_id: int = Field(
        ..., 
        description="ID định danh trích dẫn bắt đầu từ 1, tương ứng với số thứ tự [X] được gán trong câu trả lời."
    )
    article: str = Field(
        ..., 
        description="Tên Điều luật được trích dẫn (ví dụ: 'Điều 24')."
    )
    clause: str | None = Field(
        None, 
        description="Số hiệu Khoản luật được trích dẫn (ví dụ: '1' hoặc null nếu trích dẫn cả Điều)."
    )
    title: str = Field(
        ..., 
        description="Tiêu đề chính xác của Điều luật."
    )
    source_url: str = Field(
        ..., 
        description="Đường dẫn nguồn chính thống của Điều luật được trích dẫn."
    )
    evidence: str = Field(
        ..., 
        description="Đoạn văn bản trích dẫn gốc từ ngữ cảnh đóng vai trò làm bằng chứng trực tiếp cho ý khẳng định trong câu trả lời."
    )

class RAGResponse(BaseModel):
    answer: str = Field(
        ..., 
        description="Câu trả lời pháp lý hoàn chỉnh bằng tiếng Việt, được lập luận logic, dễ hiểu và gán chính xác các nhãn trích dẫn dạng [1], [2]..."
    )
    citations: list[Citation] = Field(
        ..., 
        description="Danh sách các nguồn trích dẫn chi tiết đối xứng với các nhãn [1], [2]... được sử dụng trong câu trả lời."
    )
    confidence: float = Field(
        ..., 
        description="Độ tin cậy của câu trả lời từ 0.0 đến 1.0 dựa trên mức độ bao phủ và liên quan của ngữ cảnh được cung cấp."
    )

class GeminiGenerator:
    """
    Module sinh câu trả lời RAG kết hợp trích dẫn nguồn chuẩn xác (Gemini 2.5 Flash).
    Sử dụng SDK chính thức google-genai để sinh cấu trúc đầu ra JSON được xác thực bằng Pydantic.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            logger.warning(
                "Không tìm thấy GEMINI_API_KEY. GeminiGenerator sẽ không thể chạy thực tế trừ khi được nạp khóa."
            )
        
        # Khởi tạo Client của SDK Google GenAI mới nhất
        self.client = genai.Client(api_key=self.api_key)

    def generate_answer(
        self, 
        query: str, 
        retrieved_chunks: list[dict], 
        context_text: str = None
    ) -> RAGResponse:
        """
        Nhận câu hỏi và ngữ cảnh dựng sẵn để sinh câu trả lời có cấu trúc trích dẫn:
        1. Gọi ContextBuilder để dựng khối ngữ cảnh [Nguồn X] chuẩn nếu chưa có.
        2. Tạo prompt chỉ thị nghiêm ngặt hướng tới Phương án 2.
        3. Gọi Gemini API với response_schema ràng buộc bằng Pydantic RAGResponse.
        """
        if not query:
            return RAGResponse(
                answer="Vui lòng nhập câu hỏi để tôi có thể hỗ trợ tra cứu.",
                citations=[],
                confidence=0.0
            )

        # Trường hợp không tìm thấy bất kỳ chunk luật nào có liên quan từ retriever
        if not retrieved_chunks:
            return RAGResponse(
                answer="Dựa trên Bộ luật Lao động 2019 và dữ liệu được cung cấp, tôi không tìm thấy căn cứ pháp lý liên quan để trả lời câu hỏi của bạn.",
                citations=[],
                confidence=0.0
            )

        # 1. Dựng khối văn bản ngữ cảnh (Context) chuẩn hóa nếu chưa có truyền vào từ ngoài
        if not context_text:
            from src.retrieval.context_builder import ContextBuilder
            builder = ContextBuilder()
            context_text, _ = builder.build_context(retrieved_chunks)

        # 2. Tạo prompt chỉ thị khắt khe hướng tới Phương án 2 và Tông giọng Trung lập, Giải thích dễ hiểu
        prompt = f"""Bạn là một Trợ lý Tư vấn Pháp lý Lao động Việt Nam cao cấp. Hãy trả lời câu hỏi sau đây dựa trên các tài liệu thuộc Ngữ cảnh pháp lý được cung cấp bên dưới.

CÂU HỎI TRUY VẤN:
"{query}"

NGỮ CẢNH PHÁP LÝ (CHỈ ĐƯỢC PHÉP SỬ DỤNG THÔNG TIN NÀY):
{context_text}

CÁC QUY TẮC PHÁP LÝ BẮT BUỘC KHI SINH CÂU TRẢ LỜI:
1. TUYỆT ĐỐI BÁM NGUỒN (CHỐNG ẢO TƯỞNG): Chỉ được phép sử dụng thông tin trong Ngữ cảnh được cung cấp ở trên. Tuyệt đối không tự suy diễn hay dùng tri thức bên ngoài. Nếu Ngữ cảnh không có thông tin phù hợp hoặc thiếu căn cứ rõ ràng để trả lời câu hỏi, đặt confidence = 0.0 và trả về câu trả lời từ chối lịch sự: "Dựa trên Bộ luật Lao động 2019 và dữ liệu được cung cấp, tôi không tìm thấy căn cứ pháp lý liên quan để trả lời câu hỏi của bạn."
2. GIẢI THÍCH DỄ HIỂU & TRÁNH SAO CHÉP MÁY MÓC:
   - Tuyệt đối TRÁNH việc copy-paste nguyên văn các câu chữ luật pháp quá dài dòng, khô khan và phức tạp từ ngữ cảnh sang câu trả lời.
   - Hãy diễn giải lại (paraphrase) bằng ngôn ngữ đời thường, gần gũi, dễ hiểu nhất cho người hỏi.
   - BẮT BUỘC GIỮ NGUYÊN THUẬT NGỮ CỐT LÕI: Khi diễn giải, bắt buộc phải bảo toàn các thuật ngữ định danh pháp lý chính thức của Bộ luật (ví dụ: tên loại hợp đồng lao động như 'hợp đồng lao động không xác định thời hạn', tên các chế độ như 'trợ cấp thôi việc', 'trợ cấp mất việc làm', tên các cơ quan hoặc các con số thời hạn luật định cụ thể như '30 ngày', 'nhiều nhất 02 lần'). Tuyệt đối không tự tiện bình dân hóa các cụm từ chuyên môn định danh này làm mất đi tính chuẩn xác pháp lý.
3. TÔNG GIỌNG TRUNG LẬP & TRÁNH TUYÊN BỐ KHẲNG ĐỊNH (NO CLAIM):
   - Sử dụng tông giọng khách quan, trung lập, nghiêm túc và chuyên nghiệp. Tuyệt đối TRÁNH sử dụng các từ ngữ mang tính cảm xúc cá nhân, chủ quan, đùa cợt, hoặc hoa mỹ, sáo rỗng không cần thiết.
   - CẤM CAM KẾT PHÁN QUYẾT: Tuyệt đối không đưa ra các khẳng định chắc chắn về kết quả vụ việc hoặc cam kết thắng thua cho người dùng (ví dụ: cấm dùng các câu như "bạn chắc chắn sẽ thắng kiện", "công ty bạn chắc chắn 100% sai"). Thay vào đó, hãy giải thích rõ ràng: "Theo quy định tại Điều..., người lao động có quyền..." hoặc "Hành vi này có thể có dấu hiệu vi phạm quy định về...".
   - CẤM CAM KẾT SỐ TIỀN CỤ THỂ: Khi người hỏi đưa ra câu hỏi tình huống tính tiền (trợ cấp, bồi thường), tuyệt đối không tự tiện tính ra một con số tiền cụ thể rồi cam kết với người dùng (ví dụ: cấm nói 'Bạn sẽ được nhận đúng 25 triệu'). Hãy cung cấp công thức tính pháp định từ ngữ cảnh, hướng dẫn cách tính chi tiết và chỉ rõ các biến số thực tế cần phải xác minh thêm (như thời gian đóng bảo hiểm thất nghiệp, mức lương bình quân 06 tháng liền kề).
4. XỬ LÝ THÔNG TIN THIẾU HỤT: Nếu câu hỏi của người dùng có các chi tiết phức tạp mà ngữ cảnh được cung cấp không đủ dữ kiện để khẳng định chắc chắn, hãy giải thích rõ ràng câu trả lời ở mức độ thông tin hiện có, và liệt kê các điều kiện hoặc hồ sơ/giấy tờ cần xác minh thêm để người hỏi tự đối chiếu.
5. LẬP LUẬN TẬP TRUNG & DỄ HIỂU (PHƯƠNG ÁN 2):
   - NGAY CÂU ĐẦU TIÊN: Đưa ra câu trả lời tập trung, trực diện và ngắn gọn nhất có thể đối với câu hỏi của người dùng để họ nắm được cốt lõi vấn đề ngay lập tức.
   - PHẦN TIẾP THEO: Phân tách rõ ràng thành các gạch đầu dòng ngắn gọn liệt kê cụ thể các điều kiện, thời hạn hoặc ràng buộc pháp lý đi kèm để người dùng dễ dàng theo dõi và ghi nhớ.
6. TUYÊN BỐ MIỄN TRỪ TRÁCH NHIỆM PHÁP LÝ (DEFAULT DISCLAIMER): Ở cuối câu trả lời (sau các gạch đầu dòng), bắt buộc phải có một câu tuyên bố miễn trừ độc lập, xuống dòng rõ ràng: "Lưu ý: Ý kiến tư vấn trên chỉ mang tính chất tham khảo dựa trên quy định của Bộ luật Lao động 2019 và dữ liệu ngữ cảnh hiện có tại thời điểm tra cứu."
7. GẮN NHÃN TRÍCH DẪN VÀ ĐỐI CHIẾU CHI TIẾT:
   - Mỗi tuyên bố pháp lý hoặc số liệu được nêu trong câu trả lời bắt buộc phải được gắn nhãn trích dẫn dạng [1], [2]... ở cuối câu hoặc ý tương ứng với citation_id của nguồn tài liệu trong danh sách citations.
   - Mỗi nhãn [X] xuất hiện trong câu trả lời phải có một Citation tương ứng được khai báo trong danh sách "citations", bắt đầu từ 1. Mỗi Citation phải trích xuất chính xác "evidence" là văn bản gốc làm bằng chứng trực tiếp từ ngữ cảnh.

Hãy trả về kết quả khớp chính xác tuyệt đối với cấu trúc JSON Schema được định nghĩa."""

        try:
            # 3. Cấu hình Structured Output JSON
            config = types.GenerateContentConfig(
                system_instruction="Bạn là Chuyên gia tư vấn pháp lý cao cấp về Bộ luật Lao động Việt Nam 2019.",
                response_mime_type="application/json",
                response_schema=RAGResponse,
                temperature=0.0  # Triệt tiêu tính ngẫu nhiên để tối đa hóa tính nhất quán pháp lý
            )

            # Gọi Gemini API sử dụng SDK mới
            response = self.client.models.generate_content(
                model=LLM_MODEL_NAME,
                contents=prompt,
                config=config
            )

            # 4. Phân tích kết quả JSON trả về
            if not response.text:
                raise ValueError("Gemini API trả về kết quả trống rỗng.")

            data = json.loads(response.text)
            return RAGResponse(**data)

        except Exception as e:
            logger.error(f"Lỗi trong quá trình sinh câu trả lời bằng Gemini API: {e}")
            # Trả về response an toàn khi xảy ra lỗi kết nối hoặc API
            return RAGResponse(
                answer=f"Hệ thống đang gặp sự cố kỹ thuật khi kết nối với mô hình ngôn ngữ lớn (Gemini). Lỗi: {str(e)}",
                citations=[],
                confidence=0.0
            )
