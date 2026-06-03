import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ContextBuilder:
    """
    Thành phần 6: Context Builder trong Đặc tả hệ thống ViLaborRAG.
    Đóng vai trò quyết định chunk nào được đưa vào prompt và định dạng ngữ cảnh chuẩn pháp luật.
    """
    def __init__(self, max_chunks: int = 5):
        self.max_chunks = max_chunks

    def build_context(self, retrieved_chunks: list[dict]) -> tuple[str, list[dict]]:
        """
        Dựng khối văn bản Context định dạng chuẩn theo đặc tả:
        1. Lấy tối đa top_k chunks.
        2. Nếu nhiều chunk thuộc cùng một Điều luật, tự động gộp nội dung của chúng.
        3. Định dạng văn bản ngữ cảnh chính xác cấu trúc [Nguồn X] được chỉ định.
        
        Trả về:
            context_text (str): Khối ngữ cảnh chuẩn hóa định dạng sẵn.
            merged_chunks (list[dict]): Danh sách các chunk đã được gộp.
        """
        if not retrieved_chunks:
            return "", []

        # 1. Chỉ lấy tối đa top_k chunk chất lượng nhất
        top_chunks = retrieved_chunks[:self.max_chunks]

        # 2. Thuật toán gộp thông minh các chunk cùng thuộc một Điều luật (Article)
        merged_chunks = []
        seen_articles = {}  # Map: tên Điều luật (chuẩn hóa) -> index trong merged_chunks
        
        for chunk in top_chunks:
            article = chunk.get("article")
            text = chunk.get("text", "").strip()
            if not text:
                continue
                
            # Nếu có Điều luật và Điều này đã từng xuất hiện trước đó, thực hiện gộp văn bản
            if article:
                article_clean = article.strip()
                if article_clean in seen_articles:
                    idx = seen_articles[article_clean]
                    existing_chunk = merged_chunks[idx]
                    
                    # Chỉ gộp nếu nội dung text khác biệt để tránh trùng lặp
                    if text not in existing_chunk["text"]:
                        # Nối thêm phần văn bản mới bằng nhãn phân tách
                        existing_chunk["text"] += "\n[Đoạn tiếp theo]:\n" + text
                    continue

            # Nếu chưa từng xuất hiện hoặc không có thông tin Điều luật, tạo mới phần tử
            chunk_copy = dict(chunk)
            merged_chunks.append(chunk_copy)
            if article:
                seen_articles[article.strip()] = len(merged_chunks) - 1

        # 3. Dựng chuỗi văn bản ngữ cảnh chuẩn hóa cấu trúc [Nguồn X]
        context_parts = []
        for idx, chunk in enumerate(merged_chunks):
            source_idx = idx + 1
            part = (
                f"[Nguồn {source_idx}]\n"
                f"Văn bản: {chunk.get('document_title', 'Bộ luật Lao động 2019')}\n"
                f"Số hiệu: {chunk.get('law_number', '45/2019/QH14')}\n"
                f"Điều: {chunk.get('article', 'N/A')}\n"
                f"Tiêu đề: {chunk.get('title', 'N/A')}\n"
                f"Nội dung:\n{chunk.get('text', '')}"
            )
            context_parts.append(part)

        context_text = "\n\n".join(context_parts)
        return context_text, merged_chunks
