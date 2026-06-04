from sentence_transformers import CrossEncoder

class Reranker:
    """
    Bộ tái xếp hạng (Reranker) sử dụng mô hình Cross-Encoder đa ngữ.
    Cross-Encoder đánh giá tương tác ngữ nghĩa sâu sắc giữa cặp (câu hỏi, văn bản),
    giúp lọc nhiễu và đẩy bằng chứng pháp lý chuẩn xác nhất lên hàng đầu.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        # Khởi tạo mô hình Cross-Encoder (tự động download hoặc load cache HF)
        self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """
        Tái xếp hạng danh sách các chunk đầu vào:
        1. Tạo các cặp (query, chunk.text).
        2. Tính toán điểm số tương quan chéo bằng Cross-Encoder.
        3. Sắp xếp lại danh sách chunk và lấy ra top-K.
        """
        if not query or not chunks:
            return []

        # 1. Chuẩn bị các cặp dữ liệu đầu vào cho Cross-Encoder
        pairs = [[query, chunk["text"]] for chunk in chunks]

        # 2. Dự đoán điểm số tương quan sâu
        scores = self.model.predict(pairs, show_progress_bar=False)

        # 3. Đóng gói điểm số mới vào bản sao của chunks
        reranked_chunks = []
        for idx, chunk in enumerate(chunks):
            chunk_copy = dict(chunk)
            chunk_copy["score"] = float(scores[idx])  # Ghi đè bằng điểm số tương quan sâu
            reranked_chunks.append(chunk_copy)

        # 4. Sắp xếp giảm dần theo điểm số Cross-Encoder
        reranked_chunks.sort(key=lambda x: x["score"], reverse=True)

        return reranked_chunks[:top_k]
