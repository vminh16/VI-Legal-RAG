class HybridRetriever:
    """
    Bộ truy hồi lai ghép (Hybrid Retriever) sử dụng giải thuật Reciprocal Rank Fusion (RRF).
    Kết hợp bảng xếp hạng từ BM25 và Dense, tính điểm số phạt thứ hạng RRF
    và đưa các chunk có sự tương thích cao ở cả hai phương thức lên top đầu.
    """
    def __init__(self, k_rrf: int = 60):
        self.k_rrf = k_rrf

    def fuse(
        self,
        bm25_results: list[dict],
        dense_results: list[dict],
        top_k: int = 5
    ) -> list[dict]:
        """
        Lai ghép kết quả tìm kiếm sử dụng công thức RRF:
        RRF_Score(d) = sum( 1 / (k_rrf + rank_i(d)) )
        """
        rrf_scores = {}
        chunks_map = {}

        # 1. Ghi nhận thứ hạng từ BM25
        for idx, chunk in enumerate(bm25_results):
            chunk_id = chunk["chunk_id"]
            rank = idx + 1  # 1-indexed rank
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self.k_rrf + rank)
            chunks_map[chunk_id] = chunk

        # 2. Ghi nhận thứ hạng từ Dense
        for idx, chunk in enumerate(dense_results):
            chunk_id = chunk["chunk_id"]
            rank = idx + 1  # 1-indexed rank
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self.k_rrf + rank)
            if chunk_id not in chunks_map:
                chunks_map[chunk_id] = chunk

        # 3. Đóng gói danh sách kết quả lai ghép
        fused_results = []
        for chunk_id, score in rrf_scores.items():
            chunk_copy = dict(chunks_map[chunk_id])
            chunk_copy["score"] = float(score)  # Ghi đè score nguyên bản bằng RRF score
            fused_results.append(chunk_copy)

        # 4. Sắp xếp giảm dần theo RRF score
        fused_results.sort(key=lambda x: x["score"], reverse=True)

        return fused_results[:top_k]
