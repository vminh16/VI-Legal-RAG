from pathlib import Path
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_expander import QueryExpander
from src.reranking.reranker import Reranker

class RetrievalPipeline:
    """
    Pipeline truy hồi hợp nhất (Retrieval Pipeline) của dự án ViLaborRAG.
    Đóng vai trò là API trung tâm để điều phối và chạy thử nghiệm so sánh 4 chiến lược:
    1. bm25: So khớp từ khóa chính xác.
    2. dense: Tìm kiếm ngữ nghĩa ngữ cảnh (BGE-M3).
    3. hybrid: Tìm kiếm lai ghép kết hợp từ khóa và ngữ nghĩa (RRF).
    4. hybrid_rerank: Tìm kiếm lai ghép kết hợp Cross-Encoder Reranker tái xếp hạng.
    """
    def __init__(
        self,
        chunks: list[dict] = None,
        corpus_path: str | Path = None,
        embedding_model: str = "BAAI/bge-m3",
        reranker_model: str = "BAAI/bge-reranker-base",
        index_dir: str | Path = "data/processed",
        force_rebuild: bool = False
    ):
        self.reranker_model = reranker_model
        
        # 1. Khởi tạo Retriever Từ khóa (BM25)
        self.bm25_retriever = BM25Retriever(chunks=chunks, corpus_path=corpus_path)
        
        # 2. Khởi tạo Retriever Ngữ nghĩa (Dense FAISS)
        self.dense_retriever = DenseRetriever(
            chunks=chunks,
            corpus_path=corpus_path,
            model_name=embedding_model,
            index_dir=index_dir,
            force_rebuild=force_rebuild
        )
        
        # 3. Khởi tạo Bộ gộp lai ghép RRF
        self.hybrid_retriever = HybridRetriever(k_rrf=60)
        
        # 4. Khởi tạo Bộ mở rộng câu hỏi tìm kiếm (Query Expander)
        self.query_expander = QueryExpander()
        
        # 5. Thiết lập Reranker ở chế độ Lazy-loading để tiết kiệm tài nguyên
        self.reranker = None

    def retrieve(self, query: str, strategy: str = "hybrid", top_k: int = 5) -> list[dict]:
        """
        Thực hiện chuỗi truy hồi thông tin theo chiến lược được lựa chọn:
        - "bm25": BM25 Keyword Search.
        - "dense": Vector Similarity Search.
        - "hybrid": BM25 + Dense kết hợp bằng RRF.
        - "hybrid_rerank": Hybrid (RRF) với đa câu hỏi mở rộng, được tái xếp hạng bằng Cross-Encoder.
        """
        if not query:
            return []

        strategy = strategy.lower().strip()

        # --- Chiến lược 1: BM25 ---
        if strategy == "bm25":
            return self.bm25_retriever.retrieve(query, top_k=top_k)

        # --- Chiến lược 2: Dense ---
        elif strategy == "dense":
            return self.dense_retriever.retrieve(query, top_k=top_k)

        # --- Chiến lược 3: Hybrid (Từ khóa + Ngữ nghĩa) ---
        elif strategy == "hybrid":
            # Lấy top-50 ứng viên từ mỗi bộ truy hồi để gộp
            bm25_candidates = self.bm25_retriever.retrieve(query, top_k=50)
            dense_candidates = self.dense_retriever.retrieve(query, top_k=50)
            return self.hybrid_retriever.fuse(bm25_candidates, dense_candidates, top_k=top_k)

        # --- Chiến lược 4: Hybrid + Cross-Encoder Reranker kết hợp Multi-query ---
        elif strategy == "hybrid_rerank":
            # 1. Sinh các truy vấn phụ mở rộng để bổ trợ cho câu hỏi chính
            sub_queries = self.query_expander.expand_query(query)
            all_queries = [query] + sub_queries
            
            # 2. Truy hồi ứng viên từ tất cả các truy vấn
            candidates_map = {}
            for idx, q in enumerate(all_queries):
                # Lấy 20 ứng viên cho câu hỏi chính và 15 ứng viên cho câu hỏi phụ
                k = 20 if idx == 0 else 15
                hybrid_candidates = self.retrieve(q, strategy="hybrid", top_k=k)
                
                for c in hybrid_candidates:
                    cid = c["chunk_id"]
                    # Nếu chunk trùng lặp, giữ lại chunk có score RRF cao nhất
                    if cid not in candidates_map or c["score"] > candidates_map[cid]["score"]:
                        candidates_map[cid] = c
            
            merged_candidates = list(candidates_map.values())
            
            # Giới hạn số lượng đầu vào cho Reranker để bảo đảm latency
            merged_candidates = sorted(merged_candidates, key=lambda x: x["score"], reverse=True)[:30]
            
            # 3. Rerank lại toàn bộ danh sách gộp dựa trên câu hỏi gốc của người dùng
            if self.reranker is None:
                self.reranker = Reranker(self.reranker_model)
                
            return self.reranker.rerank(query, merged_candidates, top_k=top_k)

        else:
            raise ValueError(
                f"Chiến lược truy hồi '{strategy}' không được hỗ trợ. "
                "Vui lòng chọn trong các chiến lược: 'bm25', 'dense', 'hybrid', 'hybrid_rerank'."
            )
