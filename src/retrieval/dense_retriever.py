import json
import os
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.retrieval.query_preprocessor import QueryPreprocessor

class DenseRetriever:
    """
    Bộ truy hồi ngữ nghĩa sử dụng Dense Vectors và thư viện FAISS.
    Hỗ trợ sinh vector từ SentenceTransformer, chuẩn hóa L2 tìm kiếm Cosine Similarity,
    và tích hợp lớp Cache đĩa cực kỳ mạnh mẽ để tăng tốc khởi động.
    """
    def __init__(
        self,
        chunks: list[dict] = None,
        corpus_path: str | Path = None,
        model_name: str = "BAAI/bge-m3",
        index_dir: str | Path = "data/processed",
        force_rebuild: bool = False
    ):
        self.preprocessor = QueryPreprocessor()
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        
        # 1. Nạp dữ liệu corpus
        if chunks is not None:
            self.chunks = chunks
        else:
            path = Path(corpus_path or "data/processed/corpus_structured.jsonl")
            if not path.exists():
                raise FileNotFoundError(f"Không tìm thấy file corpus để lập chỉ mục Dense tại: {path}")
            
            self.chunks = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.chunks.append(json.loads(line))
        
        # 2. Khởi tạo mô hình Embedding
        # SentenceTransformer sẽ tự động tải hoặc nạp từ cache cục bộ của HuggingFace
        self.model = SentenceTransformer(self.model_name)
        
        # 3. Quản lý Cache Index FAISS
        self.index_file = self.index_dir / "faiss_index.bin"
        self.mapping_file = self.index_dir / "faiss_mapping.json"
        
        # Load hoặc dựng chỉ mục FAISS
        if not force_rebuild and self.index_file.exists() and self.mapping_file.exists():
            self._load_cached_index()
        else:
            self._build_and_cache_index()

    def _load_cached_index(self):
        """
        Nạp nhanh vector index và mapping từ đĩa cục bộ.
        Kiểm tra tính toàn vẹn (Integrity Check) đối chiếu với corpus hiện tại.
        """
        try:
            self.index = faiss.read_index(str(self.index_file))
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                self.mapping = json.load(f)
            # Chuyển đổi key của mapping sang kiểu int để đảm bảo khớp chỉ mục FAISS
            self.mapping = {int(k): v for k, v in self.mapping.items()}
            
            # Kiểm tra tính toàn vẹn: Số lượng vector trong FAISS và số key mapping phải khớp chính xác với corpus
            if self.index.ntotal != len(self.chunks) or len(self.mapping) != len(self.chunks):
                raise ValueError("Số lượng vector cache lệch so với số lượng chunks hiện tại của corpus.")
        except Exception as e:
            # Fallback rebuild lại index nếu load cache thất bại hoặc lệch biên
            self._build_and_cache_index()

    def _build_and_cache_index(self):
        """
        Mã hóa toàn bộ corpus thành các dense vectors, xây dựng FAISS Index và lưu xuống đĩa.
        """
        if not self.chunks:
            self.index = None
            self.mapping = {}
            return

        # Trích xuất văn bản thô từ chunks
        texts = [chunk["text"] for chunk in self.chunks]
        
        # Nếu sử dụng dòng mô hình E5, bắt buộc bổ sung tiền tố passage: theo quy chuẩn của mô hình
        if "e5" in self.model_name.lower():
            texts = [f"passage: {t}" for t in texts]

        # Sinh embeddings (trả về numpy array float32)
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        ).astype('float32')

        # Chuẩn hóa L2 cho các vector để Inner Product FAISS tương đương Cosine Similarity
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        
        # Sử dụng chỉ mục Flat Inner Product để tìm kiếm độ tương đồng cosine
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        # Tạo ánh xạ index vector sang chunk_id thực tế
        self.mapping = {idx: chunk["chunk_id"] for idx, chunk in enumerate(self.chunks)}

        # Lưu trữ cache xuống đĩa
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_file))
        with open(self.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.mapping, f, ensure_ascii=False)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Truy hồi ngữ nghĩa (Dense Retrieval):
        1. Tiền xử lý câu hỏi.
        2. Mã hóa câu hỏi thành vector (thêm prefix query: nếu sử dụng mô hình E5).
        3. Tìm kiếm k-NN trên FAISS index và trả ra top-K kết quả.
        """
        if not query or self.index is None:
            return []

        # Bước 1: Tiền xử lý câu hỏi người dùng
        cleaned_query = self.preprocessor.preprocess(query)

        # Bổ sung tiền tố nếu dùng dòng mô hình E5
        if "e5" in self.model_name.lower():
            cleaned_query = f"query: {cleaned_query}"

        # Bước 2: Mã hóa câu hỏi thành vector
        query_vector = self.model.encode(
            [cleaned_query],
            show_progress_bar=False,
            convert_to_numpy=True
        ).astype('float32')

        # Chuẩn hóa L2 cho query vector
        faiss.normalize_L2(query_vector)

        # Bước 3: Thực hiện tìm kiếm K lân cận gần nhất
        distances, indices = self.index.search(query_vector, top_k)

        # Bước 4: Đóng gói kết quả đầu ra
        results = []
        for rank_idx, corpus_idx in enumerate(indices[0]):
            corpus_idx = int(corpus_idx)
            if corpus_idx == -1:  # FAISS trả về -1 nếu không tìm đủ k lân cận
                continue
            
            score = float(distances[0][rank_idx])
            chunk_id = self.mapping.get(corpus_idx)
            
            # Khôi phục đầy đủ thông tin của chunk từ danh sách chunks
            chunk = next((c for c in self.chunks if c["chunk_id"] == chunk_id), None)
            if chunk:
                chunk_copy = dict(chunk)
                chunk_copy["score"] = score
                results.append(chunk_copy)

        return results
