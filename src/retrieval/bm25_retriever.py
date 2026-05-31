import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from src.retrieval.query_preprocessor import QueryPreprocessor

class BM25Retriever:
    """
    Bộ truy hồi từ khóa sử dụng thuật toán BM25 (BM25Okapi).
    Hỗ trợ nạp corpus, token hóa thông minh và tìm kiếm từ khóa kèm tiền xử lý.
    """
    def __init__(self, chunks: list[dict] = None, corpus_path: str | Path = None):
        self.preprocessor = QueryPreprocessor()
        
        # 1. Nạp dữ liệu corpus
        if chunks is not None:
            self.chunks = chunks
        else:
            path = Path(corpus_path or "data/processed/corpus_structured.jsonl")
            if not path.exists():
                raise FileNotFoundError(f"Không tìm thấy file corpus để lập chỉ mục BM25 tại: {path}")
            
            self.chunks = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.chunks.append(json.loads(line))
                        
        # 2. Token hóa corpus và khởi tạo chỉ mục BM25Okapi
        tokenized_corpus = [self._tokenize(chunk["text"]) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> list[str]:
        """
        Token hóa văn bản thô một cách tối giản và sạch sẽ:
        Lowercase, loại bỏ các ký tự đặc biệt/dấu câu, chuẩn hóa khoảng trắng và tách từ.
        """
        if not text:
            return []
        
        # Chuyển về lowercase
        lowercased = text.lower()
        
        # Loại bỏ các dấu câu (giữ lại chữ cái, số và khoảng trắng)
        cleaned = re.sub(r'[^\w\s]', ' ', lowercased)
        
        # Tách từ bằng khoảng trắng
        tokens = cleaned.split()
        return tokens

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Thực hiện truy hồi từ khóa bằng BM25:
        1. Tiền xử lý câu hỏi (Unicode NFC, lowercase, sửa viết tắt).
        2. Token hóa câu hỏi.
        3. Tính điểm BM25 và trả về top-K chunk phù hợp nhất.
        """
        if not query or not self.chunks:
            return []

        # Bước 1: Tiền xử lý câu hỏi
        cleaned_query = self.preprocessor.preprocess(query)
        
        # Bước 2: Token hóa câu hỏi
        tokenized_query = self._tokenize(cleaned_query)
        
        # Bước 3: Tính điểm số BM25Okapi
        scores = self.bm25.get_scores(tokenized_query)
        
        # Bước 4: Sắp xếp các chunk theo điểm số giảm dần
        scored_chunks = []
        for idx, chunk in enumerate(self.chunks):
            score = float(scores[idx])
            # Bổ sung điểm score vào bản sao của chunk để tránh làm biến dạng corpus gốc
            chunk_copy = dict(chunk)
            chunk_copy["score"] = score
            scored_chunks.append(chunk_copy)
            
        # Sắp xếp giảm dần theo score
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        # Trả về top-K chunk
        return scored_chunks[:top_k]
