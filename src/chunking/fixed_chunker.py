from src.chunking.base_chunker import BaseChunker

class FixedChunker(BaseChunker):
    """
    Bộ phân đoạn kích thước cố định (Fixed-size Chunker) - làm Baseline cho RAG.
    Chia nhỏ văn bản dựa trên số lượng ký tự (characters) cố định có phần gối đầu (overlap).
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_to_chunks(self, text: str, document_metadata: dict = None) -> list[dict]:
        """
        Thực hiện chia nhỏ văn bản thô theo kích thước cố định.
        """
        if not text:
            return []

        doc_meta = document_metadata or {}
        document_title = doc_meta.get("document_title", "Bộ luật Lao động 2019")
        law_number = doc_meta.get("law_number", "45/2019/QH14")
        source_url = doc_meta.get("source_url", "https://vanban.chinhphu.vn/?docid=198540&pageid=27160")

        chunks = []
        text_len = len(text)
        step = self.chunk_size - self.chunk_overlap
        
        # Đảm bảo step luôn dương để tránh lặp vô hạn
        if step <= 0:
            step = self.chunk_size

        idx = 0
        start = 0
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_id = f"bll2019_fixed_{idx}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_title": document_title,
                    "law_number": law_number,
                    "chapter": None,
                    "section": None,
                    "article": None,
                    "clause": None,
                    "point": None,
                    "title": None,
                    "text": chunk_text,
                    "source_url": source_url
                })
                idx += 1
                
            start += step

        return chunks
