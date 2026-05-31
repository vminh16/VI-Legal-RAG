from abc import ABC, abstractmethod

class BaseChunker(ABC):
    """
    Interface cơ sở (Abstract Class) cho các bộ phân đoạn (Chunkers).
    Tất cả các bộ chunker của ViLaborRAG phải kế thừa lớp này và định nghĩa hàm split_to_chunks.
    """
    @abstractmethod
    def split_to_chunks(self, text: str, document_metadata: dict = None) -> list[dict]:
        """
        Phân tách đoạn văn bản thô thành danh sách các chunk.
        
        Args:
            text: Văn bản thô cần chia nhỏ.
            document_metadata: Metadata cấp tài liệu (như document_title, law_number...) để tự động bổ sung cho các chunk.
            
        Returns:
            list[dict]: Danh sách các chunk có cấu trúc chuẩn theo đặc tả dữ liệu.
        """
        pass
