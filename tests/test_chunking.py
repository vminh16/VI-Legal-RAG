import pytest
from src.chunking.fixed_chunker import FixedChunker
from src.chunking.structure_chunker import StructureChunker

def test_fixed_chunker_initialization():
    """Kiểm tra khởi tạo FixedChunker với các tham số khác nhau."""
    chunker = FixedChunker(chunk_size=300, chunk_overlap=50)
    assert chunker.chunk_size == 300
    assert chunker.chunk_overlap == 50

def test_fixed_chunker_splitting():
    """Kiểm tra tính đúng đắn của giải thuật chia nhỏ kích thước cố định."""
    chunker = FixedChunker(chunk_size=100, chunk_overlap=20)
    text = "Đây là văn bản thử nghiệm để chạy bộ chia nhỏ kích thước cố định của dự án ViLaborRAG." * 5
    document_metadata = {
        "document_title": "Bộ luật Lao động 2019 Test",
        "law_number": "45/TEST",
        "source_url": "https://test.url"
    }
    
    chunks = chunker.split_to_chunks(text, document_metadata)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    
    # Kiểm tra cấu trúc của từng chunk
    for idx, chunk in enumerate(chunks):
        assert "chunk_id" in chunk
        assert chunk["chunk_id"] == f"bll2019_fixed_{idx}"
        assert chunk["document_title"] == "Bộ luật Lao động 2019 Test"
        assert chunk["law_number"] == "45/TEST"
        assert chunk["source_url"] == "https://test.url"
        assert chunk["chapter"] is None
        assert chunk["article"] is None
        assert isinstance(chunk["text"], str)
        assert len(chunk["text"]) <= chunker.chunk_size

def test_structure_chunker_splitting():
    """Kiểm tra tính đúng đắn của giải thuật chia nhỏ theo cấu trúc pháp luật."""
    # Đặt max_article_len = 150 để Điều 3 (khoảng 225 ký tự) bị chia nhỏ theo các Khoản, 
    # trong khi Điều 1 (khoảng 130 ký tự) vẫn được giữ nguyên làm 1 chunk.
    chunker = StructureChunker(max_article_len=150)
    
    # Giả lập văn bản luật có cấu trúc thô sạch
    law_text = """Chương I
NHỮNG QUY ĐỊNH CHUNG
Mục 1
QUY ĐỊNH SƠ BỘ
Điều 1. Phạm vi điều chỉnh
Bộ luật Lao động quy định tiêu chuẩn lao động; quyền, nghĩa vụ, trách nhiệm của người lao động.
Điều 2. Đối tượng áp dụng
Người lao động Việt Nam.
Người sử dụng lao động.
Điều 3. Giải thích từ ngữ
Trong Bộ luật này, các từ ngữ dưới đây được hiểu như sau:
1. Người lao động là người làm việc cho người sử dụng lao động theo thỏa thuận.
2. Người sử dụng lao động là doanh nghiệp, cơ quan, tổ chức.
"""
    
    document_metadata = {
        "document_title": "Bộ luật Lao động 2019 Test",
        "law_number": "45/TEST",
        "source_url": "https://test.url"
    }
    
    chunks = chunker.split_to_chunks(law_text, document_metadata)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    
    # Kiểm tra Điều 1 (Ngắn - giữ nguyên làm 1 chunk)
    dieu1_chunks = [c for c in chunks if c["article"] == "Điều 1"]
    assert len(dieu1_chunks) == 1
    assert dieu1_chunks[0]["chunk_id"] == "bll2019_dieu_1"
    assert dieu1_chunks[0]["chapter"] == "Chương I"
    assert dieu1_chunks[0]["section"] == "Mục 1"
    assert dieu1_chunks[0]["title"] == "Phạm vi điều chỉnh"
    assert "tiêu chuẩn lao động" in dieu1_chunks[0]["text"]
    
    # Kiểm tra Điều 3 (Dài và chứa Khoản - chia nhỏ theo Khoản)
    dieu3_chunks = [c for c in chunks if c["article"] == "Điều 3"]
    assert len(dieu3_chunks) == 2  # Khoản 1 và Khoản 2
    
    assert dieu3_chunks[0]["chunk_id"] == "bll2019_dieu_3_khoan_1"
    assert dieu3_chunks[0]["clause"] == "1"
    assert "Người lao động là người làm việc" in dieu3_chunks[0]["text"]
    assert "Điều 3. Giải thích từ ngữ" in dieu3_chunks[0]["text"]  # Chứa tiêu đề cha để giữ ngữ cảnh
    
    assert dieu3_chunks[1]["chunk_id"] == "bll2019_dieu_3_khoan_2"
    assert dieu3_chunks[1]["clause"] == "2"
    assert "Người sử dụng lao động là doanh nghiệp" in dieu3_chunks[1]["text"]
