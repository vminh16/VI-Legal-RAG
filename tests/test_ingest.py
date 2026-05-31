import pytest
from pathlib import Path
from src.config import RAW_LAW_PDF_PATH
from src.ingest.pdf_extractor import PDFExtractor

def test_pdf_extractor_initialization():
    """Kiểm tra khởi tạo PDFExtractor với đường dẫn chính xác."""
    assert RAW_LAW_PDF_PATH.exists(), f"Không tìm thấy file PDF tại: {RAW_LAW_PDF_PATH}"
    extractor = PDFExtractor(RAW_LAW_PDF_PATH)
    assert extractor.pdf_path == RAW_LAW_PDF_PATH

def test_extract_raw_text():
    """Kiểm tra trích xuất text thô từ PDF."""
    extractor = PDFExtractor(RAW_LAW_PDF_PATH)
    raw_text = extractor.extract_raw_text()
    
    assert isinstance(raw_text, str)
    assert len(raw_text) > 100000, "Văn bản trích xuất quá ngắn, có thể bị lỗi đọc file."
    # Kiểm tra chứa một số từ khóa cốt lõi của Bộ luật Lao động
    assert "Bộ luật Lao động" in raw_text
    assert "Chương I" in raw_text
    assert "Điều 1" in raw_text

def test_extract_document_metadata():
    """Kiểm tra trích xuất Metadata cấp tài liệu (Document-level Metadata)."""
    extractor = PDFExtractor(RAW_LAW_PDF_PATH)
    metadata = extractor.extract_document_metadata()
    
    assert isinstance(metadata, dict)
    assert "document_title" in metadata
    assert "law_number" in metadata
    assert metadata["document_title"] == "Bộ luật Lao động 2019"
    assert "45/2019/QH14" in metadata["law_number"]

def test_get_main_text():
    """Kiểm tra việc tách biệt và chỉ lấy phần văn bản chính (từ Chương I trở đi)."""
    extractor = PDFExtractor(RAW_LAW_PDF_PATH)
    main_text = extractor.get_main_text()
    
    assert isinstance(main_text, str)
    # Phần văn bản chính phải bắt đầu bằng Chương I hoặc chứa Chương I ở phần đầu cực kỳ sớm
    # Loại bỏ phần văn bản hành chính rác trước Chương I
    assert "LỆNH" not in main_text[:200]
    assert "CHỦ TỊCH NƯỚC" not in main_text[:200]
    assert "Chương I" in main_text[:1000]
