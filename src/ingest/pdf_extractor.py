import os
from pathlib import Path
from pypdf import PdfReader

class PDFExtractor:
    """
    Class xử lý trích xuất văn bản từ tệp PDF Bộ luật Lao động.
    Hỗ trợ chế độ fallback tự động đọc từ file .txt cùng tên nếu PDF là ảnh quét (scanned image PDF).
    """
    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        self.txt_fallback_path = self.pdf_path.with_suffix('.txt')
        self._raw_text = None

    def extract_raw_text(self) -> str:
        """
        Trích xuất văn bản thô từ tệp PDF. 
        Nếu PDF là định dạng scan thô (trích xuất ra rỗng), tự động đọc từ file text pre-extracted (.txt).
        """
        if self._raw_text is not None:
            return self._raw_text

        extracted_text = ""
        try:
            # Thử trích xuất bằng pypdf
            reader = PdfReader(self.pdf_path)
            pages_text = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            extracted_text = "\n".join(pages_text)
        except Exception:
            pass

        # Kiểm tra nếu text trích xuất từ PDF quá ngắn hoặc rỗng -> Sử dụng cơ chế fallback đọc từ file .txt
        if len(extracted_text.strip()) < 1000:
            if self.txt_fallback_path.exists():
                with open(self.txt_fallback_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
            else:
                raise FileNotFoundError(
                    f"Không thể trích xuất văn bản từ PDF (dạng scan) và không tìm thấy file text fallback tại: {self.txt_fallback_path}"
                )

        self._raw_text = extracted_text
        return self._raw_text

    def extract_document_metadata(self) -> dict:
        """
        Trích xuất thông tin cốt lõi làm Metadata cấp tài liệu (Document-level Metadata).
        Ví dụ: tiêu đề tài liệu, số hiệu văn bản, ngày có hiệu lực.
        """
        # Đảm bảo đã có raw text
        text = self.extract_raw_text()

        # Giá trị mặc định theo đặc tả Bộ luật Lao động 2019
        metadata = {
            "document_title": "Bộ luật Lao động 2019",
            "law_number": "45/2019/QH14",
            "effective_date": "2021-01-01",
            "source_url": "https://vanban.chinhphu.vn/?docid=198540&pageid=27160"
        }

        # Tìm kiếm động số hiệu luật trong 2000 ký tự đầu tiên nếu có định dạng khác
        import re
        match_law_number = re.search(r'(Lu\u1eadt s\u1ed1|S\u1ed1):\s*([0-9\w/-]+QH[0-9]+)', text[:2000], re.IGNORECASE)
        if match_law_number:
            metadata["law_number"] = match_law_number.group(2).strip()

        return metadata

    def get_main_text(self) -> str:
        """
        Lấy phần văn bản chính, loại bỏ phần mở đầu hành chính trước "Chương I"
        và toàn bộ phần sitemap, liên kết, CSS/JS rác ở chân trang (sau Điều 220).
        """
        text = self.extract_raw_text()

        # 1. Tìm vị trí bắt đầu của "Chương I" (ở đầu một dòng mới)
        import re
        match_start = re.search(r'(^|\n)(Ch\u01b0\u01a1ng I\b)', text, re.IGNORECASE)
        start_idx = match_start.start(2) if match_start else 0

        # 2. Tìm vị trí bắt đầu của phần sitemap/liên kết rác ở chân trang sau Điều 220
        footer_keywords = [
            r'V\u0103n b\u1ea3n li\xean quan c\xf9ng n\u1ed9i dung',
            r'V\u0103n b\u1ea3n li\xean quan',
            r'V\u0103n b\u1ea3n h\u01b0\u1edbng d\u1eabn',
            r'\u0110\xc2Y L\xc0 N\u1ed8I DUNG C\xd3 THU PH\xcd'
        ]
        
        end_idx = len(text)
        for keyword in footer_keywords:
            match_end = re.search(keyword, text[start_idx:], re.IGNORECASE)
            if match_end:
                candidate_end = start_idx + match_end.start()
                if candidate_end < end_idx:
                    end_idx = candidate_end

        return text[start_idx:end_idx].strip()
