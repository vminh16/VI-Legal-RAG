import re
import unicodedata

class QueryPreprocessor:
    """
    Module tiền xử lý câu hỏi đầu vào (Query Preprocessor) của hệ thống RAG.
    Nhiệm vụ: Làm sạch khoảng trắng, chuẩn hóa Unicode NFC, lowercase 
    và giải quyết các từ viết tắt chuyên môn tiếng Việt một cách an toàn.
    """
    def __init__(self):
        # Định nghĩa từ điển viết tắt chuyên môn
        self.abbreviations = {
            "cty": "công ty",
            "hđlđ": "hợp đồng lao động",
            "hdlđ": "hợp đồng lao động",
            "hdld": "hợp đồng lao động",
            "nlđ": "người lao động",
            "nld": "người lao động",
            "nsdlđ": "người sử dụng lao động",
            "nsdld": "người sử dụng lao động",
            "bllđ": "bộ luật lao động",
            "blld": "bộ luật lao động",
        }
        
        # Biên dịch trước các Regex thay thế viết tắt để tối ưu hiệu năng
        # Sử dụng Lookbehind (?<!\w) và Lookahead (?!\w) đảm bảo ranh giới từ chuẩn Unicode tiếng Việt
        self.abbr_patterns = []
        for abbr, full_form in self.abbreviations.items():
            pattern = re.compile(rf"(?<!\w){re.escape(abbr)}(?!\w)", re.IGNORECASE)
            self.abbr_patterns.append((pattern, full_form))

    def preprocess(self, query: str) -> str:
        """
        Thực hiện chuỗi tiền xử lý câu hỏi:
        1. Chuẩn hóa Unicode sang dạng NFC.
        2. Chuyển về lowercase.
        3. Thay thế các từ viết tắt chuyên môn.
        4. Chuẩn hóa khoảng trắng thừa.
        """
        if not query:
            return ""

        # Bước 1: Chuẩn hóa Unicode NFC để đồng nhất bảng mã
        cleaned = unicodedata.normalize("NFC", query)

        # Bước 2: Chuyển về lowercase
        cleaned = cleaned.lower()

        # Bước 3: Thay thế các từ viết tắt chuyên môn một cách an toàn
        for pattern, full_form in self.abbr_patterns:
            cleaned = pattern.sub(full_form, cleaned)

        # Bước 4: Chuẩn hóa khoảng trắng thừa
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned
