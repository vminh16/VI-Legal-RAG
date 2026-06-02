import re
import json
import logging
import unicodedata
from pathlib import Path
from src.generation.generator import RAGResponse

logger = logging.getLogger(__name__)

def normalize_term(text: str) -> str:
    """
    Chuẩn hóa thuật ngữ tiếng Việt để đối chiếu không phân biệt dấu, chữ hoa thường hay khoảng trắng.
    Ví dụ: 'Điều 24' -> 'dieu_24', 'Khoản 1' -> 'khoan_1'
    """
    if not text:
        return ""
    
    # Đưa về chữ thường và NFC chuẩn hóa
    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)
    
    # Thay thế khoảng trắng bằng dấu gạch dưới
    text = re.sub(r'\s+', '_', text)
    text = text.replace("đ", "d")
    
    # Bản đồ xóa dấu tiếng Việt để đối chiếu cực kỳ bao dung
    vietnamese_map = {
        'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
        'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
        'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
        'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
        'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
        'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
        'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
    }
    
    for k, v in vietnamese_map.items():
        text = text.replace(k, v)
        
    # Loại bỏ ký tự đặc biệt chỉ giữ chữ cái, số và gạch dưới
    text = re.sub(r'[^\w_]', '', text)
    return text

def make_key(article: str, clause: str | None) -> tuple[str, str]:
    """Tạo khóa chuẩn hóa phục vụ tra cứu chính xác Điều/Khoản."""
    art_key = normalize_term(article)
    cl_key = normalize_term(clause) if clause else ""
    return (art_key, cl_key)


class CitationChecker:
    """
    Module kiểm tra trích dẫn tĩnh so khớp metadata đối với Corpus và Context.
    """
    def __init__(self, chunks: list[dict] = None, corpus_path: str | Path = None):
        self.chunks = []
        
        # 1. Nạp cơ sở dữ liệu Corpus
        if chunks is not None:
            self.chunks = chunks
        else:
            path = Path(corpus_path or "data/processed/corpus_structured.jsonl")
            if not path.exists():
                logger.warning(f"Không tìm thấy file corpus tại {path}. Hãy chạy ingestion pipeline trước.")
            else:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                self.chunks.append(json.loads(line))
                except Exception as e:
                    logger.error(f"Lỗi khi đọc file corpus: {e}")
                    
        # 2. Xây dựng index tra cứu nhanh theo (article, clause)
        self.corpus_index = {}
        for chunk in self.chunks:
            article = chunk.get("article")
            clause = chunk.get("clause")
            if article:
                key = make_key(article, clause)
                # Lưu chunk gốc vào index
                self.corpus_index[key] = chunk

    def check_citations(self, response: RAGResponse, retrieved_chunks: list[dict]) -> dict:
        """
        Thực hiện kiểm tra chéo trích dẫn tĩnh:
        1. So khớp nhãn trích dẫn trong văn bản [X] với danh sách citations.
        2. Xác minh sự tồn tại của Điều/Khoản trích dẫn trong Corpus CSDL.
        3. Xác minh Điều/Khoản có nằm trong Context được truy hồi hay không.
        4. Xác minh chuỗi bằng chứng (evidence) có khớp thực tế trong văn bản gốc.
        """
        errors = []
        details = []
        
        answer = response.answer or ""
        citations = response.citations or []
        
        # 1. Quét tìm tất cả các nhãn [X] xuất hiện trong câu trả lời
        bracket_ids = [int(x) for x in re.findall(r'\[(\d+)\]', answer)]
        bracket_set = set(bracket_ids)
        
        citation_ids = [c.citation_id for c in citations]
        citation_set = set(citation_ids)
        
        # Kiểm tra lệch nhãn
        if bracket_set != citation_set:
            errors.append("malformed_citation")
            
        # Tạo tập các khóa chuẩn hóa từ ngữ cảnh truy hồi để đối chiếu nhanh
        retrieved_keys = set()
        for r_chunk in (retrieved_chunks or []):
            art = r_chunk.get("article")
            cl = r_chunk.get("clause")
            if art:
                retrieved_keys.add(make_key(art, cl))
                
        # 2. Duyệt qua từng trích dẫn trong danh sách để kiểm định chi tiết
        for cit in citations:
            cit_id = cit.citation_id
            article = cit.article
            clause = cit.clause
            evidence = cit.evidence or ""
            
            cit_errors = []
            cit_key = make_key(article, clause)
            
            # Kiểm tra tồn tại trong CSDL Corpus
            corpus_chunk = self.corpus_index.get(cit_key)
            if not corpus_chunk:
                cit_errors.append("citation_not_found_in_corpus")
            else:
                # Kiểm tra xem có nằm trong Context được truy hồi không
                if cit_key not in retrieved_keys:
                    cit_errors.append("citation_not_in_retrieved_context")
                    
                # Kiểm tra tính chuẩn xác của chuỗi bằng chứng (Evidence check)
                evidence_clean = " ".join(evidence.lower().split())
                text_clean = " ".join(corpus_chunk.get("text", "").lower().split())
                if evidence_clean not in text_clean:
                    cit_errors.append("fabricated_evidence")
                    
            if cit_errors:
                errors.extend(cit_errors)
                
            details.append({
                "citation_id": cit_id,
                "article": article,
                "clause": clause,
                "is_valid": len(cit_errors) == 0,
                "errors": cit_errors
            })
            
        # Loại bỏ các lỗi trùng lặp để danh sách lỗi gọn gàng
        unique_errors = list(set(errors))
        
        return {
            "is_valid": len(unique_errors) == 0,
            "errors": unique_errors,
            "details": details
        }
