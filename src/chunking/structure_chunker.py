import re
from src.chunking.base_chunker import BaseChunker

class StructureChunker(BaseChunker):
    """
    Bộ phân đoạn theo cấu trúc pháp lý (Structure-aware Chunker).
    Phân tích cú pháp văn bản luật thành Chương -> Mục -> Điều -> Khoản -> Điểm.
    Chia nhỏ văn bản giữ nguyên ranh giới Điều/Khoản để tránh mất ngữ cảnh.
    """
    def __init__(self, max_article_len: int = 1200):
        self.max_article_len = max_article_len

    def split_to_chunks(self, text: str, document_metadata: dict = None) -> list[dict]:
        """
        Phân tích cú pháp và chia nhỏ văn bản theo cấu trúc Chương/Điều/Khoản.
        """
        if not text:
            return []

        doc_meta = document_metadata or {}
        document_title = doc_meta.get("document_title", "Bộ luật Lao động 2019")
        law_number = doc_meta.get("law_number", "45/2019/QH14")
        source_url = doc_meta.get("source_url", "https://vanban.chinhphu.vn/?docid=198540&pageid=27160")

        # Bước 1: Parse văn bản thành cấu trúc các Điều
        articles = self._parse_articles(text)
        
        # Bước 2: Tạo chunk từ các Điều dựa trên luật độ dài
        chunks = []
        for art in articles:
            chunks.extend(self._chunk_article(art, document_title, law_number, source_url))
            
        return chunks

    def _parse_articles(self, text: str) -> list[dict]:
        """
        Phân tích tài liệu thành danh sách các đối tượng Điều có đầy đủ thông tin phân tầng.
        """
        lines = text.split('\n')
        articles = []
        
        current_chapter = None
        current_chapter_title = None
        current_section = None
        current_section_title = None
        
        current_article = None
        
        # Các mẫu Regex biên dịch sẵn
        chapter_pattern = re.compile(r'^Ch\u01b0\u01a1ng\s+([IVXLCDM\d]+)', re.IGNORECASE)
        section_pattern = re.compile(r'^M\u1ee5c\s+(\d+)', re.IGNORECASE)
        article_pattern = re.compile(r'^\u0110i\u1ec1u\s+(\d+)\.\s*(.*)', re.IGNORECASE)
        
        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()
            if not line:
                idx += 1
                continue
                
            # 1. Phát hiện ranh giới Chương
            chapter_match = chapter_pattern.match(line)
            if chapter_match:
                current_chapter = line
                current_section = None
                current_section_title = None
                # Đọc dòng tiếp theo làm tiêu đề Chương
                idx += 1
                if idx < len(lines) and lines[idx].strip():
                    current_chapter_title = lines[idx].strip()
                idx += 1
                continue
                
            # 2. Phát hiện ranh giới Mục
            section_match = section_pattern.match(line)
            if section_match:
                current_section = line
                # Đọc dòng tiếp theo làm tiêu đề Mục
                idx += 1
                if idx < len(lines) and lines[idx].strip():
                    current_section_title = lines[idx].strip()
                idx += 1
                continue
                
            # 3. Phát hiện ranh giới Điều
            article_match = article_pattern.match(line)
            if article_match:
                art_num = article_match.group(1)
                art_title = article_match.group(2).strip()
                
                # Lưu Điều hiện tại trước khi tạo Điều mới
                if current_article:
                    articles.append(current_article)
                    
                current_article = {
                    "chapter": current_chapter,
                    "chapter_title": current_chapter_title,
                    "section": current_section,
                    "section_title": current_section_title,
                    "article_number": art_num,
                    "article": f"\u0110i\u1ec1u {art_num}",
                    "title": art_title,
                    "lines": [line]
                }
                idx += 1
                continue
                
            # 4. Gom văn bản cho Điều hiện tại
            if current_article:
                current_article["lines"].append(line)
                
            idx += 1
            
        # Thêm Điều cuối cùng
        if current_article:
            articles.append(current_article)
            
        return articles

    def _chunk_article(self, art: dict, doc_title: str, law_num: str, source_url: str) -> list[dict]:
        """
        Chia nhỏ một Điều thành một hoặc nhiều chunk dựa trên độ dài.
        """
        # Hợp nhất văn bản thô của Điều
        raw_art_text = "\n".join(art["lines"]).strip()
        
        chapter_str = art["chapter"]
        section_str = art["section"]
        article_name = art["article"]
        article_title = art["title"]
        
        # Nếu Điều ngắn (<= max_article_len ký tự), giữ nguyên Điều đó làm 1 chunk duy nhất
        if len(raw_art_text) <= self.max_article_len:
            cleaned_art_name = article_name.lower().replace(' ', '_').replace('điều', 'dieu')
            chunk_id = f"bll2019_{cleaned_art_name}"
            return [{
                "chunk_id": chunk_id,
                "document_title": doc_title,
                "law_number": law_num,
                "chapter": chapter_str,
                "section": section_str,
                "article": article_name,
                "clause": None,
                "point": None,
                "title": article_title,
                "text": raw_art_text,
                "source_url": source_url
            }]
            
        # Nếu Điều quá dài, tiến hành chia nhỏ theo các Khoản (Clauses)
        chunks = []
        clause_pattern = re.compile(r'^(\d+)\.\s+(.*)', re.DOTALL)
        
        intro_lines = []
        clauses = []
        
        # Phân loại dòng tiêu đề, dòng mở đầu và dòng chứa Khoản
        # Bỏ qua dòng tiêu đề Điều đầu tiên
        for line in art["lines"][1:]:
            stripped = line.strip()
            if not stripped:
                continue
                
            clause_match = clause_pattern.match(stripped)
            if clause_match:
                clauses.append({
                    "number": clause_match.group(1),
                    "text": stripped
                })
            else:
                if not clauses:
                    # Các dòng văn bản trước Khoản 1 là phần mở đầu (Introductory text)
                    intro_lines.append(stripped)
                else:
                    # Các dòng sau một Khoản nhưng không bắt đầu bằng số -> Gộp vào Khoản hiện tại
                    clauses[-1]["text"] += "\n" + stripped
                    
        intro_text = "\n".join(intro_lines).strip()
        
        # Nếu không phân tách được Khoản nào, fallback về chia nhỏ theo ký tự cố định nhưng giữ metadata
        if not clauses:
            # Fallback chia theo ký tự cố định cho Điều này
            text_to_split = raw_art_text
            art_len = len(text_to_split)
            start = 0
            idx = 0
            while start < art_len:
                end = min(start + 800, art_len)
                chunk_text = text_to_split[start:end].strip()
                if chunk_text:
                    cleaned_art_name = article_name.lower().replace(' ', '_').replace('điều', 'dieu')
                    chunks.append({
                        "chunk_id": f"bll2019_{cleaned_art_name}_part_{idx}",
                        "document_title": doc_title,
                        "law_number": law_num,
                        "chapter": chapter_str,
                        "section": section_str,
                        "article": article_name,
                        "clause": None,
                        "point": None,
                        "title": article_title,
                        "text": chunk_text,
                        "source_url": source_url
                    })
                    idx += 1
                start += 600
            return chunks

        # Đóng gói từng Khoản làm một chunk
        for c in clauses:
            clause_num = c["number"]
            clause_text = c["text"]
            
            # Xây dựng text của chunk có ngữ cảnh đầy đủ:
            # [Điều X. Tiêu đề] + [Phần dẫn (nếu có)] + [Khoản Y. Nội dung Khoản]
            parts = [f"{article_name}. {article_title}"]
            if intro_text:
                parts.append(intro_text)
            parts.append(clause_text)
            
            chunk_text = "\n".join(parts)
            cleaned_art_name = article_name.lower().replace(' ', '_').replace('điều', 'dieu')
            chunk_id = f"bll2019_{cleaned_art_name}_khoan_{clause_num}"
            
            chunks.append({
                "chunk_id": chunk_id,
                "document_title": doc_title,
                "law_number": law_num,
                "chapter": chapter_str,
                "section": section_str,
                "article": article_name,
                "clause": clause_num,
                "point": None,
                "title": article_title,
                "text": chunk_text,
                "source_url": source_url
            })
            
        return chunks
