import json
import logging
import re
from pathlib import Path
from src.config import SETTINGS, PROJECT_ROOT

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """
    Thành phần chuyên trách tính toán các chỉ số đo lường hiệu năng của RAG Pipeline:
    1. Retrieval Metrics: Recall@k, MRR@k, Hit Rate@k
    2. Citation Metrics: Fabricated Citation Rate, Unsupported Citation Rate
    3. Refusal Metrics: Refusal Accuracy
    """
    def __init__(self, corpus_path: str | Path = None):
        # 1. Nạp danh sách các Điều luật hợp lệ từ Corpus để làm Ground Truth kiểm tra bịa luật
        self.corpus_articles = set()
        
        # Xác định đường dẫn corpus
        if corpus_path is None:
            rel_path = SETTINGS.get("paths", {}).get("corpus_jsonl", "data/processed/corpus_structured.jsonl")
            corpus_path = Path(PROJECT_ROOT) / rel_path
        else:
            corpus_path = Path(corpus_path)

        if corpus_path.exists():
            try:
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            chunk = json.loads(line)
                            art = chunk.get("article")
                            if art:
                                self.corpus_articles.add(self.normalize_article(art))
                logger.info(f"Đã nạp {len(self.corpus_articles)} Điều luật từ corpus để phục vụ đánh giá.")
            except Exception as e:
                logger.error(f"Lỗi khi đọc file corpus tại {corpus_path}: {e}")
        else:
            logger.warning(f"Không tìm thấy tệp corpus tại {corpus_path}. Chỉ số Fabricated Citation Rate có thể không chính xác.")

    def normalize_article(self, article_str: str) -> str:
        """Chuẩn hóa chuỗi tên điều luật (ví dụ: 'Điều 24.' hoặc 'dieu 24' -> 'Điều 24')"""
        if not article_str:
            return ""
        clean = article_str.strip().lower()
        # Chuyển đổi không dấu cơ bản để so khớp
        clean = clean.replace("điều", "dieu").replace("dieu ", "dieu")
        match = re.search(r"dieu\s*(\d+)", clean)
        if match:
            return f"Điều {match.group(1)}"
        return article_str.strip()

    def normalize_clause(self, clause_str: str) -> str:
        """Chuẩn hóa chuỗi khoản (ví dụ: 'Khoản 1' hoặc '1' -> '1')"""
        if not clause_str:
            return ""
        clean = str(clause_str).strip().lower()
        match = re.search(r"\d+", clean)
        if match:
            return match.group(0)
        return clean

    def evaluate_retrieval(self, retrieved_chunks: list[dict], gold_sources: list[dict], k: int = 5) -> dict:
        """
        Tính toán các chỉ số truy hồi trên top-k kết quả:
        - recall_at_k: Tỷ lệ số gold sources tìm thấy trong top-k.
        - mrr_at_k: Mean Reciprocal Rank của gold source đầu tiên xuất hiện trong top-k.
        - hit_rate_at_k: 1.0 nếu tìm thấy ít nhất 1 gold source trong top-k, ngược lại 0.0.
        """
        if not gold_sources:
            return {
                "recall_at_k": 0.0,
                "mrr_at_k": 0.0,
                "hit_rate_at_k": 0.0
            }

        top_k_chunks = retrieved_chunks[:k]
        
        # Chuẩn hóa tập gold sources để dễ so sánh
        normalized_golds = []
        for gold in gold_sources:
            normalized_golds.append({
                "article": self.normalize_article(gold.get("article", "")),
                "clause": self.normalize_clause(gold.get("clause", ""))
            })

        # Chuẩn hóa tập chunks thu về từ retrieval
        normalized_chunks = []
        for chunk in top_k_chunks:
            normalized_chunks.append({
                "article": self.normalize_article(chunk.get("article", "")),
                "clause": self.normalize_clause(chunk.get("clause", ""))
            })

        found_golds_count = 0
        first_gold_rank = None

        # Kiểm tra từng gold source xem có xuất hiện trong top-k chunks không
        for gold in normalized_golds:
            found_this_gold = False
            for idx, chunk in enumerate(normalized_chunks):
                # Khớp mức độ Điều luật (Article level)
                if chunk["article"] == gold["article"]:
                    # Nếu gold source yêu cầu khoản cụ thể, kiểm tra xem khoản có khớp không
                    if gold["clause"]:
                        if chunk["clause"] == gold["clause"]:
                            found_this_gold = True
                            if first_gold_rank is None or (idx + 1) < first_gold_rank:
                                first_gold_rank = idx + 1
                            break
                    else:
                        # Nếu gold source chỉ cần đúng Điều luật
                        found_this_gold = True
                        if first_gold_rank is None or (idx + 1) < first_gold_rank:
                            first_gold_rank = idx + 1
                        break
            if found_this_gold:
                found_golds_count += 1

        recall = found_golds_count / len(gold_sources)
        mrr = 1.0 / first_gold_rank if first_gold_rank is not None else 0.0
        hit_rate = 1.0 if found_golds_count > 0 else 0.0

        return {
            "recall_at_k": recall,
            "mrr_at_k": mrr,
            "hit_rate_at_k": hit_rate
        }

    def evaluate_citations(self, citations: list[dict], retrieved_chunks: list[dict]) -> dict:
        """
        Tính toán các chỉ số trích dẫn:
        - fabricated_citation_rate: Tỷ lệ trích dẫn không tồn tại trong corpus thực tế.
        - unsupported_citation_rate: Tỷ lệ trích dẫn không nằm trong tập retrieved context.
        """
        if not citations:
            return {
                "fabricated_citation_rate": 0.0,
                "unsupported_citation_rate": 0.0,
                "citation_count": 0
            }

        fabricated_count = 0
        unsupported_count = 0

        # Tập hợp các Điều luật xuất hiện trong retrieved context (danh sách chunks thu về)
        retrieved_articles = {self.normalize_article(c.get("article", "")) for c in retrieved_chunks}

        for cit in citations:
            art = self.normalize_article(cit.get("article", ""))
            
            # 1. Kiểm tra bịa luật (không có trong corpus)
            if self.corpus_articles and art not in self.corpus_articles:
                fabricated_count += 1
            
            # 2. Kiểm tra lệch nguồn (không nằm trong context được cung cấp cho prompt)
            if art not in retrieved_articles:
                unsupported_count += 1

        total_citations = len(citations)
        return {
            "fabricated_citation_rate": fabricated_count / total_citations,
            "unsupported_citation_rate": unsupported_count / total_citations,
            "citation_count": total_citations
        }

    def evaluate_refusal(self, refused: bool, must_refuse: bool) -> dict:
        """
        Tính toán chỉ số từ chối đúng (Refusal Accuracy).
        - must_refuse=True -> Phải từ chối thành công (refused=True).
        - must_refuse=False -> Phải trả lời bình thường (refused=False).
        """
        correct = (refused == must_refuse)
        return {
            "refusal_correct": 1.0 if correct else 0.0
        }
