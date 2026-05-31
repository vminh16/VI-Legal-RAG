import unicodedata
import re
from rank_bm25 import BM25Okapi

sample_chunks = [
    {
        "chunk_id": "bll2019_dieu_1",
        "text": "Bộ luật Lao động quy định tiêu chuẩn lao động; quyền, nghĩa vụ, trách nhiệm của người lao động."
    },
    {
        "chunk_id": "bll2019_dieu_24_khoan_1",
        "text": "Người sử dụng lao động và người lao động có thỏa thuận về việc thử việc, quyền và nghĩa vụ của hai bên trong thời gian thử việc."
    }
]

def _tokenize(text: str) -> list[str]:
    # Chuẩn hóa NFC
    normalized = unicodedata.normalize("NFC", text)
    lowercased = normalized.lower()
    cleaned = re.sub(r'[^\w\s]', ' ', lowercased)
    return cleaned.split()

# In kết quả token hóa
print("Chunk 1 tokens:", _tokenize(sample_chunks[0]["text"]))
print("Chunk 2 tokens:", _tokenize(sample_chunks[1]["text"]))

query = "thử việc"
tokenized_query = _tokenize(query)
print("Query tokens:", tokenized_query)

# Lập chỉ mục
tokenized_corpus = [_tokenize(c["text"]) for c in sample_chunks]
bm25 = BM25Okapi(tokenized_corpus)
scores = bm25.get_scores(tokenized_query)
print("Scores:", scores)
