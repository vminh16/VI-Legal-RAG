from pydantic import BaseModel, Field
from src.config import MAX_QUERY_CHARS

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS, description="Câu hỏi pháp lý tiếng Việt cần tra cứu.")
    strategy: str = Field("hybrid_rerank", description="Chiến lược truy hồi: 'bm25', 'dense', 'hybrid', 'hybrid_rerank'")
    top_k: int = Field(5, ge=1, le=10, description="Số lượng văn bản truy hồi tối đa.")
    bypass_refusal: bool = Field(False, description="Cờ bỏ qua các bộ lọc từ chối để phục vụ debug kiểm thử.")

class CitationResponse(BaseModel):
    citation_id: int
    article: str
    clause: str | None = None
    title: str
    source_url: str
    evidence: str

class ChunkResponse(BaseModel):
    chunk_id: str
    text: str
    score: float = 0.0
    article: str | None = None
    clause: str | None = None
    chapter: str | None = None
    title: str | None = None

class QueryResponse(BaseModel):
    answer: str = Field(..., description="Câu trả lời pháp lý từ hệ thống RAG.")
    citations: list[CitationResponse] = Field([], description="Danh sách các nguồn trích dẫn tương ứng.")
    confidence: float = Field(0.0, description="Độ tự tin của câu trả lời.")
    refused: bool = Field(False, description="Cờ trạng thái từ chối trả lời.")
    category: str = Field("in_scope", description="Phân loại kết quả (out_of_scope, administrative_fine, v.v.).")
    retrieved_chunks: list[ChunkResponse] = Field([], description="Danh sách các đoạn văn bản truy hồi phục vụ debug.")
