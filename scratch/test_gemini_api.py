import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import traceback
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

class Citation(BaseModel):
    citation_id: int
    article: str
    clause: str | None = None
    title: str
    source_url: str
    evidence: str

class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float

api_key = os.getenv("GEMINI_API_KEY", "")
print(f"API Key configured length: {len(api_key)}")

if not api_key:
    print("Không tìm thấy GEMINI_API_KEY trong env/file .env!")
    sys.exit(1)

try:
    print("Khởi tạo genai.Client...")
    client = genai.Client(api_key=api_key)
    
    print("Đang gọi generate_content...")
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=RAGResponse,
        temperature=0.0
    )
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Hãy trả lời câu hỏi: Thử việc là gì? Giả lập trích dẫn 1 nguồn bất kỳ.",
        config=config
    )
    
    print("KẾT QUẢ API TRẢ VỀ:")
    print("Response text:", response.text)
    
except Exception as e:
    print("LỖI KHI GỌI GEMINI API:")
    traceback.print_exc()
