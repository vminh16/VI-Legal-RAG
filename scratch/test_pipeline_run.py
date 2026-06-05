import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import logging
import traceback
from src.pipeline.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)

try:
    print("Khởi tạo RAGPipeline...")
    pipeline = RAGPipeline()
    print("Khởi tạo RAGPipeline thành công!")
    
    query = "Thử việc có được trả lương không?"
    print(f"Đang gọi answer_question cho câu hỏi: '{query}'...")
    result = pipeline.answer_question(query=query, strategy="hybrid_rerank", top_k=5)
    print("KẾT QUẢ TRẢ VỀ:")
    print(result)
except Exception as e:
    print("LỖI PHÁT SINH:")
    traceback.print_exc()
