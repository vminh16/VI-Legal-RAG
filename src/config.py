import os
from pathlib import Path
from dotenv import load_dotenv

# Tự động tải biến môi trường từ tệp .env ở thư mục gốc
load_dotenv()

# Thư mục gốc của dự án (c:\Users\USER\Desktop\NLP_project\VI-Legal-RAG)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Các đường dẫn thư mục dữ liệu chính
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
BENCHMARK_DATA_DIR = DATA_DIR / "benchmark"

# File tài liệu luật gốc
RAW_LAW_PDF_PATH = RAW_DATA_DIR / "luat-lao-dong.pdf"
CORPUS_JSONL_PATH = PROCESSED_DATA_DIR / "corpus.jsonl"
BENCHMARK_JSON_PATH = BENCHMARK_DATA_DIR / "benchmark.json"

# Cấu hình thư mục lưu trữ mô hình offline
MODELS_DIR = PROJECT_ROOT / "models"
OFFLINE_EMBEDDING_DIR = MODELS_DIR / "bge-m3"
OFFLINE_RERANKER_DIR = MODELS_DIR / "bge-reranker-base"

# Cấu hình tên mô hình hoặc đường dẫn tải cục bộ (Tự động fallback sang offline nếu thư mục tồn tại)
EMBEDDING_MODEL_NAME = str(OFFLINE_EMBEDDING_DIR) if OFFLINE_EMBEDDING_DIR.exists() else "BAAI/bge-m3"
RERANKER_MODEL_NAME = str(OFFLINE_RERANKER_DIR) if OFFLINE_RERANKER_DIR.exists() else "BAAI/bge-reranker-base"
VECTOR_DB_DIR = PROCESSED_DATA_DIR / "faiss_index"

# API Key và Cấu hình LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL_NAME = "gemini-2.5-flash"  # Sử dụng model gemini-2.5-flash theo SDK mới nhất

# Thiết lập chế độ chạy
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() in ("true", "1", "yes")
