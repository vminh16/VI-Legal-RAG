import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Tải biến môi trường nhạy cảm từ tệp .env ở thư mục gốc
load_dotenv()

# Xác định thư mục gốc của dự án (c:\Users\USER\Desktop\NLP_project\VI-Legal-RAG)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "settings.yml"

# Bộ cấu hình dự phòng mặc định (Self-healing) phòng khi tệp YAML bị xóa hoặc lỗi
DEFAULT_SETTINGS = {
    "paths": {
        "raw_law_pdf": "data/raw/luat-lao-dong.pdf",
        "corpus_jsonl": "data/processed/corpus_structured.jsonl",
        "benchmark_json": "data/benchmark/benchmark.json",
        "vector_db_dir": "data/processed/faiss_index",
        "models_dir": "models"
    },
    "models": {
        "embedding": "BAAI/bge-m3",
        "reranker": "BAAI/bge-reranker-base",
        "llm": "gemini-2.5-flash"
    },
    "retrieval": {
        "top_k": 5,
        "rrf_k": 60,
        "thresholds": {
            "dense": 0.35,
            "bm25": 1.0,
            "hybrid": 0.012,
            "hybrid_rerank": -2.0
        }
    },
    "generation": {
        "temperature": 0.0,
        "default_disclaimer": "Lưu ý: Ý kiến tư vấn trên chỉ mang tính chất tham khảo dựa trên quy định của Bộ luật Lao động 2019 và dữ liệu ngữ cảnh hiện có tại thời điểm tra cứu."
    }
}

# Tự động tạo thư mục configs và tệp settings.yml nếu chưa tồn tại
if not CONFIG_PATH.parent.exists():
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

if not CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(DEFAULT_SETTINGS, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        print(f"Cảnh báo: Không thể tạo tệp cấu hình settings.yml mặc định: {e}")

# Nạp cấu hình từ tệp YAML
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)
        if not settings:
            raise ValueError("Tệp cấu hình rỗng.")
except Exception as e:
    print(f"Cảnh báo: Không thể nạp settings.yml, sử dụng cấu hình mặc định. Lỗi: {e}")
    settings = DEFAULT_SETTINGS

# Hàm phân giải đường dẫn tương đối thành tuyệt đối dựa trên thư mục gốc dự án
def get_absolute_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


# --- XUẤT CÁC HẰNG SỐ CẤU HÌNH (Duy trì tính tương thích ngược hoàn hảo) ---

# Các đường dẫn tuyệt đối
RAW_LAW_PDF_PATH = get_absolute_path(settings["paths"]["raw_law_pdf"])
CORPUS_JSONL_PATH = get_absolute_path(settings["paths"]["corpus_jsonl"])
BENCHMARK_JSON_PATH = get_absolute_path(settings["paths"]["benchmark_json"])
VECTOR_DB_DIR = get_absolute_path(settings["paths"]["vector_db_dir"])
MODELS_DIR = get_absolute_path(settings["paths"]["models_dir"])

# Thư mục lưu trữ mô hình offline
OFFLINE_EMBEDDING_DIR = MODELS_DIR / "bge-m3"
OFFLINE_RERANKER_DIR = MODELS_DIR / "bge-reranker-base"

# Cấu hình tên mô hình hoặc đường dẫn tải cục bộ (Tự động ưu tiên mô hình offline nếu tồn tại)
EMBEDDING_MODEL_NAME = str(OFFLINE_EMBEDDING_DIR) if OFFLINE_EMBEDDING_DIR.exists() else settings["models"]["embedding"]
RERANKER_MODEL_NAME = str(OFFLINE_RERANKER_DIR) if OFFLINE_RERANKER_DIR.exists() else settings["models"]["reranker"]
LLM_MODEL_NAME = settings["models"]["llm"]

# API Key và Cấu hình Môi trường
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() in ("true", "1", "yes")

# Xuất bản sao cấu hình gốc dạng từ điển nếu các module khác cần truy cập trực tiếp
SETTINGS = settings
