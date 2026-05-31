import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder

# Đảm bảo import được src
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import MODELS_DIR, OFFLINE_EMBEDDING_DIR, OFFLINE_RERANKER_DIR

def download_and_cache_models():
    """
    Tải xuống các mô hình và lưu cục bộ vào thư mục models/ để tối ưu hóa hiệu suất load.
    Cho phép hệ thống hoạt động offline 100% không cần tải lại từ HuggingFace.
    """
    print("=== BẮT ĐẦU TẢI & CACHE MÔ HÌNH OFFLINE ===")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Tải và lưu trữ mô hình Embedding: BAAI/bge-m3
    embedding_src = "BAAI/bge-m3"
    print(f"\n1. Đang tải mô hình Embedding: {embedding_src}...")
    try:
        model = SentenceTransformer(embedding_src)
        print(f"-> Đang lưu mô hình vào: {OFFLINE_EMBEDDING_DIR}...")
        model.save(str(OFFLINE_EMBEDDING_DIR))
        print("[SUCCESS] Đã lưu thành công mô hình Embedding offline!")
    except Exception as e:
        print(f"[ERROR] Không thể tải mô hình Embedding: {e}")
        
    # 2. Tải và lưu trữ mô hình Reranker: BAAI/bge-reranker-base
    reranker_src = "BAAI/bge-reranker-base"
    print(f"\n2. Đang tải mô hình Reranker: {reranker_src}...")
    try:
        reranker = CrossEncoder(reranker_src)
        print(f"-> Đang lưu mô hình vào: {OFFLINE_RERANKER_DIR}...")
        reranker.save(str(OFFLINE_RERANKER_DIR))
        print("[SUCCESS] Đã lưu thành công mô hình Reranker offline!")
    except Exception as e:
        print(f"[ERROR] Không thể tải mô hình Reranker: {e}")

    print("\n=== HOÀN THÀNH QUÁ TRÌNH TẢI MÔ HÌNH OFFLINE ===")

if __name__ == '__main__':
    download_and_cache_models()
