from src.pipeline.rag_pipeline import RAGPipeline

_pipeline_instance: RAGPipeline | None = None

def get_rag_pipeline() -> RAGPipeline:
    """
    Dependency Injection provider để lấy thực thể RAGPipeline duy nhất (Singleton).
    Giúp tránh khởi tạo lại các mô hình cồng kềnh trong mỗi request.
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        # Tự động khởi tạo dự phòng nếu chưa nạp qua Lifespan
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance

def set_rag_pipeline(pipeline: RAGPipeline):
    """Lưu thực thể RAGPipeline vào Singleton instance (thường gọi khi startup)."""
    global _pipeline_instance
    _pipeline_instance = pipeline
