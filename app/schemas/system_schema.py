from pydantic import BaseModel, Field

class SystemStatusResponse(BaseModel):
    status: str = Field("ready", description="Trạng thái hoạt động của hệ thống RAG Backend.")
    api_key_configured: bool = Field(..., description="Cờ xác nhận cấu hình khóa API Gemini.")
    models: dict = Field(..., description="Thông số cấu hình mô hình máy học đang chạy.")
    thresholds: dict = Field(..., description="Ngưỡng sàn lọc từ chối của RAG Pipeline.")
    disclaimer: str = Field(..., description="Câu miễn trừ trách nhiệm mặc định.")
