import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def rate_limit_and_retry(max_retries: int = 5, initial_backoff: float = 10.0, delay_between_calls: float = 4.0):
    """
    Decorator bọc các hàm gọi API Gemini để:
    1. Duy trì khoảng cách delay_between_calls giây giữa các cuộc gọi (throttle).
    2. Tự động thử lại khi gặp lỗi 429 (Rate Limit / Resource Exhausted) bằng exponential backoff.
    """
    # Dùng list chứa float làm bộ nhớ trạng thái để chia sẻ giữa các cuộc gọi của wrapper
    last_call_time = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 1. Thực hiện throttle: Chờ nếu khoảng cách cuộc gọi trước quá ngắn
            elapsed = time.time() - last_call_time[0]
            if elapsed < delay_between_calls:
                wait_time = delay_between_calls - elapsed
                logger.info(f"Rate Limiter: Chờ {wait_time:.2f} giây trước khi gửi request tiếp theo...")
                time.sleep(wait_time)
            
            # 2. Thử lại với exponential backoff khi gặp lỗi rate limit
            backoff = initial_backoff
            for attempt in range(1, max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    last_call_time[0] = time.time()  # Cập nhật thời điểm gọi thành công
                    return result
                except Exception as e:
                    err_msg = str(e)
                    is_rate_limit = any(
                        keyword in err_msg.lower() 
                        for keyword in ["429", "resourceexhausted", "resource_exhausted", "rate limit", "limit exceeded"]
                    )
                    
                    if is_rate_limit and attempt < max_retries:
                        logger.warning(
                            f"Lỗi Rate Limit (429/ResourceExhausted) phát hiện tại cuộc gọi API (Attempt {attempt}/{max_retries}). "
                            f"Đang ngủ chờ {backoff:.2f} giây trước khi thử lại..."
                        )
                        time.sleep(backoff)
                        backoff *= 2.0  # Nhân đôi thời gian chờ
                    else:
                        # Nếu là lỗi khác hoặc đã hết lượt thử lại, ném lỗi ra ngoài
                        raise e
            
        return wrapper
    return decorator


def wrap_pipeline_with_rate_limiter(pipeline, delay_between_calls: float = 4.0):
    """
    Bọc động (Monkey Patch) các phương thức có gọi API Gemini trong RAGPipeline 
    bằng bộ Rate Limiter để bảo vệ API khỏi bị Rate Limit trong lúc chạy thử nghiệm.
    """
    # 1. Bọc phương thức detect_query_refusal của refusal_detector
    if hasattr(pipeline, "refusal_detector") and hasattr(pipeline.refusal_detector, "detect_query_refusal"):
        orig_detect = pipeline.refusal_detector.detect_query_refusal
        pipeline.refusal_detector.detect_query_refusal = rate_limit_and_retry(
            delay_between_calls=delay_between_calls
        )(orig_detect)
        logger.info("Đã bọc Rate Limiter cho refusal_detector.detect_query_refusal")

    # 2. Bọc phương thức generate_answer của generator
    if hasattr(pipeline, "generator") and hasattr(pipeline.generator, "generate_answer"):
        orig_generate = pipeline.generator.generate_answer
        pipeline.generator.generate_answer = rate_limit_and_retry(
            delay_between_calls=delay_between_calls
        )(orig_generate)
        logger.info("Đã bọc Rate Limiter cho generator.generate_answer")

    # 3. Bọc phương thức check_faithfulness của faithfulness_checker
    if hasattr(pipeline, "faithfulness_checker") and hasattr(pipeline.faithfulness_checker, "check_faithfulness"):
        orig_check = pipeline.faithfulness_checker.check_faithfulness
        pipeline.faithfulness_checker.check_faithfulness = rate_limit_and_retry(
            delay_between_calls=delay_between_calls
        )(orig_check)
        logger.info("Đã bọc Rate Limiter cho faithfulness_checker.check_faithfulness")
        
    return pipeline
