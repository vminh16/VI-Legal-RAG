import sys
import io
import json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from src.pipeline.rag_pipeline import RAGPipeline
from src.generation.generator import RAGResponse, Citation

# Khởi tạo pipeline và các checkers
pipeline = RAGPipeline()
citation_checker = pipeline.citation_checker
faithfulness_checker = pipeline.faithfulness_checker

# Giả lập câu trả lời từ LLM (câu hỏi: ngày lễ tết thì được tính lương thế nào)
mock_answer = (
    "Theo Bộ luật Lao động 2019, tiền lương vào các ngày lễ, tết được tính như sau: "
    "Người lao động được nghỉ làm việc và hưởng nguyên lương trong các ngày lễ, tết theo quy định của pháp luật [1]. "
    "Nếu người lao động làm việc vào ngày lễ, tết, họ sẽ được trả lương ít nhất bằng 300% đơn giá tiền lương hoặc tiền lương thực trả theo công việc đang làm, chưa kể tiền lương ngày lễ, tết mà họ đã được hưởng nguyên lương [2]. "
    "Cụ thể các ngày lễ, tết mà người lao động được hưởng nguyên lương bao gồm [1]:\n"
    "*   Tết Dương lịch: 01 ngày (ngày 01 tháng 01 dương lịch).\n"
    "*   Tết Âm lịch: 05 ngày.\n"
    "*   Ngày Chiến thắng: 01 ngày (ngày 30 tháng 4 dương lịch).\n"
    "*   Ngày Quốc tế lao động: 01 ngày (ngày 01 tháng 5 dương lịch).\n"
    "*   Quốc khánh: 02 ngày (ngày 02 tháng 9 dương lịch và 01 ngày liền kề trước hoặc sau).\n"
    "*   Ngày Giỗ Tổ Hùng Vương: 01 ngày (ngày 10 tháng 3 âm lịch).\n"
    "Ngoài ra, nếu người lao động làm thêm giờ vào ban đêm trong ngày lễ, tết, họ còn được trả thêm 20% tiền lương tính theo đơn giá tiền lương hoặc tiền lương theo công việc làm vào ban ngày của ngày lễ, tết đó, bên cạnh các khoản lương đã nêu trên [3].\n"
    "Lưu ý: Ý kiến tư vấn trên chỉ mang tính chất tham khảo dựa trên quy định của Bộ luật Lao động 2019 và dữ liệu ngữ cảnh hiện có tại thời điểm tra cứu."
)

mock_citations = [
    Citation(
        citation_id=1,
        article="Điều 112",
        clause="Khoản 1",
        title="Nghỉ lễ, tết",
        source_url="https://vanban.chinhphu.vn/?docid=198540",
        evidence=(
            "1. Người lao động được nghỉ làm việc, hưởng nguyên lương trong những ngày lễ, tết sau đây:\n"
            "a) Tết Dương lịch: 01 ngày (ngày 01 tháng 01 dương lịch);\n"
            "b) Tết Âm lịch: 05 ngày;\n"
            "c) Ngày Chiến thắng: 01 ngày (ngày 30 tháng 4 dương lịch);\n"
            "d) Ngày Quốc tế lao động: 01 ngày (ngày 01 tháng 5 dương lịch);\n"
            "đ) Quốc khánh: 02 ngày (ngày 02 tháng 9 dương lịch và 01 ngày liền kề trước hoặc sau);\n"
            "e) Ngày Giỗ Tổ Hùng Vương: 01 ngày (ngày 10 tháng 3 âm lịch)."
        )
    ),
    Citation(
        citation_id=2,
        article="Điều 98",
        clause="Khoản 1",
        title="Tiền lương làm thêm giờ, làm việc vào ban đêm",
        source_url="https://vanban.chinhphu.vn/?docid=198540",
        evidence=(
            "1. Người lao động làm thêm giờ được trả lương tính theo đơn giá tiền lương hoặc tiền lương thực trả theo công việc đang làm như sau:\n"
            "... c) Vào ngày nghỉ lễ, tết, ngày nghỉ có hưởng lương, ít nhất bằng 300% chưa kể tiền lương ngày lễ, tết, ngày nghỉ có hưởng lương đối với người lao động hưởng lương ngày."
        )
    ),
    Citation(
        citation_id=3,
        article="Điều 98",
        clause="Khoản 3",
        title="Tiền lương làm thêm giờ, làm việc vào ban đêm",
        source_url="https://vanban.chinhphu.vn/?docid=198540",
        evidence=(
            "3. Người lao động làm thêm giờ vào ban đêm thì ngoài việc trả lương theo quy định tại khoản 1 và khoản 2 Điều này, người lao động còn được trả thêm 20% tiền lương tính theo đơn giá tiền lương hoặc tiền lương theo công việc làm vào ban ngày của ngày làm việc bình thường hoặc của ngày nghỉ hằng tuần hoặc của ngày nghỉ lễ, tết."
        )
    )
]

response = RAGResponse(
    answer=mock_answer,
    citations=mock_citations,
    confidence=1.0
)

# Chạy thử retrieved chunks giả lập (để kiểm tra citation_not_in_retrieved_context)
retrieved_chunks = [
    {"article": "Điều 112", "clause": None, "text": "Điều 112. Nghỉ lễ, tết\n1. Người lao động được nghỉ làm việc, hưởng nguyên lương trong những ngày lễ, tết sau đây:..."},
    {"article": "Điều 98", "clause": None, "text": "Điều 98. Tiền lương làm thêm giờ, làm việc vào ban đêm\n1. Người lao động làm thêm giờ... \n3. Người lao động làm thêm giờ vào ban đêm..."}
]

print("=== CHẠY KIỂM THỬ OFF-LINE CHECKERS ===")

# 1. Test Citation Checker
print("\n[Citation Checker Test]")
cit_report = citation_checker.check_citations(response, retrieved_chunks)
print(f"  + result is_valid: {cit_report.get('is_valid')}")
print(f"  + errors: {cit_report.get('errors')}")
print(f"  + details:")
for det in cit_report.get("details", []):
    print(f"    - ID: {det['citation_id']} - Article: {det['article']} - Clause: {det['clause']} -> Valid: {det['is_valid']}, Errors: {det['errors']}")

# 2. Test segment_claims trong Faithfulness Checker
print("\n[Claims Segmentation Test]")
claims = faithfulness_checker.check_faithfulness.__globals__['segment_claims'](mock_answer)
print(f"  + Tách được {len(claims)} claims:")
for idx, c in enumerate(claims):
    print(f"    - Claim {idx+1} (Source [{c['citation_id']}]): '{c['claim']}'")

# 3. Test check_numeric_discrepancy trong Faithfulness Checker
print("\n[Numeric Discrepancy Rule Test]")
check_numeric = faithfulness_checker.check_faithfulness.__globals__['check_numeric_discrepancy']
conflicts = []
citation_map = {c.citation_id: c for c in mock_citations}

for c_info in claims:
    claim_text = c_info["claim"]
    cit_id = c_info["citation_id"]
    cit = citation_map.get(cit_id)
    if cit:
        evidence = cit.evidence
        err = check_numeric(claim_text, evidence, cit.article, cit.clause)
        if err:
            print(f"    - Phát hiện lỗi ở Claim [{cit_id}]: {err}")
            conflicts.append(err)
        else:
            print(f"    - Claim [{cit_id}] đối chiếu số ĐẠT (OK)")

print(f"  + Tổng số lỗi mâu thuẫn số thô: {len(conflicts)}")

# 4. Test trường hợp LLM bịa đặt Khoản luật (Anti-hallucination Test)
print("\n[Anti-hallucination Verification Test]")
hallucinated_citations = [
    Citation(
        citation_id=1,
        article="Điều 112",
        clause="Khoản 5",  # Điều 112 chỉ có 3 Khoản, Khoản 5 không tồn tại
        title="Nghỉ lễ, tết",
        source_url="...",
        evidence="5. Người lao động được nghỉ làm thêm ngày lễ..."
    )
]
hallucinated_response = RAGResponse(
    answer="Người lao động được nghỉ thêm ngày lễ [1].",
    citations=hallucinated_citations,
    confidence=1.0
)
hallucinated_report = citation_checker.check_citations(hallucinated_response, retrieved_chunks)
print(f"  + Trường hợp trích dẫn Khoản 5 Điều 112 (bịa đặt):")
print(f"    - result is_valid (nên là False): {hallucinated_report.get('is_valid')}")
print(f"    - errors (nên có 'citation_not_found_in_corpus'): {hallucinated_report.get('errors')}")
