import json
from pathlib import Path
from src.config import RAW_LAW_PDF_PATH, PROCESSED_DATA_DIR
from src.ingest.pdf_extractor import PDFExtractor
from src.chunking.fixed_chunker import FixedChunker
from src.chunking.structure_chunker import StructureChunker

def save_jsonl(chunks: list[dict], dest_path: Path):
    """Ghi danh sách các chunk vào file định dạng JSON Lines."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

def validate_corpus(chunks: list[dict], name: str):
    """Kiểm tra và kiểm định chất lượng corpus đầu ra."""
    print(f"\n--- Kiểm định chất lượng cho {name} ---")
    print(f"Tổng số lượng chunks: {len(chunks)}")
    
    # 1. Kiểm tra các trường bắt buộc
    required_keys = {"chunk_id", "document_title", "law_number", "chapter", "text", "source_url"}
    all_valid = True
    for idx, c in enumerate(chunks):
        missing = required_keys - set(c.keys())
        if missing:
            print(f"[ERROR] Chunk {idx} thiếu các trường: {missing}")
            all_valid = False
            break
            
        if not c["text"].strip():
            print(f"[ERROR] Chunk {c['chunk_id']} có nội dung text trống rỗng!")
            all_valid = False
            break
            
    if all_valid:
        print("[SUCCESS] Tất cả các chunk đều đầy đủ các trường dữ liệu và không bị rỗng.")
        
    # 2. Thống kê chi tiết đối với Structure Chunker
    if "Structured" in name:
        articles = set(c["article"] for c in chunks if c["article"])
        chapters = set(c["chapter"] for c in chunks if c["chapter"])
        clauses = sum(1 for c in chunks if c["clause"])
        
        print(f"Số lượng Chương nhận diện được: {len(chapters)}")
        print(f"Số lượng Điều nhận diện được: {len(articles)}")
        print(f"Số lượng Khoản được chia nhỏ riêng biệt: {clauses}")
        
        # In kiểm định chất lượng: 10 điều đầu, 10 điều giữa và 10 điều cuối
        sorted_articles = sorted(list(articles), key=lambda x: int(x.split()[1]) if len(x.split()) > 1 and x.split()[1].isdigit() else 999)
        print(f"Danh sách 10 Điều đầu tiên: {sorted_articles[:10]}")
        print(f"Danh sách 10 Điều cuối cùng: {sorted_articles[-10:]}")
        
        # Đảm bảo nhận diện được hơn 200 Điều (Bộ luật Lao động 2019 có 220 Điều)
        if len(articles) >= 200:
            print(f"[SUCCESS] Bộ parser nhận dạng thành công {len(articles)} Điều (đạt yêu cầu tối thiểu > 200 Điều).")
        else:
            print(f"[WARNING] Bộ parser chỉ nhận dạng được {len(articles)} Điều. Cần kiểm tra lại regex ranh giới Điều.")

def main():
    print("=== BẮT ĐẦU PIPELINE TRÍCH XUẤT & PHÂN ĐOẠN LUẬT ===")
    
    # 1. Khởi tạo PDFExtractor
    print(f"\nBước 1: Khởi tạo PDFExtractor với file: {RAW_LAW_PDF_PATH.name}")
    extractor = PDFExtractor(RAW_LAW_PDF_PATH)
    
    # 2. Trích xuất Raw Text & Metadata cấp tài liệu
    print("Bước 2: Trích xuất văn bản thô & Metadata cấp tài liệu...")
    raw_text = extractor.extract_raw_text()
    metadata = extractor.extract_document_metadata()
    print(f"[OK] Trích xuất hoàn tất. Độ dài văn bản thô: {len(raw_text)} ký tự.")
    print(f"[Metadata] Tiêu đề: {metadata['document_title']}")
    print(f"[Metadata] Số hiệu: {metadata['law_number']}")
    
    # 3. Tách văn bản chính (loại bỏ phần mở đầu hành chính)
    print("Bước 3: Loại bỏ phần mở đầu hành chính rác trước 'Chương I'...")
    main_text = extractor.get_main_text()
    print(f"[OK] Tách xong. Độ dài văn bản chính thức: {len(main_text)} ký tự.")
    print(f"[Thông tin] Đã loại bỏ {len(raw_text) - len(main_text)} ký tự văn bản hành chính mở đầu.")
    
    # 4. Phân đoạn kích thước cố định (Baseline Fixed Chunker)
    print("\nBước 4: Tiến hành Fixed-size Chunking (Baseline)...")
    fixed_chunker = FixedChunker(chunk_size=500, chunk_overlap=100)
    fixed_chunks = fixed_chunker.split_to_chunks(main_text, metadata)
    fixed_dest = PROCESSED_DATA_DIR / "corpus_fixed.jsonl"
    save_jsonl(fixed_chunks, fixed_dest)
    print(f"[OK] Đã xuất bản Fixed Corpus tại: {fixed_dest}")
    validate_corpus(fixed_chunks, "Baseline Fixed Corpus")
    
    # 5. Phân đoạn cấu trúc (Structure-aware Chunker)
    print("\nBước 5: Tiến hành Structure-aware Chunking (Cấu trúc)...")
    structure_chunker = StructureChunker(max_article_len=1200)
    structured_chunks = structure_chunker.split_to_chunks(main_text, metadata)
    structured_dest = PROCESSED_DATA_DIR / "corpus_structured.jsonl"
    save_jsonl(structured_chunks, structured_dest)
    print(f"[OK] Đã xuất bản Structured Corpus tại: {structured_dest}")
    validate_corpus(structured_chunks, "Structure-aware Structured Corpus")
    
    print("\n=== PIPELINE HOÀN THÀNH THÀNH CÔNG ===")

if __name__ == '__main__':
    main()
