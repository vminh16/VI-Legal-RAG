import json
import re
from pathlib import Path

def get_ground_truth_structure(text: str) -> dict:
    """
    Quét văn bản thô chuẩn để dựng cấu trúc chuẩn (Ground Truth): Chapter -> Section -> Articles.
    """
    lines = text.split('\n')
    structure = {}
    
    current_chapter = None
    current_chapter_title = None
    current_section = None
    current_section_title = None
    
    chapter_pattern = re.compile(r'^Ch\u01b0\u01a1ng\s+([IVXLCDM\d]+)', re.IGNORECASE)
    section_pattern = re.compile(r'^M\u1ee5c\s+(\d+)', re.IGNORECASE)
    article_pattern = re.compile(r'^\u0110i\u1ec1u\s+(\d+)\.\s*(.*)', re.IGNORECASE)
    
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
            
        # Phát hiện ranh giới Chương
        chapter_match = chapter_pattern.match(line)
        if chapter_match:
            current_chapter = line
            current_section = None
            current_section_title = None
            idx += 1
            if idx < len(lines) and lines[idx].strip():
                current_chapter_title = lines[idx].strip()
            
            structure[current_chapter] = {
                "title": current_chapter_title,
                "sections": {},
                "articles": {}
            }
            idx += 1
            continue
            
        # Phát hiện ranh giới Mục
        section_match = section_pattern.match(line)
        if section_match:
            current_section = line
            idx += 1
            if idx < len(lines) and lines[idx].strip():
                current_section_title = lines[idx].strip()
            
            if current_chapter:
                structure[current_chapter]["sections"][current_section] = {
                    "title": current_section_title,
                    "articles": []
                }
            idx += 1
            continue
            
        # Phát hiện ranh giới Điều
        article_match = article_pattern.match(line)
        if article_match:
            art_num = int(article_match.group(1))
            art_title = article_match.group(2).strip()
            
            if current_chapter:
                structure[current_chapter]["articles"][art_num] = {
                    "title": art_title,
                    "section": current_section,
                    "section_title": current_section_title
                }
                
                # Gán vào Mục nếu có
                if current_section and current_section in structure[current_chapter]["sections"]:
                    structure[current_chapter]["sections"][current_section]["articles"].append(art_num)
            idx += 1
            continue
            
        idx += 1
        
    return structure

def main():
    raw_txt_path = Path("data/raw/luat-lao-dong.txt")
    corpus_path = Path("data/processed/corpus_structured.jsonl")
    
    if not raw_txt_path.exists():
        print(f"Error: Không tìm thấy file text nguồn tại: {raw_txt_path}")
        return
    if not corpus_path.exists():
        print(f"Error: Không tìm thấy file corpus structured tại: {corpus_path}")
        return
        
    print("==========================================================")
    # 1. Trích xuất Ground Truth
    with open(raw_txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    # Lấy văn bản chính tương tự như pipeline
    import re
    match_start = re.search(r'(^|\n)(Ch\u01b0\u01a1ng I\b)', text, re.IGNORECASE)
    start_idx = match_start.start(2) if match_start else 0
    
    footer_keywords = [
        r'V\u0103n b\u1ea3n li\xean quan c\xf9ng n\u1ed9i dung',
        r'V\u0103n b\u1ea3n li\xean quan',
        r'V\u0103n b\u1ea3n h\u01b0\u1edbng d\u1eabn',
        r'\u0110\xc2Y L\xc0 N\u1ed8I DUNG C\xd3 THU PH\xcd'
    ]
    end_idx = len(text)
    for keyword in footer_keywords:
        match_end = re.search(keyword, text[start_idx:], re.IGNORECASE)
        if match_end:
            candidate_end = start_idx + match_end.start()
            if candidate_end < end_idx:
                end_idx = candidate_end
    main_text = text[start_idx:end_idx].strip()
    
    ground_truth = get_ground_truth_structure(main_text)
    
    # 2. Đọc corpus đã parse
    chunks = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
                
    print(f"Tổng số lượng chunk trong corpus structured: {len(chunks)}")
    
    # 3. Phân tích từng Điều trong corpus
    # Map art_num -> chapter, section, title
    corpus_articles = {}
    for idx, c in enumerate(chunks):
        art_str = c["article"]
        if not art_str:
            continue
        art_num = int(art_str.split()[1])
        if art_num not in corpus_articles:
            corpus_articles[art_num] = []
        corpus_articles[art_num].append(c)
        
    print(f"Tổng số Điều nhận dạng được trong corpus: {len(corpus_articles)}")
    
    # 4. Đối chiếu chéo (Cross-Verification)
    mismatches = []
    missing_in_corpus = []
    
    # Duyệt qua từng Chương và Điều của Ground Truth để kiểm tra
    print("\n--- ĐỐI CHIẾU PHÂN BỐ ĐIỀU VÀO CHƯƠNG ---")
    for chapter_name, cap_data in ground_truth.items():
        print(f"\n* {chapter_name}: {cap_data['title']}")
        art_nums_in_chapter = sorted(list(cap_data["articles"].keys()))
        if art_nums_in_chapter:
            print(f"  Phạm vi Điều thực tế: Điều {art_nums_in_chapter[0]} -> Điều {art_nums_in_chapter[-1]} (Tổng: {len(art_nums_in_chapter)} Điều)")
        else:
            print("  Không chứa Điều nào!")
            
        for art_num in art_nums_in_chapter:
            gt_art = cap_data["articles"][art_num]
            
            # Kiểm tra xem Điều này có trong corpus không
            if art_num not in corpus_articles:
                missing_in_corpus.append(art_num)
                continue
                
            # Kiểm tra xem tất cả các chunk thuộc Điều này trong corpus có đúng metadata Chương/Mục không
            for chunk in corpus_articles[art_num]:
                c_chapter = chunk["chapter"]
                c_section = chunk["section"]
                
                # Chuẩn hóa so sánh (loại bỏ khoảng trắng, dấu để so khớp chính xác)
                def clean(s):
                    return re.sub(r'\s+', ' ', s.strip().lower()) if s else None
                
                if clean(c_chapter) != clean(chapter_name):
                    mismatches.append({
                        "article": art_num,
                        "chunk_id": chunk["chunk_id"],
                        "field": "chapter",
                        "expected": chapter_name,
                        "actual": c_chapter
                    })
                    
                if clean(c_section) != clean(gt_art["section"]):
                    mismatches.append({
                        "article": art_num,
                        "chunk_id": chunk["chunk_id"],
                        "field": "section",
                        "expected": gt_art["section"],
                        "actual": c_section
                    })
                    
    print("\n================== KẾT QUẢ ĐỐI CHIẾU SÂU ==================")
    if not missing_in_corpus:
        print("[SUCCESS] 100% các Điều trong văn bản thô đều có mặt đầy đủ trong corpus đã phân đoạn.")
    else:
        print(f"[FAIL] Có {len(missing_in_corpus)} Điều bị mất tích trong corpus: {missing_in_corpus}")
        
    if not mismatches:
        print("[SUCCESS] 100% các Điều được ánh xạ vào Chương và Mục chính xác hoàn hảo! Không có hiện tượng chunking nhầm Điều sang Chương hay Mục khác.")
    else:
        print(f"[FAIL] Phát hiện {len(mismatches)} lỗi ánh xạ sai lệch Chương/Mục:")
        for m in mismatches[:10]:
            print(f" - Điều {m['article']} ({m['chunk_id']}): {m['field']} thực tế là '{m['actual']}', mong muốn '{m['expected']}'")
        if len(mismatches) > 10:
            print(f" ... và {len(mismatches) - 10} lỗi khác.")
            
    print("==========================================================")

if __name__ == '__main__':
    main()
