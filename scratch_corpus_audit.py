import json
import re
from pathlib import Path

def main():
    corpus_path = Path("data/processed/corpus_structured.jsonl")
    if not corpus_path.exists():
        print(f"Error: Không tìm thấy file corpus structured tại: {corpus_path}")
        return

    print("=== BAT DAU KIEM DINH TOAN DIEN CORPUS STRUCTURED ===")
    
    # Đọc tất cả các chunk từ file JSON Lines
    chunks = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
                
    print(f"Tong so luong chunk nap duoc: {len(chunks)}")
    
    # ----------------------------------------------------
    # BÀI KIỂM TRA 1: Tính đầy đủ và tuần tự của các Điều
    # ----------------------------------------------------
    articles_found = {}
    for idx, c in enumerate(chunks):
        art = c["article"]
        if art:
            art_num = int(art.split()[1])
            if art_num not in articles_found:
                articles_found[art_num] = []
            articles_found[art_num].append((idx, c))
            
    print(f"\n1. Kiem tra danh sach cac Dieu:")
    print(f" - Tong so Dieu doc duy nhat duoc: {len(articles_found)}")
    
    # Kiểm tra xem có bị bỏ sót Điều nào từ 1 đến 220 không
    missing_articles = []
    for i in range(1, 221):
        if i not in articles_found:
            missing_articles.append(i)
            
    if not missing_articles:
        print(" [SUCCESS] Khong co Dieu nao bi bo sot. Tat ca 220 Dieu tu 1 den 220 deu hien dien day du!")
    else:
        print(f" [WARNING] Phat hien thieu cac Dieu sau: {missing_articles}")

    # ----------------------------------------------------
    # BÀI KIỂM TRA 2: Kiểm định ranh giới Chương (Chapter Mismatch)
    # ----------------------------------------------------
    print(f"\n2. Doi chieu ranh gioi va anh xa Chuong (Chapter Mapping):")
    # In ra một số Điều va Chương tuong ung de đối chứng thu cong
    test_articles = [1, 9, 13, 34, 49, 52, 59, 63, 65, 75, 90, 105, 116, 121, 136, 144, 161, 179, 186, 192, 201, 210, 220]
    for art_num in test_articles:
        if art_num in articles_found:
            first_chunk = articles_found[art_num][0][1]
            chapter = first_chunk["chapter"]
            title = first_chunk["title"]
            print(f" - {first_chunk['article']}: {title[:30]}... thuộc [{chapter}]")

    # ----------------------------------------------------
    # BÀI KIỂM TRA 3: Kiem tra khop ID va Metadata
    # ----------------------------------------------------
    print(f"\n3. Kiem tra tinh nhat quan giua chunk_id, article va clause:")
    mismatch_id_count = 0
    for art_num, chunk_list in articles_found.items():
        for idx, c in chunk_list:
            chunk_id = c["chunk_id"]
            article = c["article"]
            clause = c["clause"]
            
            # Kiểm tra xem chunk_id có chứa số Điều tương ứng không
            expected_art_str = f"dieu_{art_num}"
            if expected_art_str not in chunk_id:
                print(f" [ERROR] Mismatch ID: chunk_id='{chunk_id}' khong khop voi article='{article}'")
                mismatch_id_count += 1
                
            # Kiểm tra xem chunk_id của Khoản có chứa số Khoản tương ứng không
            if clause:
                expected_clause_str = f"khoan_{clause}"
                if expected_clause_str not in chunk_id:
                    print(f" [ERROR] Mismatch Clause ID: chunk_id='{chunk_id}' thieu thong tin clause='{clause}'")
                    mismatch_id_count += 1
                    
    if mismatch_id_count == 0:
        print(" [SUCCESS] 100% chunk_id deu nhat quan va chinh xac voi article/clause metadata!")
    else:
        print(f" [ERROR] Phat hien {mismatch_id_count} loi khong nhat quan trong ID.")

    # ----------------------------------------------------
    # BÀI KIỂM TRA 4: Kiem tra lan ranh gioi giua cac Dieu (Boundary Leak Check)
    # ----------------------------------------------------
    print(f"\n4. Do tim hien tuong ro ri ranh gioi (Boundary Leaks):")
    leak_count = 0
    for art_num, chunk_list in articles_found.items():
        for idx, c in chunk_list:
            text = c["text"]
            # Tim kiem tieu de Dieu khac bat dau o dau dong mới ben trong chunk
            # Vi du: Dieu 3 chunk khong duoc chua dong khai bao "Điều 4." o dau dong
            leaked_matches = re.findall(r'\n\u0110i\u1ec1u\s+(\d+)\.', text)
            for l_num in leaked_matches:
                if int(l_num) != art_num:
                    print(f" [ERROR] Rò rỉ ranh giới: {c['chunk_id']} chứa tiêu đề khai báo của '{c['article']}' khac!")
                    leak_count += 1
                    
    if leak_count == 0:
        print(" [SUCCESS] Khong phat hien bat ky su ro ri hay lan ranh gioi nao giua cac Dieu. Ranh gioi phan cat hoan toan tuyet doi!")
    else:
        print(f" [ERROR] Phat hien {leak_count} loi ro ri ranh gioi giua cac Dieu.")

    print("\n=== KET THUC KIEM DINH CORPUS ===")

if __name__ == '__main__':
    main()
