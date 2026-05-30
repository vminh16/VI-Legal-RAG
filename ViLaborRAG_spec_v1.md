# Đặc tả đề tài: Hệ thống hỏi đáp Bộ luật Lao động Việt Nam có trích dẫn nguồn bằng Retrieval-Augmented Generation

**Tên tiếng Việt:** Xây dựng hệ thống hỏi đáp Bộ luật Lao động Việt Nam có trích dẫn nguồn bằng Retrieval-Augmented Generation  
**Tên tiếng Anh:** Vietnamese Labor Code Question Answering with Citation-grounded Retrieval-Augmented Generation  
**Tên ngắn:** ViLaborRAG  
**Phiên bản:** 1.0  
**Ngày:** 2026-05-30  
**Phạm vi:** Đồ án môn học NLP/AI, 6–8 tuần, cá nhân hoặc nhóm 2–3 người  
**Domain:** Hỏi đáp văn bản pháp luật tiếng Việt, tập trung vào Bộ luật Lao động 2019  
**Định hướng:** RAG có trích dẫn nguồn, đánh giá truy hồi, đánh giá citation, kiểm soát hallucination và cơ chế từ chối khi thiếu căn cứ  

---

## 1. Tóm tắt đề tài

Đề tài xây dựng một hệ thống hỏi đáp tiếng Việt cho **Bộ luật Lao động 2019**, trong đó người dùng có thể đặt câu hỏi tự nhiên về các nội dung thường gặp như thử việc, hợp đồng lao động, nghỉ hằng năm, nghỉ việc, lương, làm thêm giờ, kỷ luật lao động, chấm dứt hợp đồng và quyền/nghĩa vụ của người lao động.

Hệ thống không trả lời bằng kiến thức sẵn có của mô hình ngôn ngữ. Thay vào đó, hệ thống phải:

1. Truy hồi các điều/khoản liên quan từ Bộ luật Lao động 2019.
2. Sinh câu trả lời tiếng Việt ngắn gọn, dễ hiểu.
3. Gắn trích dẫn nguồn cho từng ý chính.
4. Từ chối hoặc cảnh báo khi không tìm thấy căn cứ đủ rõ trong văn bản.
5. Lưu log để đánh giá retrieval, citation và chất lượng câu trả lời.

Đề tài không nhằm thay thế luật sư, cơ quan quản lý nhà nước hoặc tư vấn pháp lý chuyên nghiệp. Hệ thống chỉ là công cụ hỗ trợ **tra cứu thông tin có căn cứ từ văn bản nguồn**.

---

## 2. Lý do chọn Bộ luật Lao động 2019

### 2.1. Tính ứng dụng

Bộ luật Lao động là một trong các văn bản có tính ứng dụng cao trong đời sống hằng ngày. Người lao động, sinh viên đi làm thêm, nhân sự, doanh nghiệp nhỏ và người chuẩn bị ký hợp đồng đều có thể có nhu cầu tra cứu các vấn đề như:

- Thử việc tối đa bao lâu?
- Lương thử việc được trả như thế nào?
- Khi nghỉ việc có cần báo trước không?
- Một năm được nghỉ phép bao nhiêu ngày?
- Làm thêm giờ được tính lương thế nào?
- Công ty có được đơn phương chấm dứt hợp đồng không?
- Khi nào người lao động có thể bị sa thải?
- Hợp đồng lao động gồm những loại nào?

Bộ luật Lao động 2019, số 45/2019/QH14, được Cổng Thông tin điện tử Chính phủ ghi nhận là văn bản do Quốc hội ban hành ngày 20/11/2019 và có hiệu lực từ 01/01/2021. Nguồn chính thức này cần được ưu tiên khi xây dựng corpus của dự án.

Nguồn: https://vanban.chinhphu.vn/?docid=198540&pageid=27160

### 2.2. Tính khả thi

So với hệ thống hỏi đáp toàn bộ pháp luật Việt Nam, việc giới hạn vào một bộ luật giúp giảm mạnh độ phức tạp:

- Không cần xử lý quá nhiều văn bản luật, nghị định, thông tư.
- Không cần giải quyết đầy đủ bài toán sửa đổi, thay thế, bãi bỏ ở quy mô lớn.
- Dễ kiểm chứng câu trả lời hơn vì nguồn chỉ nằm trong một văn bản chính.
- Dễ tạo bộ câu hỏi benchmark thủ công.
- Dễ demo cho người dùng không chuyên.

Tuy nhiên, phạm vi hẹp cũng có giới hạn: nhiều câu hỏi thực tế về xử phạt, thủ tục, mức phạt hành chính hoặc hướng dẫn chi tiết có thể cần nghị định/thông tư. Với những câu hỏi này, hệ thống phải cảnh báo rằng corpus hiện tại chỉ gồm Bộ luật Lao động 2019.

---

## 3. Bối cảnh khoa học

### 3.1. Liên hệ với NLP

Đề tài thuộc nhóm bài toán **Question Answering**, **Information Retrieval**, **Document Understanding**, **Natural Language Generation** và **Grounded Generation**. Các thành phần trong hệ thống bám sát nhiều kiến thức NLP nền tảng:

- Tiền xử lý tiếng Việt.
- Tokenization và chuẩn hóa văn bản.
- Tách văn bản thành đơn vị có nghĩa.
- Biểu diễn văn bản bằng vector.
- Truy hồi thông tin.
- Hỏi đáp dựa trên tài liệu.
- Sinh câu trả lời bằng mô hình ngôn ngữ.
- Đánh giá độ đúng và độ bám nguồn.

### 3.2. Nền tảng RAG

Retrieval-Augmented Generation được Lewis và cộng sự đề xuất như một hướng kết hợp mô hình sinh ngôn ngữ với bộ nhớ ngoài dạng kho tài liệu truy hồi. Hướng này phù hợp với các tác vụ cần tri thức cụ thể vì mô hình có thể điều kiện hóa quá trình sinh trên các passage được truy hồi thay vì chỉ dựa vào trọng số bên trong mô hình.

Nguồn: Patrick Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020.  
https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html

### 3.3. Truy hồi trong QA

Dense Passage Retrieval của Karpukhin và cộng sự là một công trình quan trọng trong open-domain QA, dùng dual-encoder để truy hồi passage liên quan bằng biểu diễn dense. Công trình này là cơ sở để đưa dense retrieval vào pipeline RAG.

Nguồn: Vladimir Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering*, EMNLP 2020.  
https://aclanthology.org/2020.emnlp-main.550/

Tuy nhiên, trong nhiều domain, BM25 vẫn là baseline mạnh. BEIR cho thấy việc đánh giá retrieval cần xem xét khả năng tổng quát hóa qua nhiều tập dữ liệu và domain; BM25 vẫn là một baseline đáng so sánh, còn reranking thường có thể cải thiện chất lượng nhưng tăng chi phí tính toán.

Nguồn: Nandan Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*, NeurIPS Datasets and Benchmarks 2021.  
https://arxiv.org/abs/2104.08663

### 3.4. Vì sao không đưa toàn bộ bộ luật vào prompt

Nghiên cứu *Lost in the Middle* cho thấy mô hình ngôn ngữ không luôn sử dụng hiệu quả thông tin trong ngữ cảnh dài, đặc biệt khi thông tin liên quan nằm ở giữa context. Vì vậy, pipeline nên truy hồi và rerank các đoạn liên quan thay vì nhồi toàn bộ văn bản luật vào prompt.

Nguồn: Nelson F. Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, TACL 2024.  
https://aclanthology.org/2024.tacl-1.9/

---

## 4. Định nghĩa bài toán

### 4.1. Đầu vào

Hệ thống nhận:

- Một câu hỏi tiếng Việt tự nhiên `q`.
- Corpus `C` được xây dựng từ Bộ luật Lao động 2019.
- Metadata của từng chunk: tên văn bản, số hiệu văn bản, chương, mục, điều, khoản, điểm, tiêu đề, URL nguồn.

Ví dụ câu hỏi:

```text
Thử việc có được trả lương không?
Khi nghỉ việc tôi có cần báo trước không?
Người lao động được nghỉ hằng năm bao nhiêu ngày?
Khi nào công ty được đơn phương chấm dứt hợp đồng?
```

### 4.2. Đầu ra

Hệ thống trả về:

- Câu trả lời tiếng Việt.
- Citation cụ thể.
- Các đoạn evidence được dùng.
- Mức độ tự tin hoặc cảnh báo.
- Trạng thái từ chối nếu không đủ căn cứ.

Ví dụ:

```json
{
  "answer": "Có. Tiền lương trong thời gian thử việc do hai bên thỏa thuận, nhưng phải tuân theo quy định tại Bộ luật Lao động 2019. [Bộ luật Lao động 2019, Điều 26]",
  "citations": [
    {
      "document": "Bộ luật Lao động 2019",
      "law_number": "45/2019/QH14",
      "article": "Điều 26",
      "clause": null,
      "evidence_text": "..."
    }
  ],
  "confidence": "medium",
  "refusal": false
}
```

### 4.3. Ràng buộc

Hệ thống phải tuân thủ các ràng buộc sau:

1. Không bịa điều/khoản.
2. Không sinh số ngày, tỷ lệ, mức tiền, quyền hoặc nghĩa vụ nếu retrieved context không hỗ trợ.
3. Không trả lời như tư vấn pháp lý cá nhân hóa.
4. Nếu câu hỏi cần nguồn ngoài Bộ luật Lao động 2019, hệ thống phải cảnh báo phạm vi hạn chế.
5. Nếu không có nguồn đủ liên quan, hệ thống phải từ chối trả lời chắc chắn.

---

## 5. Câu hỏi nghiên cứu

Đề tài không mặc định rằng phương pháp nào chắc chắn tốt hơn. Các kết luận chỉ được đưa ra sau thực nghiệm.

### RQ1. Structure-aware chunking có giúp retrieval tốt hơn fixed-size chunking không?

Giả thuyết cần kiểm chứng: vì văn bản luật có cấu trúc Điều/Khoản/Điểm, chunking theo cấu trúc có thể giúp citation rõ hơn và giảm cắt mất ngữ cảnh. Cần đo bằng Recall@k, MRR@k và Citation Accuracy.

### RQ2. BM25, dense retrieval và hybrid retrieval khác nhau thế nào trên câu hỏi pháp luật tiếng Việt?

BM25 có ưu thế khi câu hỏi chứa từ khóa pháp lý hoặc số điều cụ thể. Dense retrieval có thể tốt hơn khi câu hỏi dùng diễn đạt đời thường. Hybrid retrieval cần được kiểm chứng bằng thí nghiệm, không được claim trước.

### RQ3. Reranker có cải thiện thứ hạng evidence đúng không?

Retriever ban đầu có thể lấy top-20 hoặc top-50 chunk. Reranker chấm lại từng cặp câu hỏi–chunk để đưa evidence liên quan lên cao hơn. Cần đo lợi ích so với chi phí độ trễ.

### RQ4. Citation-aware prompt có giảm câu trả lời thiếu căn cứ không?

Cần so sánh prompt có yêu cầu citation và prompt không yêu cầu citation. Đánh giá bằng Citation Accuracy, Citation Coverage và Hallucination Rate.

### RQ5. Hệ thống có từ chối đúng câu hỏi ngoài phạm vi không?

Cần xây tập câu hỏi ngoài phạm vi, ví dụ câu hỏi về mức phạt hành chính, nghị định xử phạt, bảo hiểm xã hội chi tiết hoặc tình huống pháp lý cá nhân cần luật sư.

---

## 6. Kiến trúc hệ thống

Pipeline tổng thể:

```text
Câu hỏi tiếng Việt
        ↓
Tiền xử lý nhẹ
        ↓
Query Analyzer
        ↓
Retrieval
  ├── BM25
  ├── Dense Retrieval
  └── Hybrid Retrieval
        ↓
Reranker
        ↓
Context Builder
        ↓
Generator với citation prompt
        ↓
Citation Checker + Faithfulness Checker
        ↓
Refusal Module
        ↓
Câu trả lời + trích dẫn + evidence
```

---

## 7. Thành phần 1: Thu thập và chuẩn hóa dữ liệu

### 7.1. Nguồn dữ liệu

Nguồn ưu tiên:

- Cổng Thông tin điện tử Chính phủ: https://vanban.chinhphu.vn/?docid=198540&pageid=27160
- Cơ sở dữ liệu quốc gia về văn bản pháp luật nếu có bản thuận tiện để trích xuất.
- Không ưu tiên nguồn blog hoặc trang tổng hợp nếu đã có nguồn chính thức.

### 7.2. Các bước xử lý

```text
1. Tải văn bản gốc.
2. Trích xuất text từ PDF/HTML/DOC.
3. Chuẩn hóa Unicode.
4. Chuẩn hóa xuống dòng, khoảng trắng, ký tự đặc biệt.
5. Tách Chương, Mục, Điều, Khoản, Điểm.
6. Tạo metadata cho từng chunk.
7. Lưu corpus dạng JSONL.
```

### 7.3. Schema dữ liệu

```json
{
  "chunk_id": "bll2019_dieu_26",
  "document_title": "Bộ luật Lao động 2019",
  "law_number": "45/2019/QH14",
  "effective_date": "2021-01-01",
  "chapter": "Chương III",
  "section": "Mục 1",
  "article": "Điều 26",
  "clause": null,
  "point": null,
  "title": "Tiền lương thử việc",
  "text": "...",
  "source_url": "https://vanban.chinhphu.vn/?docid=198540&pageid=27160",
  "source_type": "official"
}
```

### 7.4. Kiểm tra chất lượng corpus

Cần kiểm tra thủ công tối thiểu:

- 20 điều đầu tiên.
- 20 điều ngẫu nhiên ở giữa.
- 20 điều cuối.
- Các điều có nhiều khoản/điểm.
- Các điều có tiêu đề dài hoặc xuống dòng phức tạp.

Mục tiêu là đảm bảo parser không làm mất tiêu đề điều, không ghép sai điều và không cắt mất khoản.

---

## 8. Thành phần 2: Chunking

### 8.1. Fixed-size chunking

Đây là baseline bắt buộc. Văn bản được cắt theo độ dài cố định, ví dụ 300–700 tokens, có overlap.

Ưu điểm:

- Dễ triển khai.
- Phù hợp với nhiều dạng tài liệu không có cấu trúc rõ.
- Là baseline tốt để so sánh.

Nhược điểm:

- Có thể cắt ngang Điều/Khoản.
- Citation khó chính xác.
- Có thể làm mất quan hệ giữa tiêu đề điều và nội dung điều.

### 8.2. Structure-aware chunking

Văn bản được cắt theo cấu trúc pháp luật:

```text
Văn bản
→ Chương
→ Mục
→ Điều
→ Khoản
→ Điểm
```

Quy tắc đề xuất:

```text
- Nếu Điều ngắn: một Điều = một chunk.
- Nếu Điều dài: chia theo Khoản.
- Nếu Khoản quá dài: chia theo Điểm.
- Mỗi chunk luôn giữ metadata cha: Chương, Mục, Điều.
- Không cắt ngang giữa tiêu đề điều và nội dung điều.
```

### 8.3. Mục tiêu đánh giá chunking

Không đánh giá chunking bằng cảm tính. Cần đo:

- Parser accuracy trên mẫu kiểm tra thủ công.
- Recall@k của retrieval.
- Citation Accuracy.
- Tỷ lệ câu trả lời cần nhiều chunk.
- Độ dài context trung bình.

---

## 9. Thành phần 3: Tiền xử lý câu hỏi

Tiền xử lý phải nhẹ và bảo toàn nghĩa.

Nên làm:

```text
- Chuẩn hóa Unicode NFC.
- Chuẩn hóa khoảng trắng.
- Giữ nguyên dấu tiếng Việt.
- Chuyển về lowercase cho retrieval nếu cần.
- Chuẩn hóa một số viết tắt chắc chắn:
  + cty → công ty
  + hđlđ → hợp đồng lao động
  + nlđ → người lao động
  + nsdlđ → người sử dụng lao động
```

Không nên làm ở bản đầu:

```text
- Không tự động bỏ dấu toàn bộ.
- Không rewrite câu hỏi bằng LLM nếu chưa log và kiểm chứng.
- Không tự động biến câu hỏi thành câu pháp lý quá trang trọng.
- Không suy diễn thêm dữ kiện người dùng chưa cung cấp.
```

---

## 10. Thành phần 4: Retrieval

### 10.1. BM25

BM25 là retriever dựa trên từ khóa. Nó phù hợp khi câu hỏi chứa cụm từ xuất hiện trực tiếp trong văn bản như “hợp đồng lao động”, “thử việc”, “nghỉ hằng năm”, “làm thêm giờ”.

Nguồn nền tảng: Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009.  
https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf

### 10.2. Dense Retrieval

Dense retrieval biểu diễn câu hỏi và chunk bằng vector embedding, sau đó dùng cosine similarity hoặc inner product để tìm các chunk gần nhất.

Ứng viên embedding:

- `BAAI/bge-m3`
- `intfloat/multilingual-e5-base`
- các model sentence-transformers hỗ trợ đa ngữ

BGE-M3 được mô tả là hỗ trợ hơn 100 ngôn ngữ và nhiều chức năng retrieval như dense, sparse và multi-vector retrieval. Đây là lý do hợp lý để đưa vào thí nghiệm tiếng Việt, nhưng hiệu quả trên corpus Bộ luật Lao động vẫn phải được đo thực nghiệm.

Nguồn: Jianlv Chen et al., *M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings*, Findings of ACL 2024.  
https://aclanthology.org/2024.findings-acl.137/

### 10.3. Hybrid Retrieval

Hybrid retrieval kết hợp BM25 và dense retrieval.

Cách đơn giản: Reciprocal Rank Fusion.

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

Trong đó:

- `d` là chunk.
- `rank_i(d)` là thứ hạng của chunk `d` trong retriever thứ `i`.
- `k` thường đặt khoảng 60 theo thực nghiệm gốc, nhưng có thể tune.

Nguồn: Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*, SIGIR 2009.  
https://dl.acm.org/doi/10.1145/1571941.1572114

### 10.4. Chỉ số retrieval

Cần đo:

```text
Recall@1, Recall@3, Recall@5
MRR@5
nDCG@5 nếu có nhiều nguồn đúng
Top-k latency
```

---

## 11. Thành phần 5: Reranker

Retriever đầu tiên lấy top-20 hoặc top-50 chunk. Reranker chấm lại từng cặp:

```text
(question, chunk) → relevance_score
```

Mục tiêu:

- Đưa evidence đúng lên top đầu.
- Giảm chunk nhiễu trước khi đưa vào generator.
- Cải thiện chất lượng context.

Ứng viên:

- Cross-encoder reranker đa ngữ.
- BGE reranker nếu tài nguyên cho phép.
- API reranker nếu được phép sử dụng.

Không nên claim reranker luôn tốt hơn. Cần đo:

- MRR trước và sau rerank.
- Recall@k trước và sau rerank.
- Latency tăng thêm.
- Chi phí inference.

---

## 12. Thành phần 6: Context Builder

Context Builder quyết định đưa chunk nào vào prompt.

Quy tắc đề xuất:

```text
- Lấy top 3–5 chunk sau reranking.
- Mỗi chunk phải có metadata citation.
- Nếu nhiều chunk cùng một Điều, có thể gộp.
- Nếu score thấp hơn ngưỡng, chuyển sang refusal.
- Không đưa quá nhiều chunk nhiễu vào prompt.
```

Format context:

```text
[Nguồn 1]
Văn bản: Bộ luật Lao động 2019
Số hiệu: 45/2019/QH14
Điều: Điều 26
Tiêu đề: Tiền lương thử việc
Nội dung:
...

[Nguồn 2]
Văn bản: Bộ luật Lao động 2019
Số hiệu: 45/2019/QH14
Điều: Điều 27
Tiêu đề: Kết thúc thời gian thử việc
Nội dung:
...
```

---

## 13. Thành phần 7: Generator có citation prompt

Generator có thể là:

- Gemini Flash.
- Qwen.
- Llama.
- GPT API nếu được phép.
- Một LLM local nếu tài nguyên đủ.

Đề tài không bắt buộc fine-tune LLM. Trọng tâm là thiết kế RAG pipeline và đánh giá.

Prompt hệ thống:

```text
Bạn là trợ lý tra cứu Bộ luật Lao động Việt Nam.
Chỉ trả lời dựa trên các nguồn được cung cấp.
Không dùng kiến thức ngoài context.
Mỗi ý quan trọng phải kèm citation dạng [Bộ luật Lao động 2019, Điều X, Khoản Y].
Nếu nguồn không đủ thông tin, trả lời: "Tôi không tìm thấy căn cứ đủ rõ trong Bộ luật Lao động 2019."
Không đưa ra tư vấn pháp lý cá nhân hóa.
```

Format đầu ra mong muốn:

```json
{
  "answer": "...",
  "citations": [
    {
      "document": "Bộ luật Lao động 2019",
      "article": "Điều 26",
      "clause": null,
      "evidence_text": "..."
    }
  ],
  "confidence": "medium",
  "refusal": false
}
```

---

## 14. Thành phần 8: Citation Checker

Citation Checker không dựa hoàn toàn vào LLM. Nó kiểm tra bằng metadata.

Các kiểm tra bắt buộc:

```text
1. Citation có đúng format không?
2. Điều/Khoản được trích có tồn tại trong corpus không?
3. Điều/Khoản đó có nằm trong retrieved context không?
4. Citation có gắn với claim chính không?
5. Có citation bịa không?
```

Các lỗi cần log:

```text
- missing_citation
- malformed_citation
- citation_not_found_in_corpus
- citation_not_in_retrieved_context
- unsupported_claim
```

---

## 15. Thành phần 9: Faithfulness Checker

Faithfulness Checker kiểm tra xem câu trả lời có bám evidence không.

Có hai cách triển khai:

### 15.1. Rule-based nhẹ

Áp dụng cho các thông tin dễ kiểm tra:

- Số ngày.
- Phần trăm.
- Tên điều.
- Số điều/khoản.
- Các cụm nghĩa vụ/quyền được copy từ evidence.

### 15.2. LLM-as-judge có kiểm soát

Dùng LLM đánh giá từng claim so với evidence. Cần yêu cầu evaluator trả về:

```json
{
  "claim": "...",
  "supported": true,
  "evidence": "...",
  "reason": "..."
}
```

Không nên chỉ dùng LLM-as-judge làm kết quả cuối. Nên kết hợp với đánh giá thủ công trên một tập nhỏ.

RAGAS là framework đánh giá RAG có thể tham khảo vì nó đưa ra các metric cho retrieval và generation trong hệ thống RAG. Tuy nhiên, với domain pháp luật, vẫn cần ground-truth thủ công để đảm bảo độ tin cậy.

Nguồn: Shahul Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation*, EACL Demo 2024.  
https://aclanthology.org/2024.eacl-demo.16/

---

## 16. Thành phần 10: Refusal Module

Hệ thống phải biết từ chối khi không đủ căn cứ.

Các trường hợp cần từ chối/cảnh báo:

```text
- Câu hỏi ngoài phạm vi Bộ luật Lao động 2019.
- Câu hỏi cần nghị định/thông tư hướng dẫn.
- Câu hỏi yêu cầu mức phạt hành chính cụ thể.
- Câu hỏi cần kết luận pháp lý cho vụ việc cá nhân.
- Retriever không tìm được chunk đủ liên quan.
- Evidence mâu thuẫn hoặc không đủ rõ.
```

Ví dụ:

```text
Câu hỏi:
Công ty không đóng bảo hiểm cho tôi thì bị phạt bao nhiêu tiền?

Trả lời:
Tôi không tìm thấy căn cứ đủ rõ trong corpus hiện tại để kết luận mức phạt cụ thể. Corpus hiện chỉ gồm Bộ luật Lao động 2019, trong khi mức phạt hành chính thường cần tra cứu văn bản xử phạt hoặc văn bản chuyên ngành liên quan.
```

---

## 17. Benchmark đánh giá

### 17.1. Quy mô

Bản tối thiểu:

```text
100–150 câu hỏi
```

Bản tốt:

```text
200–300 câu hỏi
```

### 17.2. Nhóm câu hỏi

```text
1. Single-article QA:
   Trả lời từ một điều.

2. Multi-clause QA:
   Cần nhiều khoản trong cùng một điều.

3. Multi-article QA nhẹ:
   Cần kết hợp 2–3 điều trong cùng Bộ luật.

4. Summarization:
   Tóm tắt một nhóm quy định.

5. Out-of-scope/refusal:
   Câu hỏi ngoài phạm vi corpus.

6. Adversarial:
   Câu hỏi gài bẫy, giả định sai hoặc yêu cầu kết luận vượt nguồn.
```

### 17.3. Schema benchmark

```json
{
  "question_id": "q001",
  "question": "Thử việc có được trả lương không?",
  "question_type": "single_article",
  "gold_sources": [
    {
      "document": "Bộ luật Lao động 2019",
      "article": "Điều 26",
      "clause": null
    }
  ],
  "gold_answer_short": "...",
  "must_refuse": false,
  "notes": "Câu hỏi phổ biến về thử việc."
}
```

---

## 18. Metrics

### 18.1. Retrieval metrics

```text
Recall@k:
Gold source có nằm trong top-k không?

MRR@k:
Gold source đúng xuất hiện ở vị trí thứ mấy?

nDCG@k:
Ranking có đưa nguồn quan trọng lên cao không?
```

### 18.2. Citation metrics

```text
Citation Accuracy:
Citation sinh ra có đúng điều/khoản không?

Citation Coverage:
Bao nhiêu claim chính có citation?

Unsupported Citation Rate:
Tỷ lệ citation không nằm trong retrieved context.

Fabricated Citation Rate:
Tỷ lệ citation không tồn tại trong corpus.
```

### 18.3. Answer quality metrics

```text
Answer Correctness:
Câu trả lời có đúng theo gold answer không?

Faithfulness:
Câu trả lời có bám evidence không?

Hallucination Rate:
Tỷ lệ câu trả lời có claim không có trong nguồn.

Refusal Accuracy:
Câu hỏi ngoài phạm vi có bị từ chối đúng không?
```

### 18.4. System metrics

```text
Latency p50/p90.
Số token context trung bình.
Chi phí API trung bình trên mỗi câu hỏi.
Tỷ lệ lỗi parser/chunker.
Tỷ lệ lỗi JSON output.
```

---

## 19. Thiết kế thí nghiệm

| ID | Chunking | Retrieval | Rerank | Citation prompt | Refusal | Mục tiêu |
|---|---|---|---|---|---|---|
| A | Fixed-size | BM25 | Không | Có | Không | Baseline từ khóa |
| B | Fixed-size | Dense | Không | Có | Không | Baseline semantic |
| C | Structure-aware | BM25 | Không | Có | Không | Ảnh hưởng chunking |
| D | Structure-aware | Hybrid | Không | Có | Không | Ảnh hưởng hybrid retrieval |
| E | Structure-aware | Hybrid | Có | Có | Không | Ảnh hưởng reranker |
| F | Structure-aware | Hybrid | Có | Không | Không | Ảnh hưởng citation prompt |
| G | Structure-aware | Hybrid | Có | Có | Có | Hệ thống đầy đủ |

Kết quả cần báo cáo theo từng nhóm câu hỏi, không chỉ báo cáo trung bình toàn bộ.

---

## 20. Phân tích lỗi

Taxonomy lỗi:

```text
1. Parser error:
   Tách sai Điều/Khoản/Điểm.

2. Retrieval miss:
   Gold source không nằm trong top-k.

3. Reranking error:
   Gold source có trong candidates nhưng bị đẩy xuống thấp.

4. Context noise:
   Context đưa vào generator chứa quá nhiều chunk nhiễu.

5. Generation hallucination:
   LLM thêm thông tin ngoài nguồn.

6. Citation fabrication:
   LLM tạo citation không tồn tại.

7. Refusal error:
   Hệ thống trả lời khi đáng lẽ phải từ chối, hoặc từ chối khi có đủ nguồn.

8. Ambiguous question:
   Câu hỏi không đủ rõ, cần hỏi lại người dùng.
```

Trong báo cáo, mỗi nhóm lỗi nên có ví dụ cụ thể.

---

## 21. Công nghệ đề xuất

Stack tối thiểu:

```text
Python
pypdf / BeautifulSoup / python-docx
regex parser cho Điều/Khoản/Điểm
rank-bm25 hoặc Pyserini
sentence-transformers
FAISS hoặc ChromaDB
BGE-M3 hoặc multilingual-e5
cross-encoder reranker nếu tài nguyên cho phép
Gemini/Qwen/Llama/GPT API cho generation
FastAPI cho backend
Streamlit hoặc Gradio cho demo
RAGAS + evaluator thủ công cho đánh giá
```

Không cần fine-tune trong bản đầu. Fine-tune có thể để hướng phát triển.

---

## 22. Giao diện demo

Giao diện nên có:

```text
- Ô nhập câu hỏi.
- Nút hỏi.
- Câu trả lời.
- Danh sách nguồn trích dẫn.
- Đoạn evidence gốc.
- Confidence/refusal warning.
- Log top-k retrieved chunks cho chế độ debug.
```

Câu hỏi demo nên chuẩn bị sẵn:

```text
- Thử việc tối đa bao lâu?
- Lương thử việc được tính như thế nào?
- Người lao động nghỉ việc có cần báo trước không?
- Công ty có được đơn phương chấm dứt hợp đồng không?
- Người lao động được nghỉ hằng năm bao nhiêu ngày?
- Khi nào bị xử lý kỷ luật sa thải?
- Làm thêm giờ được quy định như thế nào?
- Người lao động chưa đủ 18 tuổi có được làm việc không?
- Hợp đồng lao động gồm những loại nào?
- Công ty không đóng bảo hiểm thì bị phạt bao nhiêu tiền?
```

Câu cuối là ví dụ tốt để kiểm tra refusal vì có thể cần văn bản ngoài Bộ luật Lao động 2019.

---

## 23. Nguyên tắc an toàn

Hệ thống phải hiển thị rõ:

```text
Hệ thống chỉ hỗ trợ tra cứu thông tin từ Bộ luật Lao động 2019.
Câu trả lời không thay thế tư vấn pháp lý từ luật sư, cơ quan nhà nước hoặc chuyên gia pháp lý.
Nếu câu hỏi cần nghị định, thông tư, văn bản xử phạt hoặc dữ kiện vụ việc cụ thể, hệ thống sẽ cảnh báo phạm vi hạn chế.
```

Không nên trả lời:

```text
Công ty chắc chắn sai.
Bạn chắc chắn được bồi thường X đồng.
Trường hợp của bạn chắc chắn thắng kiện.
```

Nên trả lời:

```text
Theo các nguồn được truy hồi trong Bộ luật Lao động 2019, quy định liên quan là...
Với corpus hiện tại, tôi chưa tìm thấy căn cứ đủ rõ để kết luận...
```

---

## 24. Roadmap triển khai 8 tuần

### Tuần 1: Corpus

- Tải văn bản nguồn.
- Trích xuất text.
- Thiết kế metadata schema.
- Parse thử Chương/Mục/Điều/Khoản.

### Tuần 2: Chunking

- Implement fixed-size chunking.
- Implement structure-aware chunking.
- Kiểm tra thủ công chất lượng parser.
- Lưu corpus JSONL.

### Tuần 3: Retrieval cơ bản

- Implement BM25.
- Implement dense retrieval.
- Tạo FAISS/Chroma index.
- Chạy thử top-k retrieval.

### Tuần 4: Hybrid retrieval và reranking

- Implement RRF.
- Thử reranker.
- So sánh BM25, dense, hybrid.

### Tuần 5: Generation và citation

- Thiết kế prompt.
- Implement generator.
- Chuẩn hóa JSON output.
- Implement citation checker cơ bản.

### Tuần 6: Benchmark

- Tạo 100–150 câu hỏi.
- Gán gold source.
- Thêm nhóm out-of-scope và adversarial.
- Viết script đánh giá retrieval.

### Tuần 7: Evaluation

- Chạy ablation.
- Tính metrics.
- Phân tích lỗi.
- Đo latency và chi phí.

### Tuần 8: Demo và báo cáo

- Xây Streamlit/Gradio demo.
- Viết README.
- Viết báo cáo.
- Chuẩn bị video demo ngắn.

---

## 25. Tiêu chí hoàn thành

### Mức tối thiểu

```text
[ ] Parse được Bộ luật Lao động 2019 thành chunk có metadata.
[ ] Có BM25 retrieval.
[ ] Có dense retrieval.
[ ] Có hybrid retrieval.
[ ] Có generation kèm citation.
[ ] Có refusal khi ngoài phạm vi.
[ ] Có 100 câu hỏi test.
[ ] Có Recall@k, MRR@k, Citation Accuracy.
[ ] Có demo web đơn giản.
```

### Mức tốt

```text
[ ] So sánh fixed-size vs structure-aware chunking.
[ ] Có reranker.
[ ] Có faithfulness/citation checker.
[ ] Có error analysis theo taxonomy.
[ ] Có latency và chi phí API.
[ ] Có README rõ ràng.
```

### Mức rất tốt

```text
[ ] Có 200–300 câu hỏi benchmark.
[ ] Có nhóm adversarial/refusal đủ rõ.
[ ] Có Docker.
[ ] Có FastAPI endpoint.
[ ] Có báo cáo paper-style.
[ ] Có demo video ngắn.
```

---

## 26. Cấu trúc thư mục đề xuất

```text
vilaborrag/
├── data/
│   ├── raw/
│   ├── processed/
│   └── benchmark/
├── src/
│   ├── ingest/
│   ├── chunking/
│   ├── retrieval/
│   ├── reranking/
│   ├── generation/
│   ├── evaluation/
│   └── api/
├── notebooks/
│   ├── 01_parse_law.ipynb
│   ├── 02_retrieval_experiments.ipynb
│   └── 03_evaluation.ipynb
├── app/
│   └── streamlit_app.py
├── configs/
├── tests/
├── README.md
└── report.md
```

---

## 27. Cấu trúc báo cáo đề xuất

```text
1. Introduction
2. Related Work
3. Dataset and Corpus Construction
4. Method
   - Legal-aware parsing
   - Chunking
   - Retrieval
   - Reranking
   - Citation-aware generation
   - Faithfulness/refusal checking
5. Experiments
6. Results
7. Error Analysis
8. Demo System
9. Limitations
10. Conclusion and Future Work
```

---

## 28. Hạn chế

Hạn chế cần nêu rõ trong báo cáo:

```text
- Hệ thống chỉ dùng Bộ luật Lao động 2019.
- Chưa xử lý đầy đủ nghị định, thông tư, văn bản xử phạt.
- Chưa xử lý tự động lịch sử sửa đổi/bãi bỏ ở quy mô lớn.
- Chưa thay thế tư vấn pháp lý.
- Benchmark được xây thủ công nên có giới hạn về quy mô.
- Kết quả LLM-as-judge nếu có chỉ là tham khảo, cần kiểm tra thủ công.
```

---

## 29. Hướng phát triển

```text
- Bổ sung nghị định/thông tư hướng dẫn.
- Xử lý trạng thái hiệu lực và văn bản thay thế.
- Mở rộng sang các bộ luật khác như Luật Bảo hiểm xã hội, Luật Cư trú, Luật Hôn nhân và Gia đình.
- Thêm multi-document retrieval.
- Tự động cập nhật corpus từ nguồn chính thức.
- Fine-tune reranker hoặc embedding cho domain pháp luật tiếng Việt.
- Tích hợp Text-to-SQL cho các bảng dữ liệu tình huống giả lập.
```

---

## 30. Tài liệu tham khảo

1. Quốc hội Việt Nam. **Bộ luật Lao động 2019, Luật số 45/2019/QH14**. Cổng Thông tin điện tử Chính phủ.  
   https://vanban.chinhphu.vn/?docid=198540&pageid=27160

2. Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. NeurIPS 2020.  
   https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html

3. Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih. **Dense Passage Retrieval for Open-Domain Question Answering**. EMNLP 2020.  
   https://aclanthology.org/2020.emnlp-main.550/

4. Stephen Robertson, Hugo Zaragoza. **The Probabilistic Relevance Framework: BM25 and Beyond**. Foundations and Trends in Information Retrieval, 2009.  
   https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf

5. Gordon V. Cormack, Charles L. A. Clarke, Stefan Büttcher. **Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods**. SIGIR 2009.  
   https://dl.acm.org/doi/10.1145/1571941.1572114

6. Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych. **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models**. NeurIPS Datasets and Benchmarks 2021.  
   https://arxiv.org/abs/2104.08663

7. Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, Zheng Liu. **M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings**. Findings of ACL 2024.  
   https://aclanthology.org/2024.findings-acl.137/

8. Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert. **RAGAS: Automated Evaluation of Retrieval Augmented Generation**. EACL Demo 2024.  
   https://aclanthology.org/2024.eacl-demo.16/

9. Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. **Lost in the Middle: How Language Models Use Long Contexts**. TACL 2024.  
   https://aclanthology.org/2024.tacl-1.9/

---

## 31. Kết luận chốt đề tài

Đề tài được chốt là:

**Xây dựng hệ thống hỏi đáp Bộ luật Lao động Việt Nam có trích dẫn nguồn bằng Retrieval-Augmented Generation.**

Đây là đề tài có tính ứng dụng rõ ràng, phù hợp với NLP hiện đại và có thể triển khai trong phạm vi đồ án môn học. Giá trị chính của đề tài không nằm ở việc gọi một LLM để trả lời, mà nằm ở toàn bộ pipeline: chuẩn hóa văn bản luật, chunking theo cấu trúc, retrieval, reranking, citation-aware generation, kiểm tra faithfulness, refusal và đánh giá có hệ thống.

Bản đầu chỉ nên tập trung vào một bộ luật để đảm bảo khả thi. Các phần mở rộng như nghị định/thông tư, xử lý hiệu lực văn bản và multi-document legal reasoning nên để hướng phát triển.
