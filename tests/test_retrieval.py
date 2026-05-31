import pytest
import unicodedata
from src.retrieval.query_preprocessor import QueryPreprocessor

def test_query_preprocessor_initialization():
    """Kiểm tra khởi tạo bộ tiền xử lý câu hỏi."""
    preprocessor = QueryPreprocessor()
    assert isinstance(preprocessor, QueryPreprocessor)

def test_query_preprocessor_basic_cleaning():
    """Kiểm tra làm sạch khoảng trắng thừa và lowercase câu hỏi."""
    preprocessor = QueryPreprocessor()
    
    # 1. Kiểm tra xóa khoảng trắng thừa ở đầu, cuối và giữa các từ
    query = "   Người   lao    động   "
    assert preprocessor.preprocess(query) == "người lao động"
    
    # 2. Kiểm tra chuyển về lowercase
    query = "HỢP ĐỒNG LAO ĐỘNG"
    assert preprocessor.preprocess(query) == "hợp đồng lao động"

def test_query_preprocessor_unicode_normalization():
    """Kiểm tra chuẩn hóa Unicode NFC."""
    preprocessor = QueryPreprocessor()
    
    # Dùng ký tự tổ hợp (Decomposed - NFD)
    # Từ "người" dạng tổ hợp: n-g-u-móc-o-móc-huyền-i
    decomposed_word = "ngu" + "\u031b" + "o" + "\u031b" + "\u0300" + "i"
    assert unicodedata.is_normalized("NFC", decomposed_word) is False
    
    processed = preprocessor.preprocess(decomposed_word)
    assert unicodedata.is_normalized("NFC", processed) is True
    assert processed == "người"

def test_query_preprocessor_abbreviation_expansion():
    """Kiểm tra dịch từ viết tắt sang từ đầy đủ."""
    preprocessor = QueryPreprocessor()
    
    # 1. Các viết tắt cơ bản
    assert preprocessor.preprocess("cty tuyển dụng") == "công ty tuyển dụng"
    assert preprocessor.preprocess("ký hđlđ mới") == "ký hợp đồng lao động mới"
    assert preprocessor.preprocess("quyền của nlđ") == "quyền của người lao động"
    
    assert preprocessor.preprocess("nghĩa vụ của nsdlđ") == "nghĩa vụ của người sử dụng lao động"
    assert preprocessor.preprocess("tra cứu bllđ 2019") == "tra cứu bộ luật lao động 2019"
    
    # 2. Viết hoa hỗn hợp
    assert preprocessor.preprocess("Quyền của NLĐ") == "quyền của người lao động"
    assert preprocessor.preprocess("NSDLĐ và CTY") == "người sử dụng lao động và công ty"
    
    # 3. Tránh thay thế nhầm các từ dính chữ khác (ranh giới từ)
    # Ví dụ: "hđlđs" không được chuyển thành "hợp đồng lao độngs"
    assert preprocessor.preprocess("hđlđs") == "hđlđs"
    assert preprocessor.preprocess("cty_moi") == "cty_moi"

def test_bm25_retriever_initialization_and_searching():
    """Kiểm tra khởi tạo và tìm kiếm với BM25Retriever."""
    from src.retrieval.bm25_retriever import BM25Retriever
    
    sample_chunks = [
        {
            "chunk_id": "bll2019_dieu_1",
            "document_title": "Bộ luật Lao động 2019",
            "law_number": "45/2019/QH14",
            "chapter": "Chương I",
            "section": None,
            "article": "Điều 1",
            "clause": None,
            "point": None,
            "title": "Phạm vi điều chỉnh",
            "text": "Bộ luật Lao động quy định tiêu chuẩn lao động; quyền, nghĩa vụ, trách nhiệm của người lao động.",
            "source_url": "https://test.url"
        },
        {
            "chunk_id": "bll2019_dieu_2",
            "document_title": "Bộ luật Lao động 2019",
            "law_number": "45/2019/QH14",
            "chapter": "Chương I",
            "section": None,
            "article": "Điều 2",
            "clause": None,
            "point": None,
            "title": "Đối tượng áp dụng",
            "text": "Người lao động làm việc tại Việt Nam và các cơ quan tổ chức cá nhân có liên quan.",
            "source_url": "https://test.url"
        },
        {
            "chunk_id": "bll2019_dieu_3",
            "document_title": "Bộ luật Lao động 2019",
            "law_number": "45/2019/QH14",
            "chapter": "Chương I",
            "section": None,
            "article": "Điều 3",
            "clause": None,
            "point": None,
            "title": "Giải thích từ ngữ",
            "text": "Doanh nghiệp tổ chức công đoàn cơ sở đại diện tập thể người lao động.",
            "source_url": "https://test.url"
        },
        {
            "chunk_id": "bll2019_dieu_4",
            "document_title": "Bộ luật Lao động 2019",
            "law_number": "45/2019/QH14",
            "chapter": "Chương I",
            "section": None,
            "article": "Điều 4",
            "clause": None,
            "point": None,
            "title": "Chính sách của Nhà nước",
            "text": "Nhà nước bảo đảm quyền và lợi ích hợp pháp của người lao động người sử dụng lao động.",
            "source_url": "https://test.url"
        },
        {
            "chunk_id": "bll2019_dieu_24_khoan_1",
            "document_title": "Bộ luật Lao động 2019",
            "law_number": "45/2019/QH14",
            "chapter": "Chương III",
            "section": "Mục 1",
            "article": "Điều 24",
            "clause": "1",
            "point": None,
            "title": "Thử việc",
            "text": "Người sử dụng lao động và người lao động có thỏa thuận về việc thử việc, quyền và nghĩa vụ của hai bên trong thời gian thử việc.",
            "source_url": "https://test.url"
        }
    ]
    
    # Khởi tạo retriever với danh sách chunk giả lập
    retriever = BM25Retriever(chunks=sample_chunks)
    
    # 1. Tìm kiếm với từ khóa "thử việc"
    results = retriever.retrieve(query="thử việc", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "bll2019_dieu_24_khoan_1"
    assert results[0]["article"] == "Điều 24"
    assert "score" in results[0]
    assert results[0]["score"] > 0
    
    # 2. Tìm kiếm với từ viết tắt "hđlđ" (retriever tự động đi qua QueryPreprocessor)
    results_abbr = retriever.retrieve(query="hđlđ", top_k=2)
    assert len(results_abbr) > 0
    assert "score" in results_abbr[0]

def test_dense_retriever_initialization_and_searching(tmp_path):
    """Kiểm tra khởi tạo, lưu/tải index FAISS và tìm kiếm với DenseRetriever."""
    from src.retrieval.dense_retriever import DenseRetriever
    
    sample_chunks = [
        {
            "chunk_id": "bll2019_dieu_1",
            "document_title": "Bộ luật Lao động 2019",
            "law_number": "45/2019/QH14",
            "chapter": "Chương I",
            "section": None,
            "article": "Điều 1",
            "clause": None,
            "point": None,
            "title": "Phạm vi điều chỉnh",
            "text": "Bộ luật Lao động quy định tiêu chuẩn lao động; quyền, nghĩa vụ, trách nhiệm của người lao động.",
            "source_url": "https://test.url"
        },
        {
            "chunk_id": "bll2019_dieu_2",
            "document_title": "Bộ luật Lao động 2019",
            "law_number": "45/2019/QH14",
            "chapter": "Chương I",
            "section": None,
            "article": "Điều 2",
            "clause": None,
            "point": None,
            "title": "Đối tượng áp dụng",
            "text": "Người lao động làm việc tại Việt Nam và các cơ quan tổ chức cá nhân có liên quan.",
            "source_url": "https://test.url"
        },
        {
            "chunk_id": "bll2019_dieu_24_khoan_1",
            "document_title": "Bộ luật Lao động 2019",
            "law_number": "45/2019/QH14",
            "chapter": "Chương III",
            "section": "Mục 1",
            "article": "Điều 24",
            "clause": "1",
            "point": None,
            "title": "Thử việc",
            "text": "Người sử dụng lao động và người lao động có thỏa thuận về việc thử việc, quyền và nghĩa vụ của hai bên trong thời gian thử việc.",
            "source_url": "https://test.url"
        }
    ]
    
    # Dùng một mô hình siêu nhẹ cho bài unit test để chạy nhanh trên CPU (không cần tải bge-m3 2.2GB trong lúc test)
    test_model = "sentence-transformers/all-MiniLM-L6-v2"
    
    # 1. Khởi tạo retriever mới (lập chỉ mục từ đầu và lưu xuống thư mục tạm tmp_path)
    retriever = DenseRetriever(
        chunks=sample_chunks,
        model_name=test_model,
        index_dir=tmp_path,
        force_rebuild=True
    )
    
    # Kiểm tra xem các file cache index có được lưu xuống đĩa không
    index_file = tmp_path / "faiss_index.bin"
    mapping_file = tmp_path / "faiss_mapping.json"
    assert index_file.exists()
    assert mapping_file.exists()
    
    # 2. Tìm kiếm ngữ nghĩa
    results = retriever.retrieve(query="công việc thử nghiệm trước khi nhận chính thức", top_k=1)
    assert len(results) == 1
    # Kỳ vọng từ "thử nghiệm" ngữ nghĩa sẽ khớp nhất với "thử việc" (Điều 24)
    assert results[0]["chunk_id"] == "bll2019_dieu_24_khoan_1"
    assert "score" in results[0]
    assert results[0]["score"] > 0
    
    # 3. Khởi tạo một retriever thứ hai chỉ định load từ cache đĩa đã có sẵn (không rebuild)
    retriever_cached = DenseRetriever(
        chunks=sample_chunks,
        model_name=test_model,
        index_dir=tmp_path,
        force_rebuild=False
    )
    
    # Thực hiện tìm kiếm và so sánh kết quả
    results_cached = retriever_cached.retrieve(query="công việc thử nghiệm trước khi nhận chính thức", top_k=1)
    assert len(results_cached) == 1
    assert results_cached[0]["chunk_id"] == "bll2019_dieu_24_khoan_1"
    assert results_cached[0]["score"] == results[0]["score"]

def test_hybrid_retriever_rrf():
    """Kiểm tra tính đúng đắn của giải thuật Reciprocal Rank Fusion (RRF)."""
    from src.retrieval.hybrid_retriever import HybridRetriever
    
    # Kết quả giả lập từ BM25 (chứa các chunk và score tương ứng)
    bm25_results = [
        {"chunk_id": "chunk_A", "text": "văn bản A"},
        {"chunk_id": "chunk_B", "text": "văn bản B"}
    ]
    
    # Kết quả giả lập từ Dense (thứ tự đảo ngược)
    dense_results = [
        {"chunk_id": "chunk_B", "text": "văn bản B"},
        {"chunk_id": "chunk_A", "text": "văn bản A"}
    ]
    
    hybrid = HybridRetriever(k_rrf=60)
    # Gộp kết quả
    results = hybrid.fuse(bm25_results, dense_results, top_k=2)
    
    assert len(results) == 2
    # Vì chunk_A hạng 1 ở BM25, hạng 2 ở Dense -> RRF score = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.01639 + 0.01613 = 0.03252
    # chunk_B cũng có thứ hạng tương đương (hạng 2 ở BM25, hạng 1 ở Dense) -> RRF score = 0.03252
    assert "score" in results[0]
    assert results[0]["score"] > 0

def test_reranker_cross_encoder():
    """Kiểm tra hoạt động tái xếp hạng của Reranker Cross-Encoder."""
    from src.retrieval.reranker import Reranker
    
    sample_chunks = [
        {"chunk_id": "chunk_1", "text": "Người lao động được quyền đơn phương chấm dứt hợp đồng lao động bất kỳ lúc nào."},
        {"chunk_id": "chunk_2", "text": "Thời gian thử việc đối với công nhân kỹ thuật tối đa không quá 30 ngày."}
    ]
    
    # Dùng mô hình CrossEncoder siêu nhẹ cho unit test
    test_reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker = Reranker(model_name=test_reranker_model)
    
    # Tái xếp hạng câu hỏi về "thử việc"
    reranked = reranker.rerank(
        query="thử việc công nhân kỹ thuật",
        chunks=sample_chunks,
        top_k=2
    )
    
    assert len(reranked) == 2
    # Kỳ vọng chunk_2 sẽ được đẩy lên số 1 vì chứa thông tin thử việc công nhân sát nghĩa nhất
    assert reranked[0]["chunk_id"] == "chunk_2"
    assert "score" in reranked[0]

def test_retrieval_pipeline_integration(tmp_path):
    """Kiểm tra pipeline tích hợp RetrievalPipeline chạy 4 chiến lược tìm kiếm."""
    from src.retrieval.retrieval_pipeline import RetrievalPipeline
    
    sample_chunks = [
        {
            "chunk_id": "bll2019_dieu_1",
            "document_title": "Bộ luật Lao động 2019",
            "law_number": "45/2019/QH14",
            "chapter": "Chương I",
            "section": None,
            "article": "Điều 1",
            "clause": None,
            "point": None,
            "title": "Phạm vi điều chỉnh",
            "text": "Bộ luật Lao động quy định tiêu chuẩn lao động; quyền, nghĩa vụ, trách nhiệm của người lao động.",
            "source_url": "https://test.url"
        },
        {
            "chunk_id": "bll2019_dieu_2",
            "document_title": "Bộ luật Lao động 2019",
            "law_number": "45/2019/QH14",
            "chapter": "Chương I",
            "section": None,
            "article": "Điều 2",
            "clause": None,
            "point": None,
            "title": "Đối tượng áp dụng",
            "text": "Người lao động làm việc tại Việt Nam và các cơ quan tổ chức cá nhân có liên quan.",
            "source_url": "https://test.url"
        },
        {
            "chunk_id": "bll2019_dieu_24_khoan_1",
            "document_title": "Bộ luật Lao động 2019",
            "law_number": "45/2019/QH14",
            "chapter": "Chương III",
            "section": "Mục 1",
            "article": "Điều 24",
            "clause": "1",
            "point": None,
            "title": "Thử việc",
            "text": "Người sử dụng lao động và người lao động có thỏa thuận về việc thử việc, quyền và nghĩa vụ của hai bên trong thời gian thử việc.",
            "source_url": "https://test.url"
        }
    ]
    
    # Khởi tạo pipeline
    pipeline = RetrievalPipeline(
        chunks=sample_chunks,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        index_dir=tmp_path,
        force_rebuild=True
    )
    
    # 1. Test chiến lược BM25
    res_bm25 = pipeline.retrieve("thử việc", strategy="bm25", top_k=1)
    assert len(res_bm25) == 1
    assert res_bm25[0]["chunk_id"] == "bll2019_dieu_24_khoan_1"
    
    # 2. Test chiến lược Dense
    res_dense = pipeline.retrieve("thử việc", strategy="dense", top_k=1)
    assert len(res_dense) == 1
    assert res_dense[0]["chunk_id"] == "bll2019_dieu_24_khoan_1"
    
    # 3. Test chiến lược Hybrid (RRF)
    res_hybrid = pipeline.retrieve("thử việc", strategy="hybrid", top_k=1)
    assert len(res_hybrid) == 1
    assert res_hybrid[0]["chunk_id"] == "bll2019_dieu_24_khoan_1"
    
    # 4. Test chiến lược Hybrid + Rerank
    res_rerank = pipeline.retrieve("thử việc", strategy="hybrid_rerank", top_k=1)
    assert len(res_rerank) == 1
    assert res_rerank[0]["chunk_id"] == "bll2019_dieu_24_khoan_1"



