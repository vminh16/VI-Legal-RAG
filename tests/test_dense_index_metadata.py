from src.retrieval.dense_retriever import build_index_metadata, is_index_metadata_current


def test_index_metadata_detects_corpus_model_and_dimension_changes():
    chunks = [
        {"chunk_id": "c1", "text": "Điều 1. Nội dung một"},
        {"chunk_id": "c2", "text": "Điều 2. Nội dung hai"},
    ]

    metadata = build_index_metadata(chunks, model_name="test-model", embedding_dimension=3)

    assert metadata["chunk_count"] == 2
    assert metadata["model_name"] == "test-model"
    assert metadata["embedding_dimension"] == 3
    assert is_index_metadata_current(metadata, chunks, "test-model", 3) is True

    changed_chunks = [
        {"chunk_id": "c1", "text": "Điều 1. Nội dung đã đổi"},
        {"chunk_id": "c2", "text": "Điều 2. Nội dung hai"},
    ]

    assert is_index_metadata_current(metadata, changed_chunks, "test-model", 3) is False
    assert is_index_metadata_current(metadata, chunks, "other-model", 3) is False
    assert is_index_metadata_current(metadata, chunks, "test-model", 4) is False
