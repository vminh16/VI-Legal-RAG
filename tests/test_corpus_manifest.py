import hashlib

from src.ingest.corpus_manifest import build_corpus_manifest


def test_build_corpus_manifest_records_source_hashes_and_chunk_summary(tmp_path):
    raw_text = tmp_path / "law.txt"
    raw_text.write_text("Bộ luật Lao động 2019", encoding="utf-8")
    chunks = [
        {
            "chunk_id": "bll2019_dieu_1",
            "law_number": "45/2019/QH14",
            "article": "Điều 1",
            "text": "Điều 1. Phạm vi điều chỉnh",
            "source_url": "https://vanban.chinhphu.vn/?docid=198540&pageid=27160",
        }
    ]

    manifest = build_corpus_manifest(
        chunks=chunks,
        source_files={"raw_text": raw_text},
        parser_version="structure_chunker_v1",
    )

    expected_hash = hashlib.sha256(raw_text.read_bytes()).hexdigest()
    assert manifest["parser_version"] == "structure_chunker_v1"
    assert manifest["chunk_count"] == 1
    assert manifest["article_count"] == 1
    assert manifest["law_numbers"] == ["45/2019/QH14"]
    assert manifest["source_files"]["raw_text"]["sha256"] == expected_hash
    assert manifest["source_urls"] == ["https://vanban.chinhphu.vn/?docid=198540&pageid=27160"]
