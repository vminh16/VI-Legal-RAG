import hashlib
from datetime import datetime, timezone
from pathlib import Path


def _source_file_info(path: str | Path) -> dict:
    source_path = Path(path)
    content = source_path.read_bytes()
    return {
        "path": str(source_path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def build_corpus_manifest(
    chunks: list[dict],
    source_files: dict[str, str | Path],
    parser_version: str,
) -> dict:
    """Tạo manifest audit cho corpus đã chunk, gồm hash nguồn và thống kê pháp lý."""
    law_numbers = sorted({chunk.get("law_number") for chunk in chunks if chunk.get("law_number")})
    articles = sorted({chunk.get("article") for chunk in chunks if chunk.get("article")})
    source_urls = sorted({chunk.get("source_url") for chunk in chunks if chunk.get("source_url")})

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parser_version": parser_version,
        "chunk_count": len(chunks),
        "article_count": len(articles),
        "law_numbers": law_numbers,
        "source_urls": source_urls,
        "source_files": {
            name: _source_file_info(path)
            for name, path in source_files.items()
        },
    }
