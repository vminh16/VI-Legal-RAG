from pathlib import Path


APP_JS = Path("app/static/app.js")


def test_frontend_defines_html_and_url_escaping_helpers():
    """Frontend phải có helper escape dữ liệu động trước khi đưa vào innerHTML."""
    source = APP_JS.read_text(encoding="utf-8")

    assert "function escapeHtml" in source
    assert "function safeExternalUrl" in source


def test_frontend_does_not_interpolate_raw_citation_or_chunk_html():
    """Evidence, title, URL và chunk text không được interpolate trực tiếp vào innerHTML."""
    source = APP_JS.read_text(encoding="utf-8")

    unsafe_fragments = [
        "${cit.evidence}",
        "${cit.title}",
        "${cit.source_url}",
        "${chunk.text}",
        "${message}</strong>"
    ]

    for fragment in unsafe_fragments:
        assert fragment not in source
