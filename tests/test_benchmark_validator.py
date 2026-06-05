from src.evaluation.benchmark_validator import validate_benchmark_data


def test_benchmark_validator_reports_size_and_schema_issues():
    data = [
        {
            "question_id": "q1",
            "question": "Thiếu gold/must_refuse",
            "question_type": "single_article",
        }
    ]

    report = validate_benchmark_data(data, min_questions=2, max_questions=3)

    assert report["is_valid"] is False
    assert "benchmark_too_small" in report["issues"]
    assert "missing_must_refuse" in report["items"][0]["issues"]
    assert "missing_gold_sources_for_answerable_question" in report["items"][0]["issues"]


def test_benchmark_validator_accepts_curated_minimal_valid_set():
    data = [
        {
            "question_id": "q1",
            "question": "Thử việc có được trả lương không?",
            "question_type": "single_article",
            "gold_sources": [{"article": "Điều 26", "clause": None}],
            "must_refuse": False,
        },
        {
            "question_id": "q2",
            "question": "Nấu phở như thế nào?",
            "question_type": "out_of_scope",
            "gold_sources": [],
            "must_refuse": True,
        },
    ]

    report = validate_benchmark_data(data, min_questions=2, max_questions=3)

    assert report["is_valid"] is True
    assert report["issues"] == []
    assert report["items"] == []
