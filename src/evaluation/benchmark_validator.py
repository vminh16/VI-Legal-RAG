REQUIRED_FIELDS = ("question_id", "question", "question_type", "must_refuse", "gold_sources")


def validate_benchmark_data(
    data: list[dict],
    min_questions: int = 100,
    max_questions: int = 300,
) -> dict:
    """Kiểm tra chất lượng cấu trúc benchmark trước khi dùng làm quality gate."""
    issues = []
    item_reports = []

    if len(data) < min_questions:
        issues.append("benchmark_too_small")
    if len(data) > max_questions:
        issues.append("benchmark_too_large")

    question_ids = set()
    duplicate_ids = set()

    for idx, item in enumerate(data):
        item_issues = []
        for field in REQUIRED_FIELDS:
            if field not in item:
                item_issues.append(f"missing_{field}")

        question_id = item.get("question_id")
        if question_id:
            if question_id in question_ids:
                duplicate_ids.add(question_id)
                item_issues.append("duplicate_question_id")
            question_ids.add(question_id)

        must_refuse = item.get("must_refuse")
        gold_sources = item.get("gold_sources")

        if must_refuse is not True and not gold_sources:
            item_issues.append("missing_gold_sources_for_answerable_question")
        if must_refuse is True and gold_sources:
            item_issues.append("refusal_question_should_not_have_gold_sources")

        if item_issues:
            item_reports.append({
                "index": idx,
                "question_id": question_id,
                "issues": item_issues,
            })

    if duplicate_ids:
        issues.append("duplicate_question_ids")

    return {
        "is_valid": not issues and not item_reports,
        "question_count": len(data),
        "issues": issues,
        "items": item_reports,
    }
