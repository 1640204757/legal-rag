"""逐字核验、案例发布稿导入与评测集的测试。

运行：
  $env:PYTHONPATH="$PWD;$PWD/.local-deps"  # 先在项目根目录运行
  python -m pytest prototype/tests -q --rootdir prototype
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prototype.verifier import (
    load_raw,
    normalize_for_compare,
    split_raw_articles,
    verify_article_exact,
    verify_cases,
    verify_quote,
)

ROOT = Path(__file__).resolve().parent.parent.parent
CASES = ROOT / "data" / "cases"
EVAL = ROOT / "data" / "eval" / "retrieval_eval_v1.json"


# ---------------------------------------------------------------- 逐字核验


def test_normalize_only_touches_whitespace():
    src = "第一十条　当事人，不予支持。"
    out = normalize_for_compare(src)
    assert "，" in out and "不予支持" in out
    assert "\u3000" not in out


def test_verifier_accepts_exact_article():
    raw = load_raw("LAW-WORK-INJURY-REGULATION")
    doc = json.loads(
        (ROOT / "data/normalized/LAW-WORK-INJURY-REGULATION.2010-12-20.json").read_text(encoding="utf-8")
    )
    a17 = next(a for a in doc["articles"] if a["article_no"] == "第十七条")
    ok, _ = verify_article_exact(raw, "第十七条", a17["text"])
    assert ok


@pytest.mark.parametrize(
    "mutate,label",
    [
        (lambda t: t.replace("30日", "31日"), "改数字"),
        (lambda t: t[:60], "截断"),
        (lambda t: t.replace("未按", "已按"), "改否定词"),
        (lambda t: t.replace("应当", "可以", 1), "改义务性用语"),
        (lambda t: t + "（注）", "加尾注"),
    ],
)
def test_verifier_rejects_tampering(mutate, label):
    raw = load_raw("LAW-WORK-INJURY-REGULATION")
    doc = json.loads(
        (ROOT / "data/normalized/LAW-WORK-INJURY-REGULATION.2010-12-20.json").read_text(encoding="utf-8")
    )
    a17 = next(a for a in doc["articles"] if a["article_no"] == "第十七条")["text"]
    mutated = mutate(a17)
    if mutated == a17:
        pytest.skip(f"{label} 未改变文本，跳过")
    ok, reason = verify_article_exact(raw, "第十七条", mutated)
    assert not ok, f"{label} 应被拒绝"
    assert reason


def test_verifier_rejects_wrong_article_number():
    raw = load_raw("LAW-WORK-INJURY-REGULATION")
    doc = json.loads(
        (ROOT / "data/normalized/LAW-WORK-INJURY-REGULATION.2010-12-20.json").read_text(encoding="utf-8")
    )
    a18 = next(a for a in doc["articles"] if a["article_no"] == "第十八条")["text"]
    ok, reason = verify_article_exact(raw, "第十七条", a18)
    assert not ok
    assert "不一致" in reason


def test_verifier_tolerates_whitespace_only():
    raw = load_raw("LAW-WORK-INJURY-REGULATION")
    doc = json.loads(
        (ROOT / "data/normalized/LAW-WORK-INJURY-REGULATION.2010-12-20.json").read_text(encoding="utf-8")
    )
    a17 = next(a for a in doc["articles"] if a["article_no"] == "第十七条")["text"]
    ok, _ = verify_article_exact(raw, "第十七条", a17.replace("，", "，\n   "))
    assert ok


def test_split_excludes_chapter_headings():
    raw = load_raw("LAW-WORK-INJURY-REGULATION")
    spans = split_raw_articles(raw)
    assert "第六条" in spans
    assert "第二章" not in spans["第六条"]


def test_quote_accepts_excerpt_rejects_concat():
    raw = load_raw("LAW-WORK-INJURY-REGULATION")
    doc = json.loads(
        (ROOT / "data/normalized/LAW-WORK-INJURY-REGULATION.2010-12-20.json").read_text(encoding="utf-8")
    )
    a17 = next(a for a in doc["articles"] if a["article_no"] == "第十七条")["text"]
    a18 = next(a for a in doc["articles"] if a["article_no"] == "第十八条")["text"]
    assert verify_quote(raw, a17[:50])[0] is True
    assert verify_quote(raw, a17[:30] + "伪造内容" + a18[:30])[0] is False


def test_all_laws_verified():
    from prototype.verifier import verify_all

    results = verify_all()
    failed = [r for r in results if not r.verified]
    assert results, "应至少核验一批法规条款"
    assert not failed, f"逐字核验失败：{[(r.doc_id, r.article_no, r.reason) for r in failed]}"


def test_all_case_sections_verified():
    results = verify_cases()
    assert results, "应核验案例分区"
    failed = [r for r in results if not r.verified]
    assert not failed, f"案例分区核验失败：{[(r.doc_id, r.article_id) for r in failed]}"


# ---------------------------------------------------------------- 案例导入


def test_six_cases_imported():
    files = sorted(CASES.glob("CASE-LABOR-TYPICAL-*.json"))
    assert len(files) == 6


def test_cases_marked_as_release_summary_only():
    for path in sorted(CASES.glob("CASE-LABOR-TYPICAL-*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        assert case["content_scope"] == "RELEASE_SUMMARY_ONLY"
        assert case["full_text_available"] is False
        assert case["court"] is None, "发布稿未标注法院，不得臆测"
        assert case["judgment_date"] is None, "发布稿未标注裁判日期，不得臆测"
        assert case["case_number"] is None, "发布稿未标注案号，不得臆测"


def test_case_sections_have_locators():
    case = json.loads((CASES / "CASE-LABOR-TYPICAL-01.json").read_text(encoding="utf-8"))
    types = {s["section_type"] for s in case["sections"]}
    assert "RELEASE_BASIC_FACTS" in types
    assert "RELEASE_JUDGMENT_RESULT" in types
    assert "RELEASE_TYPICAL_SIGNIFICANCE" in types
    for s in case["sections"]:
        if s.get("text"):
            assert s["locator"], "有正文的分区必须有原文定位"


def test_cases_pass_schema():
    pytest.importorskip("jsonschema")
    from jsonschema import Draft202012Validator, FormatChecker

    schema = json.loads((ROOT / "schema/case_document.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for path in sorted(CASES.glob("*.json")):
        instance = json.loads(path.read_text(encoding="utf-8"))
        errors = list(validator.iter_errors(instance))
        assert errors == [], f"{path.name}: {[e.message for e in errors[:3]]}"


# ---------------------------------------------------------------- 评测集


def test_eval_spec_shape():
    spec = json.loads(EVAL.read_text(encoding="utf-8"))
    assert len(spec["questions"]) >= 15
    traps = {q["trap"] for q in spec["questions"] if q.get("trap")}
    assert "unverified_domain" in traps
    assert "no_hit" in traps
    assert "hit_but_unsupported" in traps


def test_eval_runs_and_reports_metrics():
    from prototype.eval_runner import run_eval

    report = run_eval()
    assert report["total"] == 15
    assert "doc_hit_rate" in report
    assert "degrade_accuracy" in report
    assert report["false_citations"] == 0, "不得把不支持的材料标为依据"


def test_eval_unverified_domain_never_cites():
    from prototype.eval_runner import run_eval

    report = run_eval()
    for row in report["rows"]:
        if row["trap"] == "unverified_domain":
            assert row["degraded"] is True
            assert row["citations"] == 0
