"""日常评测体系回归：标注入库校验、固定拆分、指标分母和跨次比较。

只用小型合成语料检查评分口径；真实向量/4B实测由两个命令入口另行留证。
"""
import copy

import pytest

from local_rag.daily_evaluation import (
    aggregate_retrieval, aggregate_answers, audit_answer, compare_runs, fixed_split, sample_answers,
    score_retrieval, validate_questions, acceptance_gate,
)


def sample_question(qid="D001", split="dev"):
    """提供带逐标注支撑词的最小题目；返回新字典，不读写磁盘。"""
    return dict(id=qid, category="消费", query="钱交了东西没来，急死了咋办",
                split=split, expect_scenario="消费", label_status="AI草拟待人工抽检",
                expect_any=[dict(title="测试法", article_no="第二条", support_keywords=["返还", "预付款"])])


def test_label_validation_requires_real_article_and_all_support_words():
    rows = [dict(title="测试法", locator="第二条 第2行", text="第二条 应当返还预付款。")]
    assert validate_questions([sample_question()], rows) == []
    for field, value in (("title", "不存在法"), ("article_no", "第三条"),
                         ("support_keywords", ["返还", "离婚"]), ("support_keywords", [])):
        bad = sample_question()
        bad["expect_any"][0][field] = value
        assert validate_questions([bad], rows), field


def test_validation_rejects_duplicate_ids_formal_question_and_missing_draft_status():
    row = dict(title="测试法", locator="第二条", text="返还预付款")
    good = sample_question()
    assert validate_questions([good, copy.deepcopy(good)], [row])
    for change in ({"query": "依据《测试法》第二条怎么办"}, {"label_status": "已人工核准"},
                   {"expect_scenario": ""}, {"split": "train"}):
        assert validate_questions([{**good, **change}], [row])


def test_fixed_split_is_reproducible_stratified_and_does_not_mutate_input():
    questions = [{**sample_question(f"D{c:02}{n:02}"), "category": f"类{c}"}
                 for c in range(12) for n in range(10)]
    original = copy.deepcopy(questions)
    result = fixed_split(questions, seed=20261031)
    assert result == fixed_split(list(reversed(questions)), seed=20261031)
    assert questions == original
    assert sum(q["split"] == "dev" for q in result) == 80
    assert sum(q["split"] == "holdout" for q in result) == 40
    for category in {q["category"] for q in result}:
        assert {q["split"] for q in result if q["category"] == category} == {"dev", "holdout"}


def test_hit_at_four_and_mrr_use_first_acceptable_rank_and_include_errors():
    q = sample_question()
    hits = [dict(title="其他法", locator="第一条"), dict(title="测试法", locator="第2条")]
    first = score_retrieval(q, hits, mode="hybrid_qdrant")
    assert first["hit"] is True and first["reciprocal_rank"] == .5
    miss = score_retrieval(sample_question("D002"), hits[:1], mode="lexical")
    error = score_retrieval(sample_question("D003", "holdout"), [], mode="error", error="timeout")
    summary = aggregate_retrieval([first, miss, error])
    assert summary["dev"]["count"] == 2
    assert summary["dev"]["hit_at_4"] == .5
    assert summary["dev"]["mrr"] == .25
    assert summary["holdout"]["count"] == 1
    assert summary["holdout"]["errors"] == 1
    assert summary["dev"]["categories"]["消费"]["hit_at_4"] == .5
    assert summary["holdout"]["hit_at_4"] == 0
    assert score_retrieval(q, hits[:1] * 4 + hits[1:], mode="lexical")["hit"] is False


def test_run_comparison_marks_holdout_regressions_and_refuses_incompatible_runs():
    scope = dict(dataset_sha256="a", corpus_id="b", embedding_id="c", mode_counts={"hybrid_qdrant": 2})
    old = dict(scope=scope, results=[dict(id="D1", split="dev", hit=False), dict(id="D2", split="holdout", hit=True)])
    new = dict(scope=scope, results=[dict(id="D1", split="dev", hit=True), dict(id="D2", split="holdout", hit=False)])
    diff = compare_runs(old, new)
    assert diff["newly_fixed"] == ["D1"]
    assert diff["regressed"] == ["D2"]
    assert diff["holdout_regressions"] == ["D2"]
    for changed in (dict(corpus_id="other"), dict(dataset_sha256="other"),
                    dict(mode_counts={"lexical": 2}), dict(embedding_id="other")):
        mismatch = {**new, "scope": {**scope, **changed}}
        assert compare_runs(old, mismatch)["comparable"] is False
        assert compare_runs(old, mismatch)["newly_fixed"] == []
    assert compare_runs(old, {**new, "results": new["results"][:1]})["comparable"] is False


def test_answer_checks_use_raw_declaration_display_and_real_materials():
    q = sample_question()
    raw = "## 追问\n钱付了吗？\n## 处理方向\n1. 保存凭证。2. 联系商家。3. 向法院求助。[M1]\n## 证据清单\n付款记录\n## 不确定性\n需核实。USED_IDS: M1"
    display = raw.split("USED_IDS")[0].replace("[M1]", "")
    materials = [dict(material_id="M1", title="其他法", locator="第三条", text="退款")]
    checks = audit_answer(q, dict(raw_answer=raw, answer=display, blocked=False), materials)
    assert checks["missing_used_ids"] is False
    assert checks["display_internal_ids"] is False
    assert checks["missing_headings"] is False
    assert checks["missing_steps"] is False
    assert checks["wrong_institutions"] is False
    assert checks["bad_citations"] is False
    bad = audit_answer(q, dict(raw_answer="依据《其他法》第四条[M9]", answer="找单位法务[M9] USED_IDS: M9"), materials)
    assert bad["missing_used_ids"] and bad["bad_citations"]
    assert bad["wrong_institutions"] and bad["display_internal_ids"]
    assert bad["missing_headings"] and bad["missing_steps"]


def test_structured_selection_is_audited_on_assembled_final_answer():
    """结构化模式 raw 只是 SELECT/ASK，结构和声明必须核程序装配后的 final_answer。"""
    final = ("## 追问\n请补充事情发生的时间、参与主体和已有记录。\n"
             "## 处理方向\n材料[M1]原文：“第二条 应当返还预付款。”\n"
             "1. 保存已有文件、凭证和沟通记录。\n2. 到官方文本核对版本、主体、条件及例外。\n"
             "## 证据清单\n保存与问题有关的原始文件、通知、凭证和沟通记录。\n"
             "## 不确定性\n以上是材料阅读线索，不认定个案适用；待核实事项并不穷尽。\nUSED_IDS: M1")
    checks = audit_answer(sample_question(),
                          dict(raw_answer="SELECT: M1#1\nASK: NONE", final_answer=final,
                               answer="《测试法》原文：“第二条 应当返还预付款。”", blocked=False,
                               answer_assembly={"mode": "selection"}),
                          [dict(material_id="M1", title="测试法", locator="第二条", text="第二条 应当返还预付款。")])
    assert checks["missing_used_ids"] is False
    assert checks["missing_headings"] is False and checks["missing_steps"] is False
    assert checks["bad_citations"] is False and checks["unbound_articles"] == []


def test_answer_checks_allow_scenario_institutions_but_keep_error_denominator():
    q = {**sample_question(), "expect_scenario": "校园"}
    assert not audit_answer(q, {"answer": "向教务处反映", "raw_answer": "USED_IDS: 无"}, [])["wrong_institutions"]
    result = audit_answer(q, {}, [], error="timeout")
    assert result["error"] == "timeout" and result["completed"] is False
    assert result["missing_used_ids"] is None


def test_answer_sample_is_fixed_balanced_and_dev_only_by_default():
    questions = [{**sample_question(f"D{c}-{n}"), "category": str(c)} for c in range(12) for n in range(7)]
    questions += [sample_question("H1", "holdout")]
    chosen = sample_answers(questions, count=20)
    assert len(chosen) == 20 and all(q["split"] == "dev" for q in chosen)
    assert len({q["category"] for q in chosen}) == 12
    assert chosen == sample_answers(list(reversed(questions)), count=20)
    with pytest.raises(ValueError):
        sample_answers(questions, count=100)


def test_answer_summary_separates_unknown_checks_from_completed_results():
    q = sample_question()
    a = audit_answer(q, {"answer": "已拦截", "raw_answer": "USED_IDS: 无", "blocked": True}, [])
    b = audit_answer(q, {}, [], error="timeout")
    summary = aggregate_answers([a, b])
    assert summary["requested"] == 2 and summary["completed"] == 1 and summary["errors"] == 1
    assert summary["checks"]["blocked"] == dict(numerator=1, denominator=1, rate=1.0, unknown=1)


@pytest.mark.parametrize("raw", ["正文[M1]。USED_IDS: 无", "正文。USED_IDS: M99"])
def test_answer_audit_rejects_incomplete_or_unknown_declared_ids(raw):
    checks = audit_answer(sample_question(), {"raw_answer": raw, "answer": "正文"},
                          [dict(material_id="M1", title="测试法", locator="第二条")])
    assert checks["bad_citations"] is True


def test_mixed_numbering_same_step_does_not_count_twice():
    raw = "## 处理方向\n1. 保存证据。第一步继续保存证据。\nUSED_IDS: 无"
    assert audit_answer(sample_question(), {"raw_answer": raw, "answer": raw}, [])["missing_steps"]


def test_named_law_without_book_brackets_is_still_audited():
    q = {**sample_question(), "expect_any": [dict(title="中华人民共和国民法典", article_no="第二条")]}
    raw = "依据民法典第三条[M1]处理。USED_IDS: M1"
    checks = audit_answer(q, {"raw_answer": raw, "answer": raw},
                          [dict(material_id="M1", title="中华人民共和国民法典", locator="第二条")])
    assert checks["bad_citations"]


def test_document_title_is_not_miscounted_as_unknown_law():
    """真实抽检里的《催告函》是文书名称；后文有“明确要求”也不能据此认定伪造法律。"""
    raw = "发送《催告函》，明确要求归还。依据《测试法》第二条[M1]处理。USED_IDS: M1"
    checks = audit_answer(sample_question(), {"raw_answer": raw, "answer": raw},
                          [dict(material_id="M1", title="测试法", locator="第二条")])
    assert not checks["bad_citations"]


def test_unknown_law_without_article_number_remains_a_citation_issue():
    raw = "依据《不存在法》的规定，要求对方退款。USED_IDS: 无"
    assert audit_answer(sample_question(), {"raw_answer": raw, "answer": raw}, [])["bad_citations"]


def test_raw_wrong_route_remains_counted_after_guard_blocks_display():
    q = {**sample_question(), "query": "街头被猥亵后反咬告我强奸", "expect_scenario": "街头刑事"}
    checked = audit_answer(q, dict(raw_answer="提起民事反诉并找单位管理责任。USED_IDS: 无",
                                  answer="已拦截", blocked=True, declaration_repair={"repaired": True}), [])
    assert checked["raw_wrong_criminal_route"] and not checked["wrong_criminal_route"]
    assert checked["raw_wrong_institutions"] and not checked["wrong_institutions"]
    assert checked["declaration_repaired"]


def test_answer_comparison_requires_same_sample_and_reports_warning_changes():
    from local_rag.daily_evaluation import compare_answer_runs
    scope = dict(dataset_sha256="d", corpus_id="c", embedding_id="e", mode_counts={"hybrid_qdrant": 1})
    before = dict(scope=scope, runtime={"model_digest": "m"}, results=[dict(id="D1", query="问句", mode="hybrid_qdrant", checks={"declaration_warnings": True})])
    after = copy.deepcopy(before)
    after["results"][0]["checks"]["declaration_warnings"] = False
    diff = compare_answer_runs(before, after)
    assert diff["comparable"] and diff["metrics"]["declaration_warnings"]["before"]["rate"] == 1
    assert diff["metrics"]["declaration_warnings"]["after"]["rate"] == 0
    after["results"][0]["id"] = "D2"
    assert not compare_answer_runs(before, after)["comparable"]


def test_acceptance_rejects_holdout_mrr_loss_even_when_hit_count_is_equal():
    scope = dict(dataset_sha256="a", corpus_id="b", embedding_id="c", mode_counts={"hybrid_qdrant": 1})
    results = [dict(id="D1", split="dev", hit=True, mode="hybrid_qdrant")]
    baseline = dict(scope=scope, results=results, summary={"dev": {"count": 80, "hits": 30, "mrr": .2},
                    "holdout": {"count": 40, "hits": 10, "mrr": .19}, "v2": {"hits": 33}})
    current = copy.deepcopy(baseline)
    current["summary"]["dev"]["hits"] = 31
    current["summary"]["dev"]["mrr"] = .21
    assert acceptance_gate(baseline, current)["passed"]
    current["summary"]["holdout"]["mrr"] = .18
    assert not acceptance_gate(baseline, current)["passed"]

