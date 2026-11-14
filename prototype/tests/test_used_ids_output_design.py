"""输出设计调整的回归测试：USED_IDS + 程序附列原文 + 疑似标记 + 期限主体。

对应 2026-10-19 第三轮实测（v3）暴露的问题：
- rag H02：正文长引文未绑编号、“引用”段只是概述 → 改为模型不抄原文
- baseline H02：先讲一般规则再说不适用于本案 → 矛盾检查降级为“疑似”
- rag H02：期限无主体/起算/来源 → 期限检查按上下文窗口判定
- 本地模型句尾连写声明：保留标点并排除声明编号，空声明不跨行吞正文。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from prototype.rag.citation_check import USED_IDS_RE, check_answer, check_ids, find_citations, parse_used_ids
from prototype.rag.deadline_check import check_deadline_notice
from prototype.rag.postprocess import postprocess

ROOT = Path(__file__).resolve().parent.parent.parent
V3 = ROOT / "artifacts" / "cloud-runs" / "results_qwen3_8b_seen2_v3.json"

MATERIALS = [
    {
        "material_id": "M1",
        "title": "工伤保险条例",
        "locator": "第十四条",
        "version_id": "2010-12-20",
        "source_url": "https://example.invalid/m1",
        "text": "第十四条 (六)在上下班途中，受到非本人主要责任的交通事故伤害的；",
        "content_scope": "FULL_TEXT",
        "snapshot_id": "S",
    },
    {
        "material_id": "M2",
        "title": "工伤保险条例",
        "locator": "第十七条",
        "version_id": "2010-12-20",
        "source_url": "https://example.invalid/m2",
        "text": "第十七条 所在单位应当自事故伤害发生之日起30日内提出工伤认定申请；职工或者其近亲属在1年内可以直接提出。",
        "content_scope": "FULL_TEXT",
        "snapshot_id": "S",
    },
]
SNAPSHOTS = {"S": MATERIALS[0]["text"] + "\n" + MATERIALS[1]["text"]}


def _answer(body: str, used: str = "USED_IDS: M1") -> str:
    return (
        "## 追问\n请补充事故责任认定。\n"
        f"## 处理方向\n{body}\n"
        "## 证据清单\n责任认定书。\n"
        "## 不确定性\n责任认定未提供。\n\n"
        f"{used}\n"
    )


# ---------------------------------------------------------------- USED_IDS 解析


def test_parse_used_ids_variants():
    assert parse_used_ids("USED_IDS: M1, M2") == ["M1", "M2"]
    assert parse_used_ids("USED_IDS：M2 M1 M1") == ["M2", "M1"]
    assert parse_used_ids("USED_IDS: 无") == []
    assert parse_used_ids("没有这一行") is None


@pytest.mark.parametrize("separator", ["\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", "\t"])
def test_inline_used_ids_parse_without_consuming_body_punctuation(separator):
    """独立行或句尾声明均可解析，去声明不能吃掉前置标点或把声明编号计入正文。"""
    body = "正文依据[M1]得出结论" + separator
    raw = body + "USED_IDS: M1 M2 M3"
    assert parse_used_ids(raw) == ["M1", "M2", "M3"]
    assert find_citations(raw) == ["M1"]
    assert USED_IDS_RE.sub("", raw) == body


@pytest.mark.parametrize("raw", ["正文XUSED_IDS: M1", "正文_USED_IDS: M1"])
def test_used_ids_keyword_inside_identifier_is_not_a_declaration(raw):
    """没有句界/空白的普通词串不当作声明，防止任意内部子串吞掉正文。"""
    assert parse_used_ids(raw) is None
    assert USED_IDS_RE.sub("", raw) == raw


@pytest.mark.parametrize("declaration", ["USED_IDS：无", "USED_IDS:"])
def test_inline_empty_or_none_declaration_is_recognized(declaration):
    """中文冒号和空/无声明仍返回空列表，与漏声明的 None 区分。"""
    assert parse_used_ids("没有直接依据。" + declaration) == []
    assert find_citations("没有直接依据。" + declaration) == []


def test_empty_declaration_does_not_consume_next_line_body():
    """空声明的值只能取同行，下一行正文编号仍需接受正文引用检查。"""
    raw = "正文。USED_IDS:\n下一行应核实[M2]。"
    assert parse_used_ids(raw) == []
    assert find_citations(raw) == ["M2"]
    assert USED_IDS_RE.sub("", raw) == "正文。\n下一行应核实[M2]。"


# ---------------------------------------------------------------- 编号级状态


def test_ids_ok_without_any_quote():
    """新设计：不抄原文，只要编号声明有效就是 IDS_OK。"""
    result = check_ids(_answer("如果责任为非本人主要，则可能认定工伤（依据 M1）。"), MATERIALS)
    assert result["citation_state"] == "IDS_OK"
    assert result["verdict"] == "PASS"


def test_ids_missing_declaration_fails():
    result = check_ids(_answer("依据 M1 分析。", used=""), MATERIALS)
    assert result["citation_state"] == "IDS_MISSING"
    assert result["verdict"] == "FAIL"


def test_ids_none_used_is_pass_with_note():
    result = check_ids(_answer("材料不足以判断。", used="USED_IDS: 无"), MATERIALS)
    assert result["citation_state"] == "IDS_NONE_USED"
    assert result["verdict"] == "PASS_WITH_NOTE"


def test_unknown_declared_id_fails():
    result = check_ids(_answer("依据 M9。", used="USED_IDS: M9"), MATERIALS)
    assert result["citation_state"] == "IDS_UNKNOWN"
    assert result["verdict"] == "FAIL"


def test_no_material_clean_passes():
    result = check_ids(_answer("无材料时不编号。", used="USED_IDS: 无"), [])
    assert result["citation_state"] == "NO_MATERIAL_CLEAN"
    assert result["verdict"] == "PASS"


def test_no_material_with_citation_fails():
    result = check_ids(_answer("依据[M1]。", used="USED_IDS: 无"), [])
    assert result["citation_state"] == "NO_MATERIAL_HAS_CITATION"
    assert result["verdict"] == "FAIL"


# ---------------------------------------------------------------- 程序附列原文


def test_citation_cards_generated_from_materials():
    post = postprocess(_answer("依据 M1。", used="USED_IDS: M1, M2"), MATERIALS)
    assert [c["material_id"] for c in post["citation_cards"]] == ["M1", "M2"]
    for card in post["citation_cards"]:
        assert card["origin"] == "SYSTEM_APPENDED"
        assert card["label"] == "系统附列原文"
        assert card["text"]  # 正文来自材料，不是模型


def test_cards_do_not_count_as_model_verification():
    post = postprocess(_answer("依据 M1。", used="USED_IDS: M1"), MATERIALS)
    assert post["model_verbatim_citation_verified"] is False


def test_cards_fallback_when_no_used_ids():
    """缺少 USED_IDS 时，用正文编号回退生成卡片，并标记来源。"""
    answer = _answer("依据 M1 分析。", used="")
    post = postprocess(answer, MATERIALS)
    assert post["card_id_source"] == "BODY_REPAIRED"
    assert post['declaration_repair']['repaired']
    assert [c["material_id"] for c in post["citation_cards"]] == ["M1"]


# ---------------------------------------------------------------- 疑似而非阻断


def test_contradiction_is_suspected_not_blocking():
    """先讲一般规则、再说不适用于本案，不能自动判为矛盾。"""
    answer = _answer(
        "根据 M1，上下班途中非本人主要责任的交通事故，应当认定为工伤。"
        "但本案是否适用仍需核实责任认定。",
        used="USED_IDS: M1",
    )
    post = postprocess(answer, MATERIALS)
    contradictory = [i for i in post["suspected_issues"] if i["type"] == "SUSPECTED_SELF_CONTRADICTION"]
    assert contradictory, "应作为疑似问题提示"
    assert post["blocking"] is False, "疑似问题不得自动阻断"
    assert post["displayable_as_verified"] is True


def test_blocking_only_for_hard_errors():
    post = postprocess(_answer("依据 M9。", used="USED_IDS: M9"), MATERIALS)
    assert post["blocking"] is True
    assert any(i["type"] == "UNKNOWN_ID" for i in post["blocking_issues"])


# ---------------------------------------------------------------- 期限检查


def test_deadline_without_subject_flagged():
    answer = _answer("建议在事故发生之日起1年内向社会保险行政部门提出工伤认定申请。")
    result = check_deadline_notice(answer)
    assert result["has_deadline"] is True
    types = {i["type"] for i in result["issues"]}
    assert types & {"DEADLINE_NO_SUBJECT", "DEADLINE_SUBJECT_INCOMPLETE"}


def test_deadline_without_source_flagged():
    answer = _answer("单位应当在事故发生之日起30日内提出申请。")
    result = check_deadline_notice(answer)
    assert "DEADLINE_NO_SOURCE" in {i["type"] for i in result["issues"]}


def test_deadline_complete_not_flagged():
    answer = _answer(
        "依据 M2：所在单位应当自事故伤害发生之日起30日内申请；"
        "职工或者其近亲属可在1年内直接申请。"
    )
    result = check_deadline_notice(answer)
    assert result["has_deadline"] is True
    assert result["issues"] == [], f"完整期限提示不应报问题：{result['issues']}"


def test_no_deadline_no_issue():
    result = check_deadline_notice(_answer("材料不足，暂不判断期限。"))
    assert result["has_deadline"] is False
    assert result["needs_review"] is False


# ---------------------------------------------------------------- 真实 v3 答案回归


@pytest.mark.skipif(not V3.exists(), reason="缺少 v3 云端结果")
def test_real_v3_rag_h02_missing_used_ids():
    run = json.loads(V3.read_text(encoding="utf-8"))
    row = next(r for r in run["results"]["rag"] if r["id"] == "H02")
    check = check_answer(row["answer"], MATERIALS, SNAPSHOTS)
    assert check["citation_state"] == "IDS_MISSING"
    assert check["verdict"] == "FAIL"
    # 旧版才会把它算成 VERBATIM_FAIL；新版把它归到声明缺失
    assert check["legacy_verbatim"]["state"] in {"VERBATIM_FAIL", "VERBATIM_PARTIAL", "NO_QUOTE"}


@pytest.mark.skipif(not V3.exists(), reason="缺少 v3 云端结果")
def test_real_v3_rag_h02_deadline_incomplete():
    run = json.loads(V3.read_text(encoding="utf-8"))
    row = next(r for r in run["results"]["rag"] if r["id"] == "H02")
    result = check_deadline_notice(row["answer"])
    types = {i["type"] for i in result["issues"]}
    assert "DEADLINE_SUBJECT_INCOMPLETE" in types
    assert "DEADLINE_NO_SOURCE" in types


@pytest.mark.skipif(not V3.exists(), reason="缺少 v3 云端结果")
def test_real_v3_baseline_h02_unprovided_article_blocked():
    """安全升级：即使条件化表述，无材料的具体条号仍必须拦截。"""
    run = json.loads(V3.read_text(encoding="utf-8"))
    row = next(r for r in run["results"]["baseline"] if r["id"] == "H02")
    post = postprocess(row["answer"], [])
    assert post["blocking"] is True
    assert post["displayable_as_verified"] is False


@pytest.mark.skipif(not V3.exists(), reason="缺少 v3 云端结果")
def test_local_and_cloud_agree_on_v3():
    sys.path.insert(0, str(ROOT / "cloud"))
    try:
        import run_rag_eval as cloud
    finally:
        sys.path.pop(0)

    run = json.loads(V3.read_text(encoding="utf-8"))
    for mode, rows in run["results"].items():
        for row in rows:
            materials = MATERIALS if mode == "rag" else []
            local = check_answer(row["answer"], materials, SNAPSHOTS)
            remote = cloud.check_answer(row["answer"], materials, SNAPSHOTS)
            assert local["verdict"] == remote["verdict"], (mode, row["id"])
            assert local["citation_state"] == remote["citation_state"], (mode, row["id"])

            local_post = postprocess(row["answer"], materials)
            remote_post = cloud.postprocess_answer(row["answer"], materials)
            # 历史云端脚本保留旧口径；本地新增法条核验必须更严格，不能倒退。
            if local_post["blocking"] != remote_post["blocking"]:
                assert local_post["blocking"] and not remote_post["blocking"]
                assert any(i["type"] in {"UNSUPPORTED_ARTICLE", "UNSUPPORTED_LAW"}
                           for i in local_post["blocking_issues"])
            else:
                assert local_post["blocking"] == remote_post["blocking"], (mode, row["id"])
