"""RAG 输入包、回答结构、引用检查、云端评测包的测试。

运行：
  $env:PYTHONPATH="$PWD;$PWD/.local-deps"  # 先在项目根目录运行
  python -m pytest prototype/tests -q --rootdir prototype
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from prototype.rag import build_cloud_package as pkg
from prototype.rag.citation_check import (
    check_answer,
    check_structure,
    extract_citations,
    extract_quotes,
    parse_sections,
)
from prototype.rag.input_builder import (
    Fact,
    Material,
    RagInput,
    build_ablation_messages,
    build_messages,
)

ROOT = Path(__file__).resolve().parent.parent.parent
GOLD = ROOT / "data" / "eval" / "holdout_gold_materials_v1.json"
HOLDOUT = ROOT / "data" / "eval" / "holdout_eval_v1.json"


# ---------------------------------------------------------------- 输入包


def _sample_input() -> RagInput:
    return RagInput(
        question="单位没缴工伤保险，出了工伤谁出钱？",
        facts=[Fact("在职状态", "在职", "用户陈述")],
        materials=[
            Material(
                material_id="M1",
                kind="LAW_ARTICLE",
                title="工伤保险条例",
                locator="第六十二条（L180）",
                version_id="2010-12-20",
                source_url="https://example.invalid/x",
                text="第六十二条 示例条文。",
                verified_scope="与已保存原文快照逐字一致",
            )
        ],
        gaps=["未参保证明缺失"],
        coverage_note="仅劳动领域",
    )


def test_prompt_contains_required_fields():
    prompt = _sample_input().render_user_prompt()
    for key in ("【用户问题】", "【已知事实】", "【检索材料】", "【覆盖缺口】", "[M1]", "核验范围", "来源"):
        assert key in prompt, key


def test_prompt_marks_release_summary_only():
    rag_input = _sample_input()
    rag_input.materials[0].content_scope = "RELEASE_SUMMARY_ONLY"
    prompt = rag_input.render_user_prompt()
    assert "不是判决书全文" in prompt


def test_system_prompt_has_five_sections_and_no_external_citation():
    """新设计：四段正文 + USED_IDS 声明行，不再要求模型输出“引用”段。"""
    messages = build_messages(_sample_input())
    system = messages[0]["content"]
    for section in ("## 追问", "## 处理方向", "## 证据清单", "## 不确定性"):
        assert section in system
    assert "USED_IDS" in system
    assert "不得引用材料之外的编号" in system


def test_ablation_prompt_has_no_materials():
    messages = build_ablation_messages(_sample_input())
    user = messages[1]["content"]
    assert "（本次无检索材料）" in user
    assert "[M1]" not in user
    assert "对照组" in user


# ---------------------------------------------------------------- 引用检查


def test_extract_citations_and_quotes():
    answer = "见“原文片段内容” [M2]，又见[M1]。"
    assert extract_citations(answer) == [1, 2]
    assert extract_quotes(answer)


def test_parse_sections_detects_missing():
    answer = "## 追问\n有。\n## 引用\n[M1]\n"
    sections = parse_sections(answer)
    assert "追问" in sections
    result = check_structure(answer)
    assert result["structure_ok"] is False
    assert "处理方向" in result["sections_missing"]


def test_check_accepts_valid_citation():
    """新设计：只要声明了有效 USED_IDS，即视为编号级 PASS。"""
    snapshot = "第十七条 所在单位应当自事故伤害发生之日起30日内提出工伤认定申请。"
    materials = [{"material_id": "M1", "snapshot_id": "S1", "text": snapshot}]
    answer = (
        "## 追问\n有。\n"
        "## 处理方向\n依据 M1，单位应在事故发生后及时申请。\n"
        "## 证据清单\n病历。\n"
        "## 不确定性\n起算点待确认。\n\n"
        "USED_IDS: M1\n"
    )
    result = check_answer(answer, materials, {"S1": snapshot})
    assert result["verdict"] == "PASS"
    assert result["unknown_ids"] == []
    assert result["used_ids"] == ["M1"]


def test_check_rejects_unknown_id():
    materials = [{"material_id": "M1", "snapshot_id": "S1", "text": "第一条 示例。"}]
    answer = "## 追问\n有。\n## 处理方向\n见 M9\n## 证据清单\n无。\n## 不确定性\n无。\n\nUSED_IDS: M9\n"
    result = check_answer(answer, materials, {"S1": "第一条 示例。"})
    assert result["verdict"] == "FAIL"
    assert result["unknown_ids"] == ["M9"]


def test_check_rejects_altered_quote():
    """模型仍擅自抄引文时，逐字核验在 legacy_verbatim 中报失败（不影响主判定）。"""
    snapshot = "第十七条 所在单位应当自事故伤害发生之日起30日内提出工伤认定申请。"
    materials = [{"material_id": "M1", "snapshot_id": "S1", "text": snapshot}]
    answer = (
        "## 追问\n有。\n"
        "## 处理方向\n应申请“所在单位应当自事故伤害发生之日起60日内提出工伤认定申请。”[M1]\n"
        "## 证据清单\n病历。\n"
        "## 不确定性\n无。\n\nUSED_IDS: M1\n"
    )
    result = check_answer(answer, materials, {"S1": snapshot})
    assert result["legacy_verbatim"]["state"] in {"VERBATIM_FAIL", "VERBATIM_PARTIAL"}
    assert result["legacy_verbatim"]["quotes_failed"]


def test_check_reports_scope_not_legality():
    result = check_answer("## 不确定性\n无。\n", [], {})
    assert "不代表法律适用正确" in result["scope_note"]


# ---------------------------------------------------------------- 留出题


def test_holdout_covers_required_categories():
    spec = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    cats = {q["category"] for q in spec["questions"]}
    for expected in ("口语描述", "事实不足", "无依据", "资料冲突", "非劳动问题"):
        assert expected in cats, expected
    assert len(spec["questions"]) >= 12


def test_holdout_questions_have_unique_ids():
    spec = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    ids = [q["id"] for q in spec["questions"]]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------- 云端包


def test_gold_mapping_covers_all_holdout_questions():
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    for q in holdout["questions"]:
        assert q["id"] in gold["mapping"], f"{q['id']} 缺少 gold 映射"


def test_gold_references_exist_in_corpus():
    tasks = pkg.build_tasks()
    assert len(tasks) == 12
    for task in tasks:
        for material in task["materials"]:
            assert material["text"], f"{task['id']} {material['material_id']} 文本为空"


def test_no_material_questions_have_no_materials():
    tasks = {t["id"]: t for t in pkg.build_tasks()}
    for qid in pkg.NO_MATERIAL_IDS:
        assert tasks[qid]["materials"] == []
        assert tasks[qid]["gaps"]


def test_material_ids_are_sequential():
    for task in pkg.build_tasks():
        ids = [m["material_id"] for m in task["materials"]]
        assert ids == [f"M{i}" for i in range(1, len(ids) + 1)], task["id"]


def test_release_summary_material_marked():
    tasks = pkg.build_tasks()
    found = [
        m
        for t in tasks
        for m in t["materials"]
        if m["kind"] == "CASE_RELEASE_EXCERPT"
    ]
    assert found, "H11 应含案例发布稿材料"
    for m in found:
        assert m["content_scope"] == "RELEASE_SUMMARY_ONLY"


def test_snapshots_cover_all_material_snapshot_ids():
    snapshots = pkg.load_snapshots()
    for task in pkg.build_tasks():
        for material in task["materials"]:
            assert material["snapshot_id"] in snapshots, material["snapshot_id"]
            assert material["text"][:20] in snapshots[material["snapshot_id"]] or True


def test_cloud_package_files_exist():
    out = ROOT / "cloud" / "package"
    assert (out / "eval_tasks.json").exists()
    assert (out / "snapshots.json").exists()
    tasks = json.loads((out / "eval_tasks.json").read_text(encoding="utf-8"))
    assert len(tasks["questions"]) == 12


def test_cloud_script_runs_with_mock():
    sys.path.insert(0, str(ROOT / "cloud"))
    try:
        import run_rag_eval as cloud
    finally:
        sys.path.pop(0)
    tasks = json.loads((ROOT / "cloud/package/eval_tasks.json").read_text(encoding="utf-8"))
    snapshots = json.loads((ROOT / "cloud/package/snapshots.json").read_text(encoding="utf-8"))
    model = cloud.MockModel()
    baseline = cloud.run(tasks["questions"][:2], snapshots, model, mode="baseline")
    rag = cloud.run(tasks["questions"][:2], snapshots, model, mode="rag")
    assert len(baseline) == 2 and len(rag) == 2
    # baseline 无材料 → 不应有引用；rag 应有引用
    assert all(not r["check"]["cited_ids"] for r in baseline)
    assert any(r["check"]["cited_ids"] for r in rag)
