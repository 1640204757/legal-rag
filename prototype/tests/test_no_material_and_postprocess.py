"""无材料分支与后处理测试（适配新输出设计：USED_IDS + 程序附列 + 疑似标记）。

运行：
  $env:PYTHONPATH="$PWD;$PWD/.local-deps"  # 先在项目根目录运行
  python -m pytest prototype/tests -q --rootdir prototype
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from prototype.rag.input_builder import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_NO_MATERIALS,
    SYSTEM_PROMPT_WITH_MATERIALS,
    RagInput,
    build_ablation_messages,
    build_messages,
    select_system_prompt,
)
from prototype.rag.postprocess import postprocess

ROOT = Path(__file__).resolve().parent.parent.parent
V2 = ROOT / "artifacts" / "cloud-runs" / "results_qwen3_8b_seen2_v2.json"
V3 = ROOT / "artifacts" / "cloud-runs" / "results_qwen3_8b_seen2_v3.json"


# ---------------------------------------------------------------- 提示词分支


def test_no_material_prompt_forbids_all_ids():
    assert "空集" in SYSTEM_PROMPT_NO_MATERIALS
    assert "不得出现任何形如方括号加字母 M 加数字的编号标记" in SYSTEM_PROMPT_NO_MATERIALS


def test_no_material_prompt_has_no_copyable_example():
    assert not re.search(r"\[\s*M\s*\d+\s*\]", SYSTEM_PROMPT_NO_MATERIALS)


def test_with_material_prompt_asks_for_used_ids_not_quoting():
    """新设计：模型不抄原文，只声明 USED_IDS。"""
    assert "USED_IDS" in SYSTEM_PROMPT_WITH_MATERIALS
    assert "不要抄写条文原文" in SYSTEM_PROMPT_WITH_MATERIALS
    assert "系统会根据" in SYSTEM_PROMPT_WITH_MATERIALS


def test_prompt_requires_deadline_subject_and_trigger():
    assert "谁" in SYSTEM_PROMPT_WITH_MATERIALS
    assert "从何时起算" in SYSTEM_PROMPT_WITH_MATERIALS


def test_prompt_limits_followup_questions():
    assert "每次追问不超过 3 项" in SYSTEM_PROMPT_WITH_MATERIALS
    assert "不要问“是否存在合法劳动关系”" in SYSTEM_PROMPT_WITH_MATERIALS


def test_select_prompt_branch():
    assert select_system_prompt([]) is SYSTEM_PROMPT_NO_MATERIALS
    assert select_system_prompt([{"material_id": "M1"}]) is SYSTEM_PROMPT_WITH_MATERIALS


def test_build_messages_uses_no_material_prompt():
    messages = build_messages(RagInput(question="被开除了赔多少？", materials=[]))
    assert messages[0]["content"] is SYSTEM_PROMPT_NO_MATERIALS
    assert "不得出现任何 [M…] 编号" in messages[1]["content"]


def test_ablation_uses_no_material_prompt():
    messages = build_ablation_messages(RagInput(question="工伤赔多少？", materials=[{"material_id": "M1"}]))
    assert messages[0]["content"] is SYSTEM_PROMPT_NO_MATERIALS


def test_default_prompt_is_with_materials():
    assert SYSTEM_PROMPT is SYSTEM_PROMPT_WITH_MATERIALS


# ---------------------------------------------------------------- 后处理阻断


def _mats():
    return [{"material_id": "M1", "text": "第十七条 示例条文。", "snapshot_id": "S",
             "title": "条例", "locator": "第十七条"}]


def test_no_material_but_citation_blocked():
    answer = (
        "## 追问\n无。\n## 处理方向\n可以先协商。\n"
        "## 证据清单\n流水。\n## 不确定性\n缺少事实[M1]。\n\nUSED_IDS: 无\n"
    )
    result = postprocess(answer, [])
    assert "NO_MATERIAL_HAS_CITATION" in {i["type"] for i in result["blocking_issues"]}
    assert result["blocking"] is True
    assert result["displayable_as_verified"] is False


def test_unknown_id_blocked():
    answer = "## 处理方向\n依据 M9。\n## 不确定性\n无。\n\nUSED_IDS: M9\n"
    result = postprocess(answer, _mats())
    assert "UNKNOWN_ID" in {i["type"] for i in result["blocking_issues"]}
    assert result["blocking"] is True


def test_fake_example_citation_blocked():
    answer = (
        "## 处理方向\n应当认定为工伤。\n## 不确定性\n缺少材料。\n"
        "## 引用\n原文：“示例条文”[M1]（注：此处为示例引用）\n"
    )
    result = postprocess(answer, [])
    assert "FAKE_EXAMPLE_CITATION" in {i["type"] for i in result["blocking_issues"]}


def test_contradiction_is_suspected_only():
    answer = (
        "## 处理方向\n根据 M1，应当认定为工伤。但本案仍需核实。\n"
        "## 不确定性\n责任认定未提供。\n\nUSED_IDS: M1\n"
    )
    result = postprocess(answer, _mats())
    assert any(i["type"] == "SUSPECTED_SELF_CONTRADICTION" for i in result["suspected_issues"])
    assert result["blocking"] is False, "疑似问题不得自动阻断"


def test_conditional_analysis_not_flagged():
    answer = (
        "## 处理方向\n如果责任认定为非本人主要，则可能符合；若为主要责任，则不符合。\n"
        "## 不确定性\n责任认定未提供。\n\nUSED_IDS: M1\n"
    )
    result = postprocess(answer, _mats())
    assert not any("CONTRADICTION" in i["type"] for i in result["suspected_issues"])


def test_missing_used_ids_repaired_from_explicit_legal_body():
    answer = "## 处理方向\n依据 M1。\n## 不确定性\n无。\n"
    result = postprocess(answer, _mats())
    assert not any(i["type"] == "MISSING_USED_IDS" for i in result["suspected_issues"])
    assert result["card_id_source"] == "BODY_REPAIRED"
    assert result["declaration_repair"]["repaired"] and result["raw_answer"] == answer


def test_answer_never_modified():
    answer = "## 不确定性\n缺少事实[M1]。\n"
    result = postprocess(answer, [])
    assert result["raw_answer"] == answer
    assert result["answer"] == answer
    assert result["modified"] is False
    assert result["blocking"] is True


# ---------------------------------------------------------------- 真实答案回归


@pytest.mark.skipif(not V2.exists(), reason="缺少 v2 结果")
def test_v2_rag_h01_blocked():
    run = json.loads(V2.read_text(encoding="utf-8"))
    row = next(r for r in run["results"]["rag"] if r["id"] == "H01")
    result = postprocess(row["answer"], [])
    assert result["blocking"] is True
    assert result["displayable_as_verified"] is False


@pytest.mark.skipif(not V2.exists(), reason="缺少 v2 结果")
def test_v2_baseline_h02_fake_example():
    run = json.loads(V2.read_text(encoding="utf-8"))
    row = next(r for r in run["results"]["baseline"] if r["id"] == "H02")
    result = postprocess(row["answer"], [])
    types = {i["type"] for i in result["blocking_issues"]}
    assert "FAKE_EXAMPLE_CITATION" in types


@pytest.mark.skipif(not V2.exists(), reason="缺少 v2 结果")
def test_v2_baseline_h01_not_blocked():
    run = json.loads(V2.read_text(encoding="utf-8"))
    row = next(r for r in run["results"]["baseline"] if r["id"] == "H01")
    result = postprocess(row["answer"], [])
    assert result["blocking"] is False
    assert result["displayable_as_verified"] is True


@pytest.mark.skipif(not V3.exists(), reason="缺少 v3 结果")
def test_v3_rag_h01_clean_and_h02_suspected():
    run = json.loads(V3.read_text(encoding="utf-8"))
    h01 = next(r for r in run["results"]["rag"] if r["id"] == "H01")
    h02 = next(r for r in run["results"]["rag"] if r["id"] == "H02")

    r01 = postprocess(h01["answer"], [])
    assert r01["blocking"] is False, "v3 H01 无编号，不应阻断"

    r02 = postprocess(h02["answer"], _mats())
    assert any(i["type"] == "MISSING_USED_IDS" for i in r02["suspected_issues"])
