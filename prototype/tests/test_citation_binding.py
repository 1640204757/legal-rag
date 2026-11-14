"""引用绑定检查测试（适配「模型声明编号 + 程序附列原文」新设计）。

旧版测的是“模型抄的引文是否绑定到正确编号”。新设计下模型不再抄原文，
因此本文件保留**逐字核验能力**（用于诊断旧答案或发现模型擅自抄写），
并验证它在 `legacy_verbatim` 中独立报告、不影响主 verdict。

运行：
  $env:PYTHONPATH="$PWD;$PWD/.local-deps"  # 先在项目根目录运行
  python -m pytest prototype/tests -q --rootdir prototype
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prototype.rag.citation_check import check_answer, check_legacy_verbatim, extract_quote_bindings

ROOT = Path(__file__).resolve().parent.parent.parent

SNAPSHOT_ID = "LAW-WORK-INJURY-REGULATION"
M1_TEXT = "第十四条 在上下班途中，受到非本人主要责任的交通事故或者城市轨道交通、客运轮渡、火车事故伤害的；"
M2_TEXT = "第十七条 所在单位应当自事故伤害发生之日起30日内提出工伤认定申请。"
SNAPSHOTS = {SNAPSHOT_ID: M1_TEXT + "\n" + M2_TEXT, "OTHER": "第六条 用人单位应当支付二倍工资。"}
MATERIALS = [
    {"material_id": "M1", "text": M1_TEXT, "snapshot_id": SNAPSHOT_ID},
    {"material_id": "M2", "text": M2_TEXT, "snapshot_id": SNAPSHOT_ID},
]


def _answer(body: str, used: str = "USED_IDS: M1") -> str:
    return (
        "## 追问\n有。\n"
        f"## 处理方向\n{body}\n"
        "## 证据清单\n病历。\n"
        "## 不确定性\n待确认。\n\n"
        f"{used}\n"
    )


# ---------------------------------------------------------------- 绑定提取（保留能力）


def test_quote_binds_to_following_citation():
    bindings = extract_quote_bindings("见“示例引文内容” [M1]。")
    assert bindings[0]["bound_ids"] == ["M1"]


def test_quote_without_citation_is_unbound():
    bindings = extract_quote_bindings("见“示例引文内容”，未标注编号。")
    assert bindings[0]["bound_ids"] == []


def test_quote_does_not_steal_previous_citation():
    bindings = extract_quote_bindings("依据[M1]，另见“示例引文内容”[M2]。")
    assert bindings[0]["bound_ids"] == ["M2"]


# ---------------------------------------------------------------- 逐字核验（诊断用）


def test_legacy_verbatim_ok():
    answer = _answer("原文：“在上下班途中，受到非本人主要责任的交通事故或者城市轨道交通、客运轮渡、火车事故伤害的”[M1]")
    result = check_legacy_verbatim(answer, MATERIALS, SNAPSHOTS)
    assert result["state"] == "VERBATIM_OK"


def test_legacy_verbatim_mismatch():
    """引文来自 M2 却标 M1。"""
    answer = _answer("原文：“所在单位应当自事故伤害发生之日起30日内提出工伤认定申请。”[M1]")
    result = check_legacy_verbatim(answer, MATERIALS, SNAPSHOTS)
    assert result["state"] in {"VERBATIM_FAIL", "VERBATIM_PARTIAL"}
    assert result["quotes_failed"]


def test_legacy_verbatim_snapshot_only():
    """引文在快照里但不在提供材料正文中。"""
    answer = _answer("原文：“完全不在材料正文里的其他内容片段示例”[M1]")
    result = check_legacy_verbatim(answer, MATERIALS, SNAPSHOTS)
    assert result["state"] in {"VERBATIM_FAIL", "VERBATIM_PARTIAL"}


def test_legacy_verbatim_not_found():
    answer = _answer("原文：“这是一段完全编造的条文内容用于测试”[M1]")
    result = check_legacy_verbatim(answer, MATERIALS, SNAPSHOTS)
    assert result["state"] in {"VERBATIM_FAIL", "VERBATIM_PARTIAL"}


def test_legacy_does_not_decide_main_verdict():
    """逐字核验失败不应改变主 verdict（新设计下编号声明才是主判定）。"""
    answer = _answer("原文：“这是编造的内容示例片段”[M1]", used="USED_IDS: M1")
    result = check_answer(answer, MATERIALS, SNAPSHOTS)
    assert result["citation_state"] == "IDS_OK"
    assert result["verdict"] == "PASS"
    assert result["legacy_verbatim"]["state"] in {"VERBATIM_FAIL", "VERBATIM_PARTIAL"}


# ---------------------------------------------------------------- 编号级主判定


def test_main_verdict_uses_used_ids():
    result = check_answer(_answer("如果非本人主要责任，则可能认定工伤。"), MATERIALS, SNAPSHOTS)
    assert result["citation_state"] == "IDS_OK"
    assert result["verdict"] == "PASS"


def test_missing_used_ids_blocks():
    result = check_answer(_answer("依据 M1 分析。", used=""), MATERIALS, SNAPSHOTS)
    assert result["citation_state"] == "IDS_MISSING"
    assert result["verdict"] == "FAIL"


def test_unknown_id_blocks():
    result = check_answer(_answer("依据 M9。", used="USED_IDS: M9"), MATERIALS, SNAPSHOTS)
    assert result["citation_state"] == "IDS_UNKNOWN"
    assert result["verdict"] == "FAIL"


# ---------------------------------------------------------------- 云端包材料带正文


def test_cloud_package_materials_carry_text():
    tasks = json.loads((ROOT / "cloud/package/eval_tasks.json").read_text(encoding="utf-8"))["questions"]
    for task in tasks:
        for material in task["materials"]:
            assert material.get("text"), f"{task['id']} {material['material_id']} 缺正文"
            assert material.get("snapshot_id"), f"{task['id']} {material['material_id']} 缺快照引用"
