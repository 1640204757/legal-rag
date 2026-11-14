"""引用状态与真实失败案例回归（适配新输出设计）。

保留 v1/v2 真实云端答案作为回归证据：
- v1 H01：旧检查器误判 PASS（无引文）
- v2 各题：暴露无材料编造、伪示例引用、自相矛盾
- v3 H02：正文抄长引文但未声明 USED_IDS、期限无主体

运行：
  $env:PYTHONPATH="$PWD;$PWD/.local-deps"  # 先在项目根目录运行
  python -m pytest prototype/tests -q --rootdir prototype
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from prototype.rag.citation_check import check_answer, check_ids
from prototype.rag.postprocess import postprocess

ROOT = Path(__file__).resolve().parent.parent.parent
V1 = ROOT / "artifacts" / "cloud-runs" / "results_qwen3_8b_first2.json"
V2 = ROOT / "artifacts" / "cloud-runs" / "results_qwen3_8b_seen2_v2.json"
V3 = ROOT / "artifacts" / "cloud-runs" / "results_qwen3_8b_seen2_v3.json"

MATERIALS = [
    {"material_id": "M1", "text": "第十四条 在上下班途中，受到非本人主要责任的交通事故伤害的；",
     "snapshot_id": "S", "title": "条例", "locator": "第十四条", "version_id": "v1",
     "source_url": "https://example.invalid/1", "content_scope": "FULL_TEXT"},
    {"material_id": "M2", "text": "第十七条 单位30日内申请；职工1年内可直接申请。",
     "snapshot_id": "S", "title": "条例", "locator": "第十七条", "version_id": "v1",
     "source_url": "https://example.invalid/2", "content_scope": "FULL_TEXT"},
]
SNAPSHOTS = {"S": MATERIALS[0]["text"] + "\n" + MATERIALS[1]["text"]}


def _answer(body: str, used: str = "USED_IDS: M1") -> str:
    return (
        "## 追问\n有。\n"
        f"## 处理方向\n{body}\n"
        "## 证据清单\n病历。\n"
        "## 不确定性\n待确认。\n\n"
        f"{used}\n"
    )


# ---------------------------------------------------------------- 编号级状态


def test_no_citation_state_when_materials_given():
    result = check_ids(_answer("建议先协商。", used=""), MATERIALS)
    assert result["citation_state"] == "IDS_MISSING"


def test_ids_ok_state():
    result = check_ids(_answer("依据 M1 判断。"), MATERIALS)
    assert result["citation_state"] == "IDS_OK"
    assert result["verdict"] == "PASS"


def test_no_material_clean_state():
    result = check_ids(_answer("无可引用原文。", used="USED_IDS: 无"), [])
    assert result["citation_state"] == "NO_MATERIAL_CLEAN"
    assert result["verdict"] == "PASS"


def test_no_material_with_citation_state():
    result = check_ids(_answer("不足以支持确定结论[M1]。", used="USED_IDS: 无"), [])
    assert result["citation_state"] == "NO_MATERIAL_HAS_CITATION"
    assert result["verdict"] == "FAIL"


def test_ids_none_used_state():
    result = check_ids(_answer("材料无关。", used="USED_IDS: 无"), MATERIALS)
    assert result["citation_state"] == "IDS_NONE_USED"
    assert result["verdict"] == "PASS_WITH_NOTE"


# ---------------------------------------------------------------- 真实失败案例


@pytest.mark.skipif(not V1.exists(), reason="缺少 v1 结果")
def test_v1_h01_fabricated_id_now_fails():
    run = json.loads(V1.read_text(encoding="utf-8"))
    row = next(r for r in run["results"]["rag"] if r["id"] == "H01")
    materials = [
        {"material_id": "M1", "text": "第六条 未订立书面劳动合同应支付二倍工资。", "snapshot_id": "S2"},
        {"material_id": "M2", "text": "第十八条 违法解除后应付工资。", "snapshot_id": "S2"},
    ]
    result = check_answer(row["answer"], materials, {"S2": materials[0]["text"] + materials[1]["text"]})
    # 旧版无 USED_IDS → 新判定为 IDS_MISSING；无论哪种，都不是 PASS
    assert result["verdict"] != "PASS"


@pytest.mark.skipif(not V2.exists(), reason="缺少 v2 结果")
def test_v2_baseline_h02_fake_example_blocked():
    run = json.loads(V2.read_text(encoding="utf-8"))
    row = next(r for r in run["results"]["baseline"] if r["id"] == "H02")
    post = postprocess(row["answer"], [])
    assert post["blocking"] is True
    assert any(i["type"] == "FAKE_EXAMPLE_CITATION" for i in post["blocking_issues"])


@pytest.mark.skipif(not V2.exists(), reason="缺少 v2 结果")
def test_v2_rag_h01_no_material_citation_blocked():
    run = json.loads(V2.read_text(encoding="utf-8"))
    row = next(r for r in run["results"]["rag"] if r["id"] == "H01")
    post = postprocess(row["answer"], [])
    assert post["blocking"] is True
    assert any(i["type"] == "NO_MATERIAL_HAS_CITATION" for i in post["blocking_issues"])


@pytest.mark.skipif(not V3.exists(), reason="缺少 v3 结果")
def test_v3_rag_h02_missing_used_ids():
    run = json.loads(V3.read_text(encoding="utf-8"))
    row = next(r for r in run["results"]["rag"] if r["id"] == "H02")
    result = check_answer(row["answer"], MATERIALS, SNAPSHOTS)
    assert result["citation_state"] == "IDS_MISSING"
    assert result["verdict"] == "FAIL"


@pytest.mark.skipif(not V3.exists(), reason="缺少 v3 结果")
def test_v3_baseline_h02_unprovided_article_blocked():
    run = json.loads(V3.read_text(encoding="utf-8"))
    row = next(r for r in run["results"]["baseline"] if r["id"] == "H02")
    post = postprocess(row["answer"], [])
    assert post["blocking"] is True
    assert any(i["type"] == "UNSUPPORTED_ARTICLE" for i in post["blocking_issues"])


# ---------------------------------------------------------------- 本地与云端一致


@pytest.mark.skipif(not (V2.exists() and V3.exists()), reason="缺少云端结果")
def test_local_and_cloud_agree():
    sys.path.insert(0, str(ROOT / "cloud"))
    try:
        import run_rag_eval as cloud
    finally:
        sys.path.pop(0)

    for path in (V2, V3):
        run = json.loads(path.read_text(encoding="utf-8"))
        for mode, rows in run["results"].items():
            for row in rows:
                materials = MATERIALS if mode == "rag" else []
                local = check_answer(row["answer"], materials, SNAPSHOTS)
                remote = cloud.check_answer(row["answer"], materials, SNAPSHOTS)
                assert local["verdict"] == remote["verdict"], (path.name, mode, row["id"])
                assert local["citation_state"] == remote["citation_state"], (path.name, mode, row["id"])
