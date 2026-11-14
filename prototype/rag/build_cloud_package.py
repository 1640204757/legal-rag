"""生成云端评测包：eval_tasks.json + snapshots.json。

材料来源分两种，均写入包内，便于对照：
1. gold（默认）：来自 data/eval/holdout_gold_materials_v1.json 的人工核对映射；
2. auto（对照）：本项目词面检索的自动结果，仅记录用于比较，不作为默认材料。

为什么默认用 gold：
- 实测纯词面检索在口语化提问上准确率不足（H01/H03/H10 首版自动检索给出的材料明显不相关）；
- 把不相关材料塞给模型，会让“RAG 是否带来改善”这一结论失真；
- gold 映射的来源、依据与缺口全部可审计，云端不再重新检索，baseline/rag 题面一致。

边界：
- 只使用本地已保存快照，不联网；
- gold 选材为人工核对，未经领域专家复核；
- 语料外问题（H05/H06/H08/H09）固定不给材料。

用法：
    python -m prototype.rag.build_cloud_package --out cloud/package
    python -m prototype.rag.build_cloud_package --out cloud/package --auto-for-compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
CASES_DIR = DATA / "cases"
HOLDOUT = DATA / "eval" / "holdout_eval_v1.json"
GOLD = DATA / "eval" / "holdout_gold_materials_v1.json"

# 语料外/非劳动题：强制不给材料
NO_MATERIAL_IDS = {"H05", "H06", "H08", "H09"}


def load_snapshots() -> dict[str, str]:
    """读取云端评测包使用的本地快照；不在此函数中登录或抓取外网。"""
    from prototype.verifier import load_raw

    snapshots = {
        "LAW-LABOR-INTERP-2": load_raw("LAW-LABOR-INTERP-2"),
        "LAW-WORK-INJURY-REGULATION": load_raw("LAW-WORK-INJURY-REGULATION"),
    }
    from prototype.case_importer import RELEASE_FILE

    snapshots["CASE-LABOR-TYPICAL-RELEASE"] = (DATA / "raw" / RELEASE_FILE).read_text(encoding="utf-8")
    return snapshots


def _law_articles() -> dict[tuple[str, str], dict]:
    """索引：(doc_id, article_no) -> 条款信息。"""
    from prototype.importer import load_documents

    out = {}
    for doc in load_documents():
        for art in doc.articles:
            out[(doc.doc_id, art.article_no)] = {
                "title": doc.title,
                "locator": f"{art.article_no}（{art.locator}）",
                "version_id": doc.version_id,
                "source_url": doc.official_url,
                "text": art.text,
                "doc_id": doc.doc_id,
            }
    return out


def _cases() -> dict[str, dict]:
    """读取云端评测所需案例材料，保留摘要范围与来源。"""
    out = {}
    if not CASES_DIR.exists():
        return out
    for path in sorted(CASES_DIR.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        body = "\n".join(
            f"【{s['section_type']}】\n{s['text']}"
            for s in case.get("sections", [])
            if s.get("text")
        )
        out[case["case_id"]] = {
            "title": case["title"],
            "locator": case.get("citation_card", {}).get("locator") or "发布稿",
            "version_id": "2025-08-01",
            "source_url": case["source_url"],
            "text": case["title"] + "\n" + body,
            "content_scope": case.get("content_scope", "UNKNOWN"),
        }
    return out


def build_gold_materials(task_id: str, gold: dict, law_index: dict, case_index: dict) -> list[dict]:
    """按人工核对映射组装材料。编号从 M1 起，法规在前、案例在后。"""
    spec = gold["mapping"].get(task_id)
    if not spec:
        return []

    materials: list[dict] = []
    counter = 1

    for ref in spec.get("articles", []):
        if "#" not in ref:
            raise ValueError(f"非法条款引用：{ref}")
        doc_id, article_no = ref.split("#", 1)
        info = law_index.get((doc_id, article_no))
        if info is None:
            raise KeyError(f"gold 映射引用了不存在的条款：{ref}")
        materials.append(
            {
                "material_id": f"M{counter}",
                "kind": "LAW_ARTICLE",
                "title": info["title"],
                "locator": info["locator"],
                "version_id": info["version_id"],
                "source_url": info["source_url"],
                "text": info["text"],
                "verified_scope": "与已保存原文快照逐字一致（离线快照，非在线复核）",
                "content_scope": "FULL_TEXT",
                "snapshot_id": info["doc_id"],
            }
        )
        counter += 1

    for case_id in spec.get("cases", []):
        info = case_index.get(case_id)
        if info is None:
            raise KeyError(f"gold 映射引用了不存在的案例：{case_id}")
        materials.append(
            {
                "material_id": f"M{counter}",
                "kind": "CASE_RELEASE_EXCERPT",
                "title": info["title"],
                "locator": info["locator"],
                "version_id": info["version_id"],
                "source_url": info["source_url"],
                "text": info["text"],
                "verified_scope": "与已保存发布稿快照逐字一致",
                "content_scope": info["content_scope"],
                "snapshot_id": "CASE-LABOR-TYPICAL-RELEASE",
            }
        )
        counter += 1
    return materials


def build_auto_materials(query: str, *, top_k: int = 3) -> list[dict]:
    """自动检索结果，仅用于对照记录。"""
    from prototype.importer import LexicalIndex, build_index, load_documents

    docs = load_documents()
    index = build_index(docs)
    lex = LexicalIndex(index)
    rows = []
    for i, (_, row, _) in enumerate(lex.score(query)[:top_k], start=1):
        rows.append(
            {
                "material_id": f"M{i}",
                "kind": "LAW_ARTICLE",
                "title": row["title"],
                "locator": f"{row['article_no']}（{row['locator']}）",
                "source_url": row["official_url"],
            }
        )
    return rows


def build_tasks(*, include_auto_compare: bool = False) -> list[dict]:
    """组装评测题、所选材料和覆盖缺口，写出供评测脚本使用的数据包。"""
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    law_index = _law_articles()
    case_index = _cases()

    tasks = []
    for q in holdout["questions"]:
        spec = gold["mapping"].get(q["id"], {})
        if q["id"] in NO_MATERIAL_IDS:
            materials = []
        else:
            materials = build_gold_materials(q["id"], gold, law_index, case_index)

        gaps = list(spec.get("gaps", []))
        if q["id"] in NO_MATERIAL_IDS:
            gaps = ["本题为语料外/本领域外问题，未提供任何检索材料"] + gaps
        elif not materials:
            gaps.insert(0, "未检索到可用材料")

        task = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "facts": q.get("facts", []),
            "materials": materials,
            "gaps": gaps,
            "expect": q.get("expect", {}),
            "material_source": "gold" if materials else "none",
            "selection_rationale": spec.get("rationale", ""),
        }
        if include_auto_compare:
            task["auto_retrieval_compare"] = {
                "note": "自动词面检索结果，仅作对照；准确率不足，未用作默认材料",
                "materials": build_auto_materials(q["question"]) if q["id"] not in NO_MATERIAL_IDS else [],
            }
        tasks.append(task)
    return tasks


def main(argv=None) -> int:
    """解析本模块命令行参数，执行对应操作；具体数据读写由下层函数负责。"""
    parser = argparse.ArgumentParser(description="生成云端评测包")
    parser.add_argument("--out", default="cloud/package")
    parser.add_argument("--auto-for-compare", action="store_true", help="附带自动检索对照（不改变默认材料）")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(include_auto_compare=args.auto_for_compare)
    snapshots = load_snapshots()

    (out_dir / "eval_tasks.json").write_text(
        json.dumps(
            {
                "_note": "留出题 + 人工核对材料（gold）。编号已固定；baseline 与 rag 共用同一题面。",
                "version": "1.0",
                "created": "2026-10-19",
                "material_source": "gold（人工核对，未经营领域专家复核）",
                "questions": tasks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "snapshots.json").write_text(
        json.dumps(snapshots, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    with_materials = sum(1 for t in tasks if t["materials"])
    release_only = sum(
        1 for t in tasks for m in t["materials"] if m["content_scope"] == "RELEASE_SUMMARY_ONLY"
    )
    print(f"tasks={len(tasks)} with_materials={with_materials} no_materials={len(tasks) - with_materials}")
    print(f"materials_total={sum(len(t['materials']) for t in tasks)} release_summary_materials={release_only}")
    print(f"snapshots={len(snapshots)} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
