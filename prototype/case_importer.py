"""官方公开案例发布稿导入。

边界（务必保留）：
- 只导入最高人民法院官网“典型案例发布”稿件；
- 发布稿包含【基本案情】【裁判结果】【典型意义】，这是**官方发布的摘要性内容**，
  不是判决书全文；不得当作全文使用，也不得冒充“本院查明/本院认为”的完整文书；
- 字段层面显式记录 `content_scope = RELEASE_SUMMARY_ONLY` 与 `full_text_available = false`；
- 案号、法院、审级、裁判日期在发布稿中未标注，一律记 null，不推测、不回填；
- 每条案例保留原文定位（在发布稿文本中的字符区间），供逐字核验。

用法：
    python -m prototype.case_importer import
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
CASES_DIR = DATA_DIR / "cases"

RELEASE_FILE = "case-labor-typical-release.txt"
RELEASE_URL = "https://www.court.gov.cn/zixun/xiangqing/472681.html"
RELEASE_TITLE = "最高法发布劳动争议典型案例"

CN_NUM = "一二三四五六七八九十"

SECTION_PATTERNS = {
    "basic_facts": "【基本案情】",
    "judgment_result": "【裁判结果】",
    "typical_significance": "【典型意义】",
}

CASE_HEAD_RE = re.compile(r"^案例([一二三四五六七八九十]+)$", re.MULTILINE)


def load_release_text() -> str:
    """读取已保存的官方案例发布稿文本，供案例分区抽取使用。"""
    path = RAW_DIR / RELEASE_FILE
    if not path.exists():
        raise RuntimeError(f"缺少发布稿原文：{path}")
    return path.read_text(encoding="utf-8")


def release_checksum() -> str:
    """计算案例发布稿快照的内容校验值，用于追溯资料来源。"""
    return hashlib.sha256((RAW_DIR / RELEASE_FILE).read_bytes()).hexdigest()


def slice_body(text: str) -> tuple[str, int]:
    """截取“劳动争议典型案例”目录之后的正文，返回 (正文, 偏移)。"""
    marker = "劳动争议典型案例"
    idx = text.find(marker)
    if idx < 0:
        raise RuntimeError("未在发布稿中定位到“劳动争议典型案例”标题")
    # 跳过目录段：从第一个“案例一”正文标题开始
    first = re.search(r"^案例一$", text[idx:], re.MULTILINE)
    if not first:
        raise RuntimeError("未定位到案例一正文")
    start = idx + first.start()
    return text[start:], start


def split_cases(body: str) -> list[dict]:
    """按“案例X”切分，返回含标题、分区与定位的列表。"""
    heads = list(CASE_HEAD_RE.finditer(body))
    cases = []
    for i, m in enumerate(heads):
        start = m.start()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        block = body[start:end]
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) < 2:
            continue
        case_no = m.group(1)
        title = lines[1]

        sections: dict[str, dict] = {}
        current_key = None
        buffer: list[str] = []
        for line in lines[2:]:
            matched = next((k for k, pat in SECTION_PATTERNS.items() if line.startswith(pat)), None)
            if matched:
                if current_key:
                    sections[current_key] = _make_section(current_key, buffer)
                current_key = matched
                buffer = []
                continue
            if current_key:
                buffer.append(line)
        if current_key:
            sections[current_key] = _make_section(current_key, buffer)

        cases.append(
            {
                "case_no_cn": f"案例{case_no}",
                "title": title,
                "sections": sections,
                "release_locator": f"chars {start}-{end}",
            }
        )
    return cases


def _make_section(key: str, buffer: list[str]) -> dict:
    """构造带类型与定位的案例发布稿分区，不将其冒充裁判文书全文。"""
    text = "\n".join(buffer)
    return {
        "section_type": {
            "basic_facts": "RELEASE_BASIC_FACTS",
            "judgment_result": "RELEASE_JUDGMENT_RESULT",
            "typical_significance": "RELEASE_TYPICAL_SIGNIFICANCE",
        }[key],
        "text": text,
        "char_len": len(text),
    }


def build_case_documents() -> list[dict]:
    """构建本批官方发布稿的结构化案例记录；当前资料能力受这份发布稿范围限制。"""
    text = load_release_text()
    body, offset = slice_body(text)
    parsed = split_cases(body)
    if not parsed:
        raise RuntimeError("未解析出任何案例")

    retrieved_at = _release_retrieved_at()
    imported_at = datetime.now().astimezone().isoformat(timespec="seconds")
    checksum = release_checksum()

    docs = []
    for idx, case in enumerate(parsed, start=1):
        case_id = f"CASE-LABOR-TYPICAL-{idx:02d}"
        sections_payload = []
        for key in ("basic_facts", "judgment_result", "typical_significance"):
            section = case["sections"].get(key)
            if section is None:
                sections_payload.append(
                    {
                        "section_type": {
                            "basic_facts": "RELEASE_BASIC_FACTS",
                            "judgment_result": "RELEASE_JUDGMENT_RESULT",
                            "typical_significance": "RELEASE_TYPICAL_SIGNIFICANCE",
                        }[key],
                        "text": None,
                        "locator": None,
                        "extract_method": "OFFICIAL_TEXT",
                        "extraction_status": "NOT_PROVIDED_IN_RELEASE",
                        "review_status": "UNREVIEWED",
                        "subject_label": "COURT",
                    }
                )
                continue
            sections_payload.append(
                {
                    "section_type": section["section_type"],
                    "text": section["text"],
                    "locator": f"{case['release_locator']}#{key}",
                    "char_len": section["char_len"],
                    "extract_method": "OFFICIAL_TEXT",
                    "extraction_status": "PARSED_FROM_RELEASE",
                    "review_status": "UNREVIEWED",
                    "subject_label": "COURT",
                }
            )

        docs.append(
            {
                "schema_version": "0.3.0",
                "case_id": case_id,
                "case_number": None,
                "official_library_id": None,
                "title": case["title"],
                "court": None,
                "level": "UNKNOWN",
                "procedure": "LABOR",
                "judgment_type": "UNKNOWN",
                "judgment_date": None,
                "publish_date": _release_publish_date(),
                "jurisdiction": "CN",
                "cause_of_action_raw": "劳动争议",
                "cause_of_action_standard": None,
                "standard_version": None,
                "charge_raw": None,
                "charge_standard": None,
                "identities": [
                    {
                        "identity_type": "TYPICAL_CASE",
                        "name": RELEASE_TITLE,
                        "number": case["case_no_cn"],
                        "grant_body": "最高人民法院",
                        "document": RELEASE_TITLE,
                        "date": _release_publish_date(),
                        "source": RELEASE_URL,
                        "verification_status": "VERIFIED",
                        "verified_at": retrieved_at,
                        "normative_effect": "REFERENCE",
                    }
                ],
                "sections": sections_payload,
                "official_gists": [
                    {
                        "gist_type": "TYPICAL_SIGNIFICANCE",
                        "text": case["sections"].get("typical_significance", {}).get("text"),
                        "source": RELEASE_URL,
                        "locator": f"{case['release_locator']}#typical_significance",
                        "verified_at": None,
                    }
                ] if case["sections"].get("typical_significance") else [],
                "model_summary": None,
                "applicable_law_versions": [],
                "related_cases": [],
                "effective_status": "UNKNOWN",
                "public_status": "PUBLIC",
                "collection_policy": "COLLECT",
                "content_scope": "RELEASE_SUMMARY_ONLY",
                "full_text_available": False,
                "content_scope_note": (
                    "仅收录最高人民法院官网发布的典型案例稿件（基本案情/裁判结果/典型意义），"
                    "不是判决书全文；案号、法院、审级、裁判日期发布稿未标注，故为 null。"
                ),
                "source_url": RELEASE_URL,
                "source_body": "最高人民法院新闻局",
                "retrieved_at": retrieved_at,
                "imported_at": imported_at,
                "content_hash": checksum,
                "verified_at": None,
                "citation_card": {
                    "case_number": None,
                    "court": None,
                    "level": "UNKNOWN",
                    "judgment_date": None,
                    "identity_summary": ["典型案例（最高人民法院发布）"],
                    "effective_status": "UNKNOWN",
                    "source_url": RELEASE_URL,
                    "locator": case["release_locator"],
                    "coverage_note": "仅发布稿摘要，非判决书全文；不宣称代表裁判文书原貌。",
                },
            }
        )
    return docs


def _release_meta() -> dict:
    """读取案例发布稿的采集元数据，供来源与时间字段使用。"""
    path = RAW_DIR / "capture_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = next((c for c in payload["captures"] if c["doc_id"] == "CASE-LABOR-TYPICAL-RELEASE"), None)
    return entry or {}


def _release_retrieved_at() -> str:
    """取得发布稿的真实采集时间，不用当前导入时间替代。"""
    meta = _release_meta()
    if not meta.get("retrieved_at"):
        raise RuntimeError("采集记录缺少 CASE-LABOR-TYPICAL-RELEASE 的 retrieved_at")
    return meta["retrieved_at"]


def _release_publish_date() -> str | None:
    """读取发布稿所记录的发布日期；不从案情猜测裁判日期。"""
    return _release_meta().get("publish_date")


def cmd_import() -> int:
    """执行资料导入命令，把本模块处理结果写入项目数据目录。"""
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    docs = build_case_documents()
    for doc in docs:
        out = CASES_DIR / f"{doc['case_id']}.json"
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        fields = [s["section_type"] for s in doc["sections"] if s.get("text")]
        print(f"imported {doc['case_id']} sections={len(fields)} scope={doc['content_scope']} -> {out.name}")
    print(f"CASES_IMPORTED ({len(docs)})")
    return 0


def main(argv: list[str] | None = None) -> int:
    """解析本模块命令行参数，执行对应操作；具体数据读写由下层函数负责。"""
    parser = argparse.ArgumentParser(description="案例发布稿导入")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("import")
    args = parser.parse_args(argv)
    if args.cmd == "import":
        return cmd_import()
    return 2


if __name__ == "__main__":
    sys.exit(main())
