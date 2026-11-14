"""逐字核验：把规范化条款与原文快照做字符级比对。

设计原则：
- 只做“引用文本是否为原文的连续子串”这一件事，不做编辑距离、不做模糊匹配；
- 比对前只做**不改变含义**的归一化（空白折叠、全角空格、零宽字符、换行）；
- 任何数字、否定词、标点都保留；归一化后不相等即为核验失败；
- 核验结果写入 citation_eligibility.verified_verbatim，但**不因通过就允许“法定依据”标签**。

用法：
    python -m prototype.verifier            # 核验全部条款
    python -m prototype.verifier --report   # 输出 JSON 报告
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

# 仅剥离不影响语义的字符；不碰数字、否定词与标点
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)


def normalize_for_compare(text: str) -> str:
    """归一化：零宽字符、全角空格、各类空白折叠为单空格。

    允许的归一化仅限空白层面；不折叠标点、不删数字、不做同义替换。
    """
    text = text.translate(_ZERO_WIDTH)
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 每行内折叠空白，行与行之间用单换行；再整体折叠连续换行
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    return text.strip()


def _whitespace_insensitive(text: str) -> str:
    """去掉全部空白，仅用于“空白差异不算差异”的比对。

    不删除数字、标点、否定词；只删除空格、换行、制表符。
    """
    return re.sub(r"\s+", "", normalize_for_compare(text))


@dataclass
class VerbatimResult:
    doc_id: str
    article_id: str
    article_no: str
    verified: bool
    reason: str = ""
    normalized_len: int = 0


def load_raw(doc_id: str) -> str:
    """读取原文快照，并按统一正文边界截取。

    与导入共用 BODY_BOUNDARIES，避免核验区间把页脚/导航算进条款。
    """
    from prototype.importer import capture_for, extract_body_text

    manifest = json.loads((RAW_DIR / "capture_manifest.json").read_text(encoding="utf-8"))
    entry = next((c for c in manifest["captures"] if c["doc_id"] == doc_id), None)
    if entry is None:
        raise RuntimeError(f"采集记录中没有 {doc_id}")
    capture_for(doc_id, entry["raw_file"])
    raw = (RAW_DIR / entry["raw_file"]).read_text(encoding="utf-8")
    return extract_body_text(doc_id, raw)


ARTICLE_HEAD_RE = re.compile(r"^第([一二三四五六七八九十百]+)条\s*")
CHAPTER_HEAD_RE = re.compile(r"^第([一二三四五六七八九十百]+)[章节]\s*")
STRUCTURE_HEAD_RE = re.compile(r"^(第[一二三四五六七八九十百]+[章节]|附录|附件)")


def split_raw_articles(raw_text: str) -> dict[str, str]:
    """从原文中按“第X条”切分，返回 {条号: 原文片段}。

    片段 = 该条标题起，到下一条标题或下一个章节/附录标题前。
    章节标题不属于条款正文，必须排除，否则会把“第二章 工伤保险基金”算进上一条。
    """
    norm = normalize_for_compare(raw_text)
    lines = norm.split("\n")
    spans: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines:
        article_match = ARTICLE_HEAD_RE.match(line)
        if article_match:
            key = f"第{article_match.group(1)}条"
            current = key
            spans.setdefault(key, []).append(line)
            continue
        if STRUCTURE_HEAD_RE.match(line):
            current = None
            continue
        if current is not None:
            spans[current].append(line)
    return {k: "\n".join(v).strip() for k, v in spans.items()}


def verify_article_exact(raw_text: str, article_no: str, article_text: str) -> tuple[bool, str]:
    """条款级精确比对：文本必须与原文中同一编号的完整片段一致。

    比子串包含更严格：
    - 截断的文本不再通过（必须覆盖整条到下一标题前）；
    - 张冠李戴不再通过（条号必须与原文片段编号一致）；
    - 仍容忍空白/换行排版差异。
    """
    spans = split_raw_articles(raw_text)
    expected = spans.get(article_no)
    if expected is None:
        return False, f"原文中未找到 {article_no} 的切片"

    actual = normalize_for_compare(article_text)
    if _whitespace_insensitive(expected) == _whitespace_insensitive(actual):
        return True, ""

    exp_tight = _whitespace_insensitive(expected)
    act_tight = _whitespace_insensitive(actual)
    if act_tight and act_tight in exp_tight:
        return False, "条款文本是原文片段的截断，未覆盖完整条款"
    if exp_tight in act_tight:
        return False, "条款文本包含原文片段之外的增补内容"
    limit = min(len(exp_tight), len(act_tight))
    i = 0
    while i < limit and exp_tight[i] == act_tight[i]:
        i += 1
    return False, f"与原文 {article_no} 片段不一致（去空白后首处差异位于第 {i} 字符）"


def verify_quote(raw_text: str, quote: str) -> tuple[bool, str]:
    """引用级核验：判断展示给用户的引用是否为原文的连续片段。

    允许截断（引用本来就可能是节选），但必须逐字一致、不得拼接、不得改写。
    """
    if not _whitespace_insensitive(quote):
        return False, "引用为空"
    if _whitespace_insensitive(quote) in _whitespace_insensitive(raw_text):
        return True, ""
    return False, "引用文本不是原文的连续片段（可能拼接、改写或跨条拼接）"


def verify_article(raw_text: str, article_text: str) -> tuple[bool, str]:
    """兼容旧接口：子串级检查（保留用于引用片段）。"""
    if not normalize_for_compare(article_text):
        return False, "条款文本为空"

    raw_norm = normalize_for_compare(raw_text)
    art_norm = normalize_for_compare(article_text)
    if art_norm in raw_norm:
        return True, ""

    raw_tight = _whitespace_insensitive(raw_text)
    art_tight = _whitespace_insensitive(article_text)
    if art_tight in raw_tight:
        return True, ""

    first_line = art_norm.split("\n")[0]
    if first_line and first_line in raw_norm:
        return False, "首行命中但完整条款不是连续子串（可能跨行切割或含增补文本）"
    limit = min(len(art_tight), len(raw_tight))
    i = 0
    while i < limit and art_tight[i] == raw_tight[i]:
        i += 1
    return False, f"未在原文中定位到该条款（去空白后首处差异位于第 {i} 字符）"


def verify_cases() -> list[VerbatimResult]:
    """核验案例发布稿分区文本是否为发布稿原文的连续片段。"""
    from prototype.case_importer import RELEASE_FILE

    release = (RAW_DIR / RELEASE_FILE).read_text(encoding="utf-8")
    release_tight = _whitespace_insensitive(release)
    results: list[VerbatimResult] = []
    cases_dir = DATA_DIR / "cases"
    if not cases_dir.exists():
        return results
    for path in sorted(cases_dir.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        for section in case.get("sections", []):
            text = section.get("text")
            if not text:
                continue
            ok = _whitespace_insensitive(text) in release_tight
            results.append(
                VerbatimResult(
                    doc_id=case["case_id"],
                    article_id=section["section_type"],
                    article_no=section["section_type"],
                    verified=ok,
                    reason="" if ok else "发布稿分区文本不是原文连续片段",
                    normalized_len=len(text),
                )
            )
    return results


def verify_all() -> list[VerbatimResult]:
    """汇总既有法规条款与案例分区的快照逐字核验结果，不进行在线官网复核。"""
    results: list[VerbatimResult] = []
    for path in sorted((DATA_DIR / "normalized").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        raw = load_raw(doc["doc_id"])
        for art in doc["articles"]:
            article_no = art.get("article_no") or ""
            ok, reason = verify_article_exact(raw, article_no, art["text"])
            results.append(
                VerbatimResult(
                    doc_id=doc["doc_id"],
                    article_id=art["article_id"],
                    article_no=article_no,
                    verified=ok,
                    reason=reason,
                    normalized_len=len(normalize_for_compare(art["text"])),
                )
            )
    return results


def cmd_verify(as_json: bool = False) -> int:
    """输出逐字核验结果和失败项，供命令行验证使用。"""
    results = verify_all()
    case_results = verify_cases()
    all_results = results + case_results
    passed = [r for r in all_results if r.verified]
    failed = [r for r in all_results if not r.verified]

    if as_json:
        print(json.dumps(
            {
                "law_total": len(results),
                "law_verified": sum(1 for r in results if r.verified),
                "case_total": len(case_results),
                "case_verified": sum(1 for r in case_results if r.verified),
                "total": len(all_results),
                "verified": len(passed),
                "failed": len(failed),
                "results": [r.__dict__ for r in all_results],
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0 if not failed else 1

    by_doc: dict[str, list[VerbatimResult]] = {}
    for r in all_results:
        by_doc.setdefault(r.doc_id, []).append(r)
    for doc_id, rows in by_doc.items():
        ok = sum(1 for r in rows if r.verified)
        print(f"{doc_id}: {ok}/{len(rows)} 条逐字核验通过")
    for r in failed:
        print(f"  FAIL {r.doc_id} {r.article_no} ({r.article_id}): {r.reason}")
    print(
        f"VERBATIM_OK ({len(passed)}/{len(all_results)}; 法规 {len(results)} + 案例 {len(case_results)})"
        if not failed
        else f"VERBATIM_FAILED ({len(failed)})"
    )
    return 0 if not failed else 1


def apply_eligibility() -> int:
    """把逐字核验结果写入 citation_eligibility。

    注意：即使 verified_verbatim=True，其余三项（版本、依据资格、事实匹配）仍为 false，
    verdict 固定为 UNVERIFIED，不会产生“法定依据”资格。
    """
    from datetime import datetime

    checked_at = datetime.now().astimezone().isoformat(timespec="seconds")
    by_doc: dict[str, list[VerbatimResult]] = {}
    for r in verify_all():
        by_doc.setdefault(r.doc_id, []).append(r)

    updated = 0
    for path in sorted((DATA_DIR / "normalized").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows = by_doc.get(doc["doc_id"], [])
        all_verbatim = bool(rows) and all(r.verified for r in rows)
        doc["citation_eligibility"] = {
            "verified_verbatim": all_verbatim,
            "version_confirmed": False,
            "authority_eligible": False,
            "facts_matched": False,
            "verdict": "UNVERIFIED",
            "checked_at": checked_at,
            "notes": (
                "逐字核验通过，但版本适用/依据资格/事实匹配未完成，故仍为 UNVERIFIED，"
                "不得标“本案处置法定依据”"
                if all_verbatim
                else "逐字核验存在失败条款"
            ),
        }
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        updated += 1
    print(f"ELIGIBILITY_WRITTEN ({updated} docs, verdict=UNVERIFIED)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """解析本模块命令行参数，执行对应操作；具体数据读写由下层函数负责。"""
    parser = argparse.ArgumentParser(description="逐字核验")
    parser.add_argument("--report", action="store_true", help="以 JSON 输出")
    parser.add_argument("--write-eligibility", action="store_true", help="把核验结果写入规范化文件")
    args = parser.parse_args(argv)
    if args.write_eligibility:
        return apply_eligibility()
    return cmd_verify(as_json=args.report)


if __name__ == "__main__":
    sys.exit(main())
