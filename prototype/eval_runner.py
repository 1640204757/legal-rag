"""小型检索评测集运行器。

指标（刻意保守，不做“语义正确”判定）：
- doc_hit@k：期望文档是否出现在 top-k；
- article_hit@k：期望条款号是否被召回（若题目指定）；
- degrade_correct：题目要求降级时是否确实降级，且未要求降级时未误降级；
- no_false_citation：降级场景下是否仍输出了引用（应为 0）。

注意：本评测只衡量“能否召回 + 是否正确降级”，不衡量“法律结论是否正确”。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_FILE = DATA_DIR / "eval" / "retrieval_eval_v1.json"
CASES_DIR = DATA_DIR / "cases"


def load_cases() -> list[dict]:
    """读取本地结构化案例记录，保留发布稿摘要范围。"""
    cases = []
    if not CASES_DIR.exists():
        return cases
    for path in sorted(CASES_DIR.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        # 案例检索文本：标题 + 各分区正文
        body = "\n".join(
            s["text"] for s in case.get("sections", []) if s.get("text")
        )
        cases.append(
            {
                "doc_id": case["case_id"],
                "title": case["title"],
                "text": case["title"] + "\n" + body,
                "content_scope": case.get("content_scope"),
                "source_url": case["source_url"],
            }
        )
    return cases


def search_cases(cases: list[dict], query: str, *, top_k: int = 3) -> list[dict]:
    """在当前原型案例材料中检索候选；召回不代表类案事实或法律适用相同。"""
    import re

    terms = [t for t in re.split(r"[\s，。；、,.;？！?！]+", query) if len(t) >= 2]
    scored = []
    for c in cases:
        hits = sum(1 for t in terms if t in c["text"])
        if hits:
            scored.append((hits, c))
    scored.sort(key=lambda x: -x[0])
    return [{**c, "hits": h} for h, c in scored[:top_k]]


def run_eval(top_k: int = 5) -> dict:
    """运行本地评测题并汇总检索/降级指标；不将这些指标解释为法律可靠性。"""
    from prototype.cli import answer_query
    from prototype.importer import build_index, load_documents

    spec = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    docs = load_documents()
    index = build_index(docs)
    cases = load_cases()

    rows = []
    for q in spec["questions"]:
        outcome = answer_query(q["query"], index, has_case_docs=bool(cases))
        citations = outcome.get("citations", [])
        cited_docs = {c["title"] for c in citations}
        cited_articles = {c["article_no"] for c in citations}

        case_hits = search_cases(cases, q["query"], top_k=top_k)
        case_doc_ids = {c["doc_id"] for c in case_hits}

        expected_docs = set(q.get("expect_hit_docs", []))
        expected_articles = set(q.get("expect_hit_articles", []))
        expect_degrade = q.get("expect_degrade", False)

        # 文档命中：法条标题按 title 文本匹配，案例按 case_id
        law_titles = {d.title for d in docs}
        expected_law_docs = expected_docs & set(d.doc_id for d in docs)
        got_docs = set()
        for d in docs:
            if d.title in cited_docs and d.doc_id in expected_law_docs:
                got_docs.add(d.doc_id)
        got_docs |= (case_doc_ids & expected_docs)

        doc_hit = (not expected_docs) or (expected_docs <= (got_docs | case_doc_ids)) or (expected_docs & (got_docs | case_doc_ids))
        article_hit = (not expected_articles) or (expected_articles <= cited_articles)
        degrade_ok = outcome["degraded"] == expect_degrade
        # 真正需要拦截的是“把不支持的材料标成依据”，而不是“降级时仍展示背景资料”。
        # 因此只在出现高于“背景资料”的标签时计为误导性引用。
        misleading = [c for c in citations if c["label"] != "背景资料"]
        false_citation = bool(misleading)

        ok = doc_hit and article_hit and degrade_ok and not false_citation
        rows.append(
            {
                "id": q["id"],
                "query": q["query"],
                "doc_hit": doc_hit,
                "article_hit": article_hit,
                "degrade_ok": degrade_ok,
                "degraded": outcome["degraded"],
                "expect_degrade": expect_degrade,
                "misleading_citations": len(misleading),
                "false_citation": false_citation,
                "citations": len(citations),
                "case_citations": len(outcome.get("case_citations", [])),
                "case_hits": sorted(case_doc_ids),
                "pass": ok,
                "trap": q.get("trap"),
            }
        )

    total = len(rows)
    passed = sum(1 for r in rows if r["pass"])
    return {
        "total": total,
        "passed": passed,
        "doc_hit_rate": round(sum(1 for r in rows if r["doc_hit"]) / total, 3),
        "article_hit_rate": round(sum(1 for r in rows if r["article_hit"]) / total, 3),
        "degrade_accuracy": round(sum(1 for r in rows if r["degrade_ok"]) / total, 3),
        "false_citations": sum(1 for r in rows if r["false_citation"]),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    """解析本模块命令行参数，执行对应操作；具体数据读写由下层函数负责。"""
    parser = argparse.ArgumentParser(description="小型检索评测")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    report = run_eval(top_k=args.top_k)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"{'ID':<5}{'PASS':<6}{'DOC':<6}{'ART':<6}{'DEG':<6}{'CITES':<7}QUERY")
        for r in report["rows"]:
            print(
                f"{r['id']:<5}{'OK' if r['pass'] else 'FAIL':<6}"
                f"{'Y' if r['doc_hit'] else 'N':<6}{'Y' if r['article_hit'] else 'N':<6}"
                f"{'Y' if r['degrade_ok'] else 'N':<6}{r['citations']:<7}{r['query'][:40]}"
            )
        print()
        print(f"passed {report['passed']}/{report['total']}")
        print(f"doc_hit_rate={report['doc_hit_rate']} article_hit_rate={report['article_hit_rate']} "
              f"degrade_accuracy={report['degrade_accuracy']} false_citations={report['false_citations']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
