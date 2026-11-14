"""命令行入口：导入、自检、检索（含真实降级链路）、导出首版数据。

命令：
  python -m prototype.cli import              # 解析原文 → data/normalized/*.json
  python -m prototype.cli check               # Schema 校验导入结果
  python -m prototype.cli search <query>      # 用户链路：检索 → 相关性 → 降级 → 引用 → 覆盖声明
  python -m prototype.cli self-test           # 自检（含 CLI 链路级降级）
  python -m prototype.cli audit-capture       # 校验采集记录与原文快照一致

边界说明（务必保留）：
- evaluate_support() 只是**词面相关性检查**，不是法律支持性核验；
- 真正能否支持结论，需逐字核验 + 版本适用 + 依据资格 + 事实要件匹配，本原型均未实现；
- 因此任何引用最多标为“背景资料”，不会生成“本案处置法定依据”。
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

from prototype.importer import (
    DATA_DIR,
    LABEL_BACKGROUND,
    STATUTORY_LABEL_ENABLED,
    build_index,
    coverage_statement,
    load_capture_manifest,
    load_documents,
    render_citation,
    search,
)

NORMALIZED_DIR = DATA_DIR / "normalized"
SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schema"

# 尚未建立期限/程序规则的领域（与文档 pending 列表对应，均为“未核验”）
UNVERIFIED_DOMAINS = {
    "criminal-limitation": {
        "keywords": ("追诉时效", "追诉期限", "刑事时效", "羁押期限", "上诉期限", "抗诉期限"),
        "topic": "刑事追诉时效与羁押/上诉期限",
    },
    "administrative-deadline": {
        "keywords": ("行政复议期限", "行政诉讼期限", "起诉期限", "行政复议申请期"),
        "topic": "行政复议申请期与行政诉讼起诉期",
    },
}


def cmd_import() -> int:
    """执行资料导入命令，把本模块处理结果写入项目数据目录。"""
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_documents()
    for doc in docs:
        payload = doc.to_schema_dict()
        out = NORMALIZED_DIR / f"{doc.doc_id}.{doc.version_id}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"imported {doc.doc_id} version={doc.version_id} articles={len(doc.articles)} "
            f"retrieved_at={doc.retrieved_at} imported_at={doc.imported_at}"
        )
    return 0


def cmd_check() -> int:
    """执行原型数据格式检查并报告结果；格式通过不等于法律结论正确。"""
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError:
        print("jsonschema 未安装：请设置 PYTHONPATH 指向工作区依赖目录")
        return 2

    schema = json.loads((SCHEMA_DIR / "legal_document.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    failures = 0
    for path in sorted(NORMALIZED_DIR.glob("*.json")):
        instance = json.loads(path.read_text(encoding="utf-8"))
        errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
        if errors:
            failures += 1
            print(f"FAIL {path.name}")
            for e in errors[:6]:
                print("   ", list(e.path), e.message)
        else:
            print(f"PASS {path.name}")
    print("SCHEMA_OK" if failures == 0 else f"SCHEMA_FAILED ({failures})")
    return 0 if failures == 0 else 1


def cmd_audit_capture() -> int:
    """校验采集记录：快照哈希一致；法规需有规范化结果且 retrieved_at 一致；案例发布稿需有案例导入结果。"""
    from prototype.importer import capture_for

    manifest = load_capture_manifest()
    cases_dir = DATA_DIR / "cases"
    problems = []
    checked_laws = 0
    checked_cases = 0

    for doc_id, capture in manifest.items():
        try:
            capture_for(doc_id, capture["raw_file"])
        except RuntimeError as exc:
            problems.append(f"{doc_id}: {exc}")
            continue

        is_case_release = capture.get("content_scope") == "RELEASE_SUMMARY_ONLY"

        if is_case_release:
            imported = sorted(cases_dir.glob("CASE-LABOR-TYPICAL-*.json")) if cases_dir.exists() else []
            if not imported:
                problems.append(f"{doc_id}: 缺少案例导入结果")
                continue
            payload = json.loads(imported[0].read_text(encoding="utf-8"))
            if payload["retrieved_at"] != capture["retrieved_at"]:
                problems.append(
                    f"{doc_id}: retrieved_at 不一致（案例={payload['retrieved_at']} 采集={capture['retrieved_at']}）"
                )
            for path in imported:
                case = json.loads(path.read_text(encoding="utf-8"))
                if case.get("content_scope") != "RELEASE_SUMMARY_ONLY":
                    problems.append(f"{path.name}: content_scope 应为 RELEASE_SUMMARY_ONLY")
                if case.get("full_text_available") is not False:
                    problems.append(f"{path.name}: full_text_available 应为 false")
            checked_cases += len(imported)
            continue

        normalized_hits = list(NORMALIZED_DIR.glob(f"{doc_id}.*.json"))
        if not normalized_hits:
            problems.append(f"{doc_id}: 缺少规范化结果")
            continue
        payload = json.loads(normalized_hits[0].read_text(encoding="utf-8"))
        if payload["retrieved_at"] != capture["retrieved_at"]:
            problems.append(
                f"{doc_id}: retrieved_at 不一致（规范化={payload['retrieved_at']} 采集={capture['retrieved_at']}）"
            )
        if payload.get("capture_manifest_ref") != "data/raw/capture_manifest.json":
            problems.append(f"{doc_id}: 缺少采集记录引用")
        else:
            checked_laws += 1
    if problems:
        for p in problems:
            print("FAIL", p)
        print("CAPTURE_AUDIT_FAILED")
        return 1
    print(f"CAPTURE_AUDIT_OK (法规 {checked_laws} + 案例 {checked_cases})")
    return 0


# ---------------------------------------------------------------- 用户链路


def detect_unverified_domain(query: str) -> dict | None:
    """以当前原型关键词规则提示未验证领域；不是完整领域分类器。"""
    for key, spec in UNVERIFIED_DOMAINS.items():
        if any(k in query for k in spec["keywords"]):
            return {"key": key, "topic": spec["topic"]}
    return None


def check_lexical_relevance(query: str, hits: list[dict]) -> dict:
    """**词面相关性检查**（不是法律支持性核验）。

    仅判断用户问题中的主题词是否出现在命中条款原文中。
    两级判定，均属词面方法：
    1) 整词命中：query 去掉泛化词后按标点切分，词长>=2，需出现在条款文本中；
    2) 二元组重叠：整词都未命中时，用字符二元组重叠数 >= 3 作为“疑似相关”。
       该级只用于“是否需要降级”的保守判断，且输出标记 match_mode=bigram，
       仍不等于法律支持；对外展示时一律标“背景资料”。
    局限：不做同义词、不做语义匹配；词面命中也不代表法律上支持。
    """
    generic = {
        "可以", "能", "吗", "怎么办", "是否", "如何", "怎么", "能主张", "主张",
        "有没有", "什么", "多少", "多少天", "多久", "需要", "应该", "请问",
        "什么时候", "什么时间", "自什么时候", "是多久", "多长时间",
    }
    topic = query
    for g in sorted(generic, key=len, reverse=True):
        topic = topic.replace(g, " ")
    terms = [t for t in re.split(r"[\s，。；、,.;？！?！（）()]+", topic) if len(t) >= 2]

    matched = [h for h in hits if any(t in h["text"] for t in terms)]
    if matched:
        return {
            "check_type": "LEXICAL_RELEVANCE_ONLY",
            "match_mode": "exact",
            "terms": terms,
            "matched": matched,
            "is_legal_support": False,
            "note": "词面命中不等于法律支持；未做版本适用与事实匹配。",
        }

    from prototype.importer import _bigrams

    query_grams = _bigrams(query)
    gram_matched = [
        h for h in hits
        if len(query_grams & _bigrams(h["text"])) >= 3
    ]
    return {
        "check_type": "LEXICAL_RELEVANCE_ONLY",
        "match_mode": "bigram" if gram_matched else "none",
        "terms": terms,
        "matched": gram_matched,
        "is_legal_support": False,
        "note": "仅二元组重叠，属疑似相关，不是法律支持。",
    }


def answer_query(query: str, index: list[dict], *, has_case_docs: bool = False) -> dict:
    """用户实际链路：检索 → 未验证领域判断 → 词面相关性 → 降级 → 引用 → 覆盖声明。"""
    result: dict = {
        "query": query,
        "coverage": coverage_statement(has_case_docs),
        "citations": [],
        "case_citations": [],
        "case_hits": 0,
        "degraded": False,
        "degrade_reasons": [],
        "statutory_label_enabled": STATUTORY_LABEL_ENABLED,
    }

    unverified = detect_unverified_domain(query)
    if unverified:
        result["degraded"] = True
        result["degrade_reasons"].append(
            f"{unverified['topic']}所属领域尚未完成规则核验：不输出确定期限或确定程序"
        )
        result["answer_notice"] = (
            f"{unverified['topic']}属于尚未核验的领域，本次不给出确定期限或确定程序。"
            "可先整理时间线、主体关系与现有材料；如需确定期间，请以官方文本或专业人士意见为准。"
        )
        return result

    hits = search(index, query)
    cases = load_case_index()
    case_hits = search(cases, query, top_k=3) if cases else []
    result["hits"] = len(hits)
    result["case_hits"] = len(case_hits)

    if not hits and not case_hits:
        result["degraded"] = True
        result["degrade_reasons"].append("检索未命中：不得编造出处或凭印象作答")
        result["answer_notice"] = "本次检索未找到可核验的条款或案例，不提供法律依据层面的结论；可补充事实或换用更具体的表述。"
        return result

    relevance = check_lexical_relevance(query, hits) if hits else {"matched": [], "terms": [], "match_mode": "none"}
    result["relevance"] = {
        "check_type": "LEXICAL_RELEVANCE_ONLY",
        "match_mode": relevance.get("match_mode", "none"),
        "terms": relevance.get("terms", []),
        "matched": len(relevance["matched"]),
        "is_legal_support": False,
    }

    if hits and not relevance["matched"] and not case_hits:
        result["degraded"] = True
        result["degrade_reasons"].append("仅有检索命中，但原文未出现问题主题词：不能作为该问题的依据")
        result["answer_notice"] = "检索到相关条款，但原文未直接涉及所问主题，不能作为该问题的依据；请补充事实或咨询专业人士。"
        result["citations"] = [render_citation(row, raw_text=_raw_for(row)) for row in hits]
        return result

    # 法条引用（带逐字核验）
    rows = relevance["matched"] if relevance["matched"] else hits
    result["citations"] = [render_citation(row, raw_text=_raw_for(row)) for row in rows]

    # 案例引用：只作为“类案参考”，且明确标注仅发布稿摘要
    for case in case_hits:
        result["case_citations"].append(
            {
                "label": "类案参考",
                "case_id": case["case_id"],
                "title": case["title"],
                "content_scope": case["content_scope"],
                "full_text_available": case["full_text_available"],
                "source_url": case["source_url"],
                "locator": case["locator"],
                "scope_note": (
                    "仅最高人民法院发布的典型案例稿件摘要（基本案情/裁判结果/典型意义），"
                    "不是判决书全文；不得当作全文引用。"
                ),
            }
        )

    if relevance["matched"]:
        result["degrade_reasons"].append("即使词面命中，也因版本适用/依据资格/事实匹配未完成，只能标为背景资料")
        result["answer_notice"] = "以下条款与问题在词面上相关，可作背景参考；是否支持具体结论仍需逐字核验与事实核对。"
    else:
        # 无法条主题词命中：即使有案例，也只能算部分降级——没有法律依据
        result["degraded"] = True
        result["degrade_reasons"].append("法条未命中主题词：本次无法提供法律依据，案例仅作类案参考")
        result["answer_notice"] = "未检索到与问题直接相关的法条，以下仅为类似案例参考，不能替代法律依据，也不能据此得出确定结论。"
    return result


def _raw_for(row: dict) -> str | None:
    """取该文档的原文正文，用于引用片段逐字核验。失败返回 None（不阻断展示）。"""
    try:
        from prototype.verifier import load_raw

        return load_raw(row["doc_id"])
    except Exception:
        return None


def cmd_search(query: str, *, as_json: bool = False) -> int:
    """执行原型检索命令并展示引用/覆盖与降级结果，不绕过主回答流程。"""
    docs = load_documents()
    index = build_index(docs)
    outcome = answer_query(query, index, has_case_docs=bool(load_case_index()))

    if as_json:
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
        return 0

    print(f"query={query!r} hits={outcome.get('hits', 0)} case_hits={outcome.get('case_hits', 0)} degraded={outcome['degraded']}")
    if outcome["degrade_reasons"]:
        for reason in outcome["degrade_reasons"]:
            print(f"  ! {reason}")
    if outcome.get("answer_notice"):
        print(f"  > {outcome['answer_notice']}")
    for card in outcome["citations"]:
        qv = card.get("quote_verification", {})
        qv_note = ""
        if qv.get("checked"):
            qv_note = " 逐字核验=通过" if qv.get("verified") else f" 逐字核验=失败({qv.get('reason')})"
        print(f"- [{card['label']}] {card['title']} {card['article_no']} v{card['version_id']}{qv_note}")
        print(f"  locator={card['locator']} url={card['source_url']}")
        print(f"  text={card['text']}...")
    for case in outcome.get("case_citations", []):
        print(f"- [{case['label']}] 案例 {case['case_id']} {case['title'][:40]}")
        print(f"  scope={case['content_scope']} url={case['source_url']}")
        print(f"  note={case['scope_note']}")
    print(f"覆盖声明：{outcome['coverage']}")
    return 0


def load_case_index() -> list[dict]:
    """加载案例发布稿为可检索条目（仅发布稿摘要）。"""
    cases_dir = DATA_DIR / "cases"
    if not cases_dir.exists():
        return []
    rows = []
    for path in sorted(cases_dir.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        body = "\n".join(s["text"] for s in case.get("sections", []) if s.get("text"))
        rows.append(
            {
                "case_id": case["case_id"],
                "title": case["title"],
                "text": case["title"] + "\n" + body,
                "content_scope": case.get("content_scope", "UNKNOWN"),
                "full_text_available": case.get("full_text_available", False),
                "source_url": case["source_url"],
                "locator": case.get("citation_card", {}).get("locator"),
            }
        )
    return rows


# ---------------------------------------------------------------- 自检


def _scenario(name: str, ok: bool, detail: str) -> tuple[str, bool, str]:
    """准备原型自检场景；仅用于核对链路行为，不作为独立法律评测。"""
    print(("PASS" if ok else "FAIL") + f" {name}: {detail}")
    return name, ok, detail


def _run_cli(argv: list[str]) -> tuple[int, str]:
    """在进程内执行 CLI 命令，捕获输出，用于链路级测试。"""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(argv)
    return code, buffer.getvalue()


def cmd_self_test() -> int:
    """运行原型内置自检，汇总已固定的检索与降级场景。"""
    results = []
    docs = {d.doc_id: d for d in load_documents()}
    index = build_index(list(docs.values()))

    # --- 数据层面 ---
    labor = docs["LAW-LABOR-INTERP-2"]
    first = labor.articles[0]
    results.append(_scenario(
        "原文可定位",
        first.article_no == "第一条" and first.locator.startswith("L"),
        f"{labor.doc_id} {first.article_no} -> {first.locator}",
    ))
    results.append(_scenario("条款数量-解释二", len(labor.articles) == 21, f"articles={len(labor.articles)}"))
    injury = docs["LAW-WORK-INJURY-REGULATION"]
    results.append(_scenario("条款数量-工伤保险条例", len(injury.articles) == 67, f"articles={len(injury.articles)}"))

    # --- 采集时间不被导入时间覆盖 ---
    manifest = load_capture_manifest()
    results.append(_scenario(
        "采集时间保留",
        labor.retrieved_at == manifest["LAW-LABOR-INTERP-2"]["retrieved_at"]
        and labor.imported_at is not None
        and labor.imported_at != labor.retrieved_at,
        f"retrieved_at={labor.retrieved_at} imported_at={labor.imported_at}",
    ))

    # --- 引用标签不可绕过 ---
    hits = search(index, "工伤")
    card = render_citation(hits[0])
    results.append(_scenario(
        "非法定依据标签",
        card["label"] == LABEL_BACKGROUND,
        f"label={card['label']} reasons={card['label_reasons'][:1]}",
    ))
    forged = {"verdict": "ELIGIBLE", "verified_verbatim": True, "version_confirmed": True,
              "authority_eligible": True, "facts_matched": True}
    results.append(_scenario(
        "伪造核验也不出法定依据",
        render_citation(hits[0], eligibility=forged)["label"] == LABEL_BACKGROUND,
        "STATUTORY_LABEL_ENABLED=False，任何 eligibility 都不生成该标签",
    ))
    results.append(_scenario(
        "render_citation 不接受调用方指定标签",
        _signature_has_no_label(),
        "签名已移除 label 参数",
    ))

    # --- 用户链路降级（走 CLI 入口，不是只调辅助函数） ---
    code, out = _run_cli(["search", "区块链智能合约"])
    results.append(_scenario(
        "链路降级-无命中",
        code == 0 and "检索未命中" in out and "degraded=True" in out,
        out.splitlines()[0] if out else "",
    ))

    code, out = _run_cli(["search", "刑事追诉时效是多久"])
    results.append(_scenario(
        "链路降级-未验证领域",
        code == 0 and "degraded=True" in out and "不给出确定期限" in out,
        "未验证领域在检索前被拦截",
    ))

    code, out = _run_cli(["search", "可以 精神损害"])
    results.append(_scenario(
        "链路降级-命中但不支持",
        code == 0
        and "degraded=True" in out
        and ("无法提供法律依据" in out or "仅有检索命中" in out),
        "命中条款但原文无主题词，走降级",
    ))

    code, out = _run_cli(["search", "竞业限制"])
    results.append(_scenario(
        "链路-词面相关仍标背景资料",
        code == 0 and "[背景资料]" in out and "本案处置法定依据" not in out,
        "词面命中也不升级标签",
    ))

    code, out = _run_cli(["search", "工伤"])
    results.append(_scenario(
        "链路-覆盖声明",
        code == 0 and "覆盖声明" in out and "案例" in out,
        "用户链路输出覆盖声明（含案例覆盖范围）",
    ))

    # --- 案例发布稿链路 ---
    code, out = _run_cli(["search", "转包给个人 招用的人受伤 谁承担工伤保险责任"])
    results.append(_scenario(
        "链路-案例类案参考",
        code == 0 and "类案参考" in out and "不是判决书全文" in out,
        "案例仅作类案参考并声明非全文",
    ))

    code, out = _run_cli(["search", "工伤"])
    results.append(_scenario(
        "链路-引用逐字核验",
        code == 0 and "逐字核验=通过" in out,
        "引用卡片附带逐字核验结果",
    ))

    failed = [r for r in results if not r[1]]
    print("SELF_TEST_OK" if not failed else f"SELF_TEST_FAILED ({len(failed)})")
    return 0 if not failed else 1


def _signature_has_no_label() -> bool:
    """检查引用渲染接口不能接收外部指定标签，避免绕过资格判定。"""
    import inspect

    return "label" not in inspect.signature(render_citation).parameters


def main(argv: list[str] | None = None) -> int:
    """解析本模块命令行参数，执行对应操作；具体数据读写由下层函数负责。"""
    parser = argparse.ArgumentParser(description="OpenLawGuide 首批原型")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("import")
    sub.add_parser("check")
    sub.add_parser("audit-capture")
    sub.add_parser("self-test")
    p_search = sub.add_parser("search")
    p_search.add_argument("query")
    p_search.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    if args.cmd == "import":
        return cmd_import()
    if args.cmd == "check":
        return cmd_check()
    if args.cmd == "audit-capture":
        return cmd_audit_capture()
    if args.cmd == "self-test":
        return cmd_self_test()
    if args.cmd == "search":
        return cmd_search(args.query, as_json=getattr(args, "as_json", False))
    return 2


if __name__ == "__main__":
    sys.exit(main())
