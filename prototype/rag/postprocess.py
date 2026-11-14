"""后处理：模型输出解析、引用卡片生成、疑似问题标记。

## 输出设计（2026-10-19 调整）

模型**不再**靠自由文本同时完成“分析 + 抄原文 + 编号绑定”。改为：

1. 模型输出四段正文及材料声明；合法且唯一绑定的正文引用可补齐有效声明，原始回答不变；
2. **引用卡片由程序从对应材料原文生成**，并显式标记“系统附列原文”；
3. 程序附列的原文**不算模型逐字引用成功**，两者分开记录。

## 判定语义

- `blocking`：可以自动判定的硬错误（编造编号、无材料却有编号、伪示例引用）。
- `suspected`：**疑似、需人工复核**，不自动阻断。自相矛盾检查属于此类。

## 不可放宽

- 失败答案原样保留：`answer == raw_answer`、`modified == False`；
- 不删除编号后重新判定为通过；
- 程序附列的原文不进入“模型引用通过率”。

只依赖标准库。
"""

from __future__ import annotations

import re

# 编号识别统一复用 citation_check，避免两套正则漂移
from prototype.rag.citation_check import (  # noqa: E402
    BARE_CITATION_RE,
    BRACKET_CITATION_RE,
    USED_IDS_RE,
    find_citations as _find_citations,
    parse_used_ids as _parse_used_ids,
    body_material_ids,
)

CITATION_RE = BRACKET_CITATION_RE
SECTION_RE = re.compile(r"^#{1,4}\s*(.+?)\s*$")

# 可以自动判定的硬错误
BLOCKING_TYPES = {
    "NO_MATERIAL_HAS_CITATION",
    "UNKNOWN_ID",
    "FAKE_EXAMPLE_CITATION",
}

# 疑似问题：只提示人工复核，不自动阻断
SUSPECTED_TYPES = {
    "SUSPECTED_SELF_CONTRADICTION",
    "MISSING_USED_IDS",
    "ID_CITED_BUT_NOT_DECLARED",
    "DEADLINE_NO_SUBJECT",
    "DEADLINE_NO_TRIGGER",
    "DEADLINE_NO_SOURCE",
}


def split_sections(answer: str) -> dict[str, str]:
    """按标题拆分回答，供后处理检查指定段落使用。"""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in (answer or "").splitlines():
        m = SECTION_RE.match(line.strip())
        if m:
            current = m.group(1).strip()
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def find_citations(answer: str) -> list[str]:
    """统一提取回答正文中的材料编号，不把USED_IDS声明本身当作正文引用。"""
    return _find_citations(answer)


def parse_used_ids(answer: str):
    """解析模型声明的已用材料编号，区分缺少声明与明确未使用材料。"""
    return _parse_used_ids(answer)


def detect_fake_example_citation(answer: str) -> bool:
    """检查伪示例引用标记，防止无材料时靠示例措辞保留假编号。"""
    markers = ("示例引用", "此处为示例", "仅作示例", "示例编号", "假设引用")
    text = answer or ""
    return any(m in text for m in markers) and bool(CITATION_RE.search(text))


def detect_suspected_contradiction(answer: str) -> dict:
    """**疑似**自相矛盾（不自动阻断）。

    只作为人工复核线索：处理方向含确定表述，且不确定性段说信息不足。
    合法的“先讲一般规则、再说不适用于本案”也会命中，因此不能自动判错。
    """
    sections = split_sections(answer)
    direction = sections.get("处理方向", "")
    uncertainty = sections.get("不确定性", "")

    determinate = any(
        k in direction for k in ("应当认定为", "可以认定为", "符合工伤认定条件", "即可认定", "构成工伤")
    )
    insufficient = any(
        k in uncertainty
        for k in (
            "不足以支持",
            "无法判断",
            "缺少",
            "未提供",
            "未能提供",
            "尚不能确定",
            "缺乏",
            "无法确定",
        )
    )
    hit = determinate and insufficient
    return {
        "hit": hit,
        "determinate_phrase": determinate,
        "insufficient_phrase": insufficient,
        "detail": (
            "疑似自相矛盾：处理方向使用确定表述，不确定性段又说信息不足；"
            "也可能是“先讲一般规则再说明本案不适用”，需人工复核"
        )
        if hit
        else "",
        "needs_review": hit,
    }


def detect_citation_in_section(answer: str, section_name: str) -> bool:
    """检查指定回答段落是否出现材料编号，供分区规则判断。"""
    sections = split_sections(answer)
    return bool(CITATION_RE.search(sections.get(section_name, "")))


def build_citation_cards(used_ids: list[str], materials: list[dict]) -> list[dict]:
    """由程序生成引用卡片：正文来自材料原文，**标记为系统附列**。

    这些卡片不是模型逐字引用，不得计入模型“引用通过率”。
    """
    by_id = {m["material_id"]: m for m in materials}
    cards = []
    for mid in used_ids:
        material = by_id.get(mid)
        if not material:
            cards.append(
                {
                    "material_id": mid,
                    "origin": "SYSTEM_APPENDED",
                    "available": False,
                    "note": f"材料 {mid} 不存在，无法附列原文",
                }
            )
            continue
        cards.append(
            {
                "material_id": mid,
                "origin": "SYSTEM_APPENDED",
                "available": True,
                "title": material.get("title", ""),
                "locator": material.get("locator", ""),
                "version_id": material.get("version_id", ""),
                "source_url": material.get("source_url", ""),
                "content_scope": material.get("content_scope", "FULL_TEXT"),
                "text": material.get("text", ""),
                "label": "系统附列原文",
                "note": "由程序从材料原文附加，不是模型逐字引用",
            }
        )
    return cards


def postprocess(answer: str, materials: list[dict]) -> dict:
    """对一条生成结果做后处理判定。

    返回字段：
    - answer / raw_answer：恒等，不做静默修正
    - model_cited_ids：模型中出现的编号
    - used_ids：模型声明与安全补齐后的有效编号；original_used_ids保留原始解析
    - declaration_repair / normalized_answer：补齐审计与仅供后续核验的文本，不覆盖raw
    - citation_cards：程序生成的附列原文卡片（origin=SYSTEM_APPENDED）
    - blocking / blocking_issues：硬错误
    - suspected / suspected_issues：疑似、需人工复核
    - displayable_as_verified：无硬错误才放行
    - model_verbatim_citation_verified：**恒为 False**，因为本设计不再要求模型抄原文
    """
    provided = {m["material_id"] for m in materials}
    cited = find_citations(answer)
    cited_unique = sorted(set(cited), key=lambda s: int(s[1:]))
    # 检查所有声明行，防止末尾假编号或重复声明绕过正文编号检查。
    declared_ids = parse_used_ids(answer) or []
    unknown = sorted({c for c in cited + declared_ids if c not in provided}, key=lambda s: int(s[1:]))

    used_ids = parse_used_ids(answer)
    blocking = []
    suspected = []
    from prototype.rag.grounding import article_grounding_issues
    blocking.extend(article_grounding_issues(answer, materials))
    original_used_ids = used_ids
    body_ids = body_material_ids(answer, materials)
    # 先检查原始编号/法条，再补可唯一绑定的正文证据；未知编号不能靠删除声明或自愈洗白。
    repaired = bool(body_ids and not unknown and not blocking and not detect_fake_example_citation(answer)
                    and (used_ids is None or not set(body_ids) <= set(used_ids)))
    if repaired:
        used_ids = sorted(set(used_ids or []) | set(body_ids), key=lambda mid: int(mid[1:]))
    normalized_answer = (USED_IDS_RE.sub('', answer).rstrip() + '\nUSED_IDS: ' + ' '.join(used_ids)) if repaired else answer

    if not materials and cited:
        blocking.append(
            {
                "type": "NO_MATERIAL_HAS_CITATION",
                "detail": f"本轮未提供材料，但回答出现编号：{', '.join(cited_unique)}",
            }
        )

    if unknown:
        blocking.append({"type": "UNKNOWN_ID", "detail": f"出现未提供的编号：{', '.join(unknown)}"})

    if detect_fake_example_citation(answer):
        blocking.append(
            {"type": "FAKE_EXAMPLE_CITATION", "detail": "以“示例引用/此处为示例”等方式使用编号"}
        )

    contradiction = detect_suspected_contradiction(answer)
    if contradiction["hit"]:
        suspected.append({"type": "SUSPECTED_SELF_CONTRADICTION", "detail": contradiction["detail"]})

    if materials and used_ids is None:
        suspected.append(
            {
                "type": "MISSING_USED_IDS",
                "detail": "未找到 USED_IDS 行，无法确认模型声明使用哪些材料",
            }
        )

    if used_ids is not None:
        declared = set(used_ids)
        cited_set = set(cited_unique)
        extra = sorted(cited_set - declared, key=lambda s: int(s[1:]))
        if extra:
            suspected.append(
                {
                    "type": "ID_CITED_BUT_NOT_DECLARED",
                    "detail": f"正文出现但未在 USED_IDS 声明：{', '.join(extra)}",
                }
            )

    # 期限提示：主体 / 起算 / 依据编号（线索检测，需人工复核）
    from prototype.rag.deadline_check import check_deadline_notice

    deadline = check_deadline_notice(answer)
    for issue in deadline["issues"]:
        suspected.append({"type": issue["type"], "detail": issue["detail"]})

    # 卡片生成：优先用声明的 USED_IDS；缺失时回退到正文中出现的合法编号，
    # 并在卡片上标记 fallback，便于区分“模型声明”与“系统推断”。
    if used_ids is not None:
        card_ids = used_ids
        card_source = "BODY_REPAIRED" if repaired else "USED_IDS"
    else:
        card_ids = [c for c in cited_unique if c in provided]
        card_source = "BODY_FALLBACK"

    cards = build_citation_cards(card_ids, materials)
    for card in cards:
        card["id_source"] = card_source

    return {
        "answer": answer,
        "raw_answer": answer,
        "modified": False,
        "normalized_answer": normalized_answer,
        "original_used_ids": original_used_ids,
        "declaration_repair": {"repaired": repaired, "original_used_ids": original_used_ids,
                               "body_ids": body_ids, "effective_used_ids": used_ids},
        "model_cited_ids": cited_unique,
        "used_ids": used_ids,
        "card_id_source": card_source,
        "citation_cards": cards,
        "deadline_check": deadline,
        "blocking_issues": blocking,
        "suspected_issues": suspected,
        "blocking": bool(blocking),
        "suspected": bool(suspected),
        "displayable_as_verified": not blocking,
        "model_verbatim_citation_verified": False,
        "note": (
            "失败答案原样保留；程序附列原文（origin=SYSTEM_APPENDED）不计入模型引用通过率"
        ),
    }
