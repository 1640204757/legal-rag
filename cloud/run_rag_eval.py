"""自包含 RAG 评测脚本（可直接上传到 Colab/云端运行）。

设计目标：
- 只依赖标准库 + 可选 torch/transformers；不 import 本项目其他模块，避免上传整个仓库。
- 输入两个 JSON：eval_tasks.json（含每组题的材料与快照）与 snapshots.json（原文快照）。
- 输出：逐题结果、汇总指标、耗时；支持 --mock 干跑（不加载模型也能验证流程）。

用法：
    python run_rag_eval.py --tasks eval_tasks.json --snapshots snapshots.json --mock
    python run_rag_eval.py --tasks eval_tasks.json --snapshots snapshots.json \
        --model Qwen/Qwen3-8B --out results.json

两种模式对同一批题分别跑：
    baseline   不提供检索材料
    rag        提供检索材料
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

# ================================================================
# 以下三段由本地模块同步生成，勿在本文件手改
# ================================================================

# ---------------- 引用检查（citation_check） ----------------
# 编号内允许空格：[M1] / [ M1 ] / [M 1]
CITATION_RE = re.compile(r"\[\s*M\s*(\d+)\s*\]")
QUOTE_PATTERNS = (
    re.compile(r"“([^”]{2,400})”"),
    re.compile(r"「([^」]{2,400})」"),
)
EXPLICIT_QUOTE_RE = re.compile(r"原文\s*[:：]\s*$")
SECTION_RE = re.compile(r"^#{1,4}\s*(.+?)\s*$")
USED_IDS_RE = re.compile(r"^\s*USED_IDS\s*[:：]\s*(.*)$", re.MULTILINE)

REQUIRED_SECTIONS = ("追问", "处理方向", "证据清单", "不确定性")

BINDING_WINDOW = 30
COMPACT_GAP = 8
EMPHASIS_MAX_LEN = 8

LEGACY_FAILING_TYPES = {
    "UNBOUND",
    "UNKNOWN_ID",
    "MISMATCH",
    "SNAPSHOT_ONLY",
    "OTHER_DOC",
    "NOT_FOUND",
    "SNAPSHOT_MISMATCH",
    "AMBIGUOUS_BINDING",
}

FAKE_MARKERS = ("示例引用", "此处为示例", "仅作示例", "示例编号", "假设引用")


def _tight(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def extract_citations(answer: str) -> list[int]:
    return sorted({int(m.group(1)) for m in CITATION_RE.finditer(answer or "")})


def find_citations(answer: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in CITATION_RE.finditer(answer or ""):
        key = f"M{m.group(1)}"
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def parse_used_ids(answer: str):
    """解析 USED_IDS 行；未出现返回 None，`无` 返回 []。"""
    m = USED_IDS_RE.search(answer or "")
    if not m:
        return None
    raw = m.group(1).strip()
    if not raw or raw in {"无", "none", "NONE", "-"}:
        return []
    ids = re.findall(r"M(\d+)", raw)
    seen, out = set(), []
    for n in ids:
        key = f"M{n}"
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def detect_fake_example(answer: str) -> bool:
    text = answer or ""
    return any(m in text for m in FAKE_MARKERS) and bool(CITATION_RE.search(text))


def parse_sections(answer: str) -> dict[str, str]:
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


def check_structure(answer: str) -> dict:
    sections = parse_sections(answer)
    missing = [s for s in REQUIRED_SECTIONS if s not in sections]
    return {
        "sections_present": [s for s in REQUIRED_SECTIONS if s in sections],
        "sections_missing": missing,
        "structure_ok": not missing,
    }


# ---------------------------------------------------------------- 编号级判定


def check_ids(answer: str, materials: list[dict]) -> dict:
    """编号级校验（主判定）。"""
    provided = {m["material_id"] for m in materials}
    cited = find_citations(answer)
    unknown_in_body = [c for c in cited if c not in provided]

    used_ids = parse_used_ids(answer)
    unknown_declared = [c for c in (used_ids or []) if c not in provided]
    unknown = sorted(set(unknown_in_body) | set(unknown_declared), key=lambda s: int(s[1:]))

    reasons = []
    if not materials:
        if cited:
            state = "NO_MATERIAL_HAS_CITATION"
            reasons.append(f"本轮无材料，但回答出现编号：{', '.join(cited)}")
        else:
            state = "NO_MATERIAL_CLEAN"
    elif detect_fake_example(answer):
        state = "FAKE_EXAMPLE"
        reasons.append("用“示例引用/此处为示例”等方式保留编号")
    elif unknown:
        state = "IDS_UNKNOWN"
        reasons.append(f"出现未提供的编号：{', '.join(unknown)}")
    elif used_ids is None:
        state = "IDS_MISSING"
        reasons.append("有材料但缺少 USED_IDS 声明行")
    elif not used_ids:
        state = "IDS_NONE_USED"
        reasons.append("声明未使用任何材料")
    else:
        state = "IDS_OK"

    verdict = {
        "NO_MATERIAL_CLEAN": "PASS",
        "NO_MATERIAL_HAS_CITATION": "FAIL",
        "IDS_OK": "PASS",
        "IDS_NONE_USED": "PASS_WITH_NOTE",
        "IDS_MISSING": "FAIL",
        "IDS_UNKNOWN": "FAIL",
        "FAKE_EXAMPLE": "FAIL",
    }[state]

    undeclared = [c for c in cited if c not in (used_ids or [])]
    if state == "IDS_OK" and undeclared:
        reasons.append(f"正文出现但未在 USED_IDS 声明：{', '.join(undeclared)}（需人工复核）")

    return {
        "citation_state": state,
        "verdict": verdict,
        "cited_ids": cited,
        "used_ids": used_ids,
        "unknown_ids": unknown,
        "undeclared_ids": undeclared,
        "reasons": reasons,
    }


# ---------------------------------------------------------------- 逐字核验（旧答案诊断）


def _paragraph_end(text: str, start: int) -> int:
    rest = text[start:]
    blank = re.search(r"\n\s*\n", rest)
    heading = re.search(r"\n#{1,4}\s", rest)
    ends = [m.start() for m in (blank, heading) if m]
    return start + (min(ends) if ends else len(rest))


def _quote_spans(answer: str):
    text = answer or ""
    spans = []
    for pattern in QUOTE_PATTERNS:
        for m in pattern.finditer(text):
            prefix = text[max(0, m.start() - 12) : m.start()]
            spans.append((m.start(), m.end(), m.group(1), bool(EXPLICIT_QUOTE_RE.search(prefix))))
    spans.sort(key=lambda x: x[0])
    return spans


def _unique(items):
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_quote_bindings(answer: str) -> list[dict]:
    text = answer or ""
    bindings, seen = [], set()
    for start, end, quote, explicit in _quote_spans(answer):
        tight = _tight(quote)
        if not tight or tight in seen:
            continue
        seen.add(tight)

        bound, binding = [], "none"
        if end >= 0:
            para_end = _paragraph_end(text, end)
            window_end = min(end + BINDING_WINDOW, para_end)
            nearest = CITATION_RE.search(text, end, window_end) if window_end > end else None
            if nearest and len(_tight(text[end : nearest.start()])) > COMPACT_GAP:
                nearest = None
            if nearest:
                compact_end = min(nearest.end() + COMPACT_GAP, para_end)
                nearby = _unique(f"M{m.group(1)}" for m in CITATION_RE.finditer(text, end, compact_end))
                if len(nearby) > 1:
                    bindings.append({"quote": quote, "tight": tight, "bound_ids": [],
                                     "binding": "ambiguous", "candidate_ids": nearby, "kind": "quote"})
                    continue
                bound = [f"M{nearest.group(1)}"]
                binding = "explicit" if explicit else "adjacent"
            if not bound:
                segment = text[end:para_end]
                ids = _unique(f"M{m.group(1)}" for m in CITATION_RE.finditer(segment))
                if len(ids) == 1:
                    bound, binding = ids, "paragraph"
                elif len(ids) > 1:
                    bindings.append({"quote": quote, "tight": tight, "bound_ids": [],
                                     "binding": "ambiguous", "candidate_ids": ids, "kind": "quote"})
                    continue

        kind = "quote" if bound else ("emphasis" if len(tight) < EMPHASIS_MAX_LEN else "quote")
        bindings.append({"quote": quote, "tight": tight, "bound_ids": bound,
                         "binding": binding, "kind": kind})
    return bindings


def _classify_quote(binding, provided, material_text, material_snapshot, snapshots) -> dict:
    quote, tight, bound_ids = binding["quote"], binding["tight"], binding["bound_ids"]
    base = {"quote": quote, "bound_ids": bound_ids, "binding": binding.get("binding")}

    if binding.get("binding") == "ambiguous":
        return {**base, "type": "AMBIGUOUS_BINDING", "detail": "同一处引文有多个候选编号"}
    if not bound_ids:
        return {**base, "type": "UNBOUND", "detail": "引文未绑定编号"}
    unknown = [mid for mid in bound_ids if mid not in provided]
    if unknown:
        return {**base, "type": "UNKNOWN_ID", "detail": f"编号未提供：{', '.join(unknown)}"}

    for mid in bound_ids:
        if tight in _tight(material_text.get(mid, "")):
            sid = material_snapshot.get(mid, "")
            ok = bool(sid) and tight in _tight(snapshots.get(sid, ""))
            return {**base, "type": "BOUND_OK" if ok else "SNAPSHOT_MISMATCH",
                    "detail": "与材料一致" if ok else "与材料一致但快照中找不到"}
    for mid in bound_ids:
        sid = material_snapshot.get(mid, "")
        if sid and tight in _tight(snapshots.get(sid, "")):
            return {**base, "type": "SNAPSHOT_ONLY", "detail": "来自快照中未提供的条款"}
    for other_id, text in material_text.items():
        if other_id not in bound_ids and tight in _tight(text):
            return {**base, "type": "MISMATCH", "detail": f"引文实际来自 {other_id}"}
    for sid, text in snapshots.items():
        if tight in _tight(text):
            return {**base, "type": "OTHER_DOC", "detail": f"来自其他文档 {sid}"}
    return {**base, "type": "NOT_FOUND", "detail": "快照中找不到（改写或编造）"}


def check_legacy_verbatim(answer: str, materials: list[dict], snapshots: dict[str, str]) -> dict:
    """逐字引文核验，仅用于诊断旧答案；不参与主 verdict。"""
    provided = {m["material_id"] for m in materials}
    material_text = {m["material_id"]: m.get("text", "") for m in materials}
    material_snapshot = {m["material_id"]: m.get("snapshot_id", "") for m in materials}

    bindings = [b for b in extract_quote_bindings(answer) if b.get("kind") != "emphasis"]
    emphasis = [b["quote"] for b in extract_quote_bindings(answer) if b.get("kind") == "emphasis"]
    details = [_classify_quote(b, provided, material_text, material_snapshot, snapshots) for b in bindings]
    failures = [d for d in details if d["type"] in LEGACY_FAILING_TYPES]

    if not details:
        state = "NO_QUOTE"
    elif not failures:
        state = "VERBATIM_OK"
    elif len(failures) == len(details):
        state = "VERBATIM_FAIL"
    else:
        state = "VERBATIM_PARTIAL"

    return {
        "state": state,
        "quotes_total": len(details),
        "quotes_verified": len(details) - len(failures),
        "quotes_failed": failures,
        "quote_details": details,
        "emphasis_quotes": emphasis,
        "note": "仅诊断用；模型按新设计不应自行抄条文，此结果不计入模型引用通过率",
    }


# ---------------------------------------------------------------- 统一入口


def check_answer(answer: str, materials: list[dict], snapshots: dict[str, str]) -> dict:
    """主检查：编号级判定 + 结构 + （若有引文）逐字诊断。"""
    ids = check_ids(answer, materials)
    structure = check_structure(answer)
    legacy = check_legacy_verbatim(answer, materials, snapshots)
    reasons = list(ids["reasons"])
    if legacy["state"] in {"VERBATIM_FAIL", "VERBATIM_PARTIAL"}:
        reasons.append(f"检测到 {legacy['quotes_total']} 处自行抄写的引文，其中 {len(legacy['quotes_failed'])} 处未通过（仅诊断）")

    return {
        **ids,
        "structure": structure,
        "structure_ok": structure["structure_ok"],
        "legacy_verbatim": legacy,
        "scope_note": (
            "只验证编号存在性与结构，不代表法律适用正确；"
            "程序附列原文不计入模型引用通过率。"
        ),
    }


# ---------------- 后处理（postprocess） ----------------
import re

CITATION_RE = re.compile(r"\[\s*M\s*(\d+)\s*\]")
SECTION_RE = re.compile(r"^#{1,4}\s*(.+?)\s*$")
USED_IDS_RE = re.compile(r"^\s*USED_IDS\s*[:：]\s*(.*)$", re.MULTILINE)

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


def find_citations(text: str) -> list[str]:
    return [f"M{m.group(1)}" for m in CITATION_RE.finditer(text or "")]


def parse_used_ids(answer: str) -> list[str] | None:
    """解析 `USED_IDS: M1, M2` 行；没有该行返回 None（区别于空列表）。"""
    m = USED_IDS_RE.search(answer or "")
    if not m:
        return None
    raw = m.group(1).strip()
    if not raw or raw in {"无", "none", "NONE", "-"}:
        return []
    ids = re.findall(r"M(\d+)", raw)
    seen, out = set(), []
    for n in ids:
        key = f"M{n}"
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def detect_fake_example_citation(answer: str) -> bool:
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
        k in uncertainty for k in ("不足以支持", "无法判断", "缺少", "未能提供", "尚不能确定")
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


def postprocess_answer(answer: str, materials: list[dict]) -> dict:
    """对一条生成结果做后处理判定。

    返回字段：
    - answer / raw_answer：恒等，不做静默修正
    - model_cited_ids：模型中出现的编号
    - used_ids：模型声明的使用编号（USED_IDS 行）
    - citation_cards：程序生成的附列原文卡片（origin=SYSTEM_APPENDED）
    - blocking / blocking_issues：硬错误
    - suspected / suspected_issues：疑似、需人工复核
    - displayable_as_verified：无硬错误才放行
    - model_verbatim_citation_verified：**恒为 False**，因为本设计不再要求模型抄原文
    """
    provided = {m["material_id"] for m in materials}
    cited = find_citations(answer)
    cited_unique = sorted(set(cited), key=lambda s: int(s[1:]))
    unknown = sorted({c for c in cited if c not in provided}, key=lambda s: int(s[1:]))

    used_ids = parse_used_ids(answer)
    blocking = []
    suspected = []

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
    
    deadline = check_deadline_notice(answer)
    for issue in deadline["issues"]:
        suspected.append({"type": issue["type"], "detail": issue["detail"]})

    # 卡片生成：优先用声明的 USED_IDS；缺失时回退到正文中出现的合法编号，
    # 并在卡片上标记 fallback，便于区分“模型声明”与“系统推断”。
    if used_ids is not None:
        card_ids = used_ids
        card_source = "USED_IDS"
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


# ---------------- 期限检查（deadline_check） ----------------
DEADLINE_RE = re.compile(r"(?:\d+\s*(?:日|天|个月|年)内|自[^，。；]{0,24}之日起\s*\d+\s*(?:日|天|个月|年))")
CITATION_RE = re.compile(r"\[\s*M\s*(\d+)\s*\]")

# 主体：必须能看出“谁”来申请/主张
SUBJECT_GROUPS = {
    "EMPLOYER": ("用人单位", "单位应当", "单位应", "所在单位"),
    "WORKER": ("职工", "劳动者", "本人", "近亲属", "工伤职工"),
}
RECEIVER_WORDS = ("社会保险行政部门", "劳动能力鉴定", "人民法院", "仲裁")  # 受理机关，不算申请主体

TRIGGER_WORDS = ("之日起", "发生之日", "诊断", "鉴定之日", "知道或者应当知道", "起算")

# 期限句前后的上下文窗口字符数
CONTEXT_WINDOW = 60


def _context(text: str, start: int, end: int) -> str:
    return text[max(0, start - CONTEXT_WINDOW) : min(len(text), end + CONTEXT_WINDOW)]


def extract_deadline_mentions(answer: str) -> list[str]:
    return [m.group(0) for m in DEADLINE_RE.finditer(answer or "")]


def analyze_deadline(answer: str) -> list[dict]:
    """逐处分析期限表述，返回每处的主体/起算/依据判定。"""
    text = answer or ""
    results = []
    for m in DEADLINE_RE.finditer(text):
        ctx = _context(text, m.start(), m.end())
        # 主体只看期限句窗口内，且排除受理机关
        subject_ctx = ctx
        for receiver in RECEIVER_WORDS:
            subject_ctx = subject_ctx.replace(receiver, "")
        subjects = [k for k, words in SUBJECT_GROUPS.items() if any(w in subject_ctx for w in words)]
        has_trigger = any(w in ctx for w in TRIGGER_WORDS)
        has_citation = bool(CITATION_RE.search(ctx))
        results.append(
            {
                "mention": m.group(0),
                "subjects": subjects,
                "has_trigger": has_trigger,
                "has_citation": has_citation,
                "context": ctx.strip(),
            }
        )
    return results


def check_deadline_notice(answer: str) -> dict:
    """检查每处期限是否具备主体、起算事实与依据编号。"""
    analyses = analyze_deadline(answer)
    if not analyses:
        return {
            "has_deadline": False,
            "mentions": [],
            "details": [],
            "issues": [],
            "needs_review": False,
            "note": "未出现期限表述",
        }

    issues = []
    for item in analyses:
        mention = item["mention"]
        if not item["subjects"]:
            issues.append(
                {
                    "type": "DEADLINE_NO_SUBJECT",
                    "detail": f"“{mention}”附近未写明申请/主张主体（单位、职工或近亲属）",
                }
            )
        elif len(item["subjects"]) == 1:
            issues.append(
                {
                    "type": "DEADLINE_SUBJECT_INCOMPLETE",
                    "detail": (
                        f"“{mention}”只出现单一主体（{'/'.join(item['subjects'])}）；"
                        "若材料区分单位与职工的期限，应分别说明主体与起算"
                    ),
                }
            )
        if not item["has_trigger"]:
            issues.append(
                {
                    "type": "DEADLINE_NO_TRIGGER",
                    "detail": f"“{mention}”附近未说明从何时起算",
                }
            )
        if not item["has_citation"]:
            issues.append(
                {
                    "type": "DEADLINE_NO_SOURCE",
                    "detail": f"“{mention}”附近未关联材料编号作为依据",
                }
            )

    return {
        "has_deadline": True,
        "mentions": [a["mention"] for a in analyses],
        "details": analyses,
        "issues": issues,
        "needs_review": bool(issues),
        "note": "期限需同时具备主体、起算事实与依据编号；本检查为线索检测，需人工复核",
    }


# ---------------------------------------------------------------- 提示词与输入


# 两套系统提示词：有材料 / 无材料。
# 无材料分支禁止一切编号，且不展示可被照抄的引用示例。

# 两套系统提示词：有材料 / 无材料。
# 无材料分支禁止一切编号，且不展示可被照抄的引用示例。

SYSTEM_PROMPT_WITH_MATERIALS = """你是面向普通人的法律事务辅助助手。本轮**提供了检索材料**，请按以下规则回答：

【引用规则】
1. 只能引用【检索材料】里出现过的编号（M1、M2……），不得引用材料之外的任何编号、条文或案例。
2. 引用原文使用固定格式，编号紧跟在引号之后、不留空格：[M1]
   注意：本提示词不给示例文本，请从材料原文中逐字摘抄，不得改写、拼接或用省略号代替。
3. 只能在“引用”段和“处理方向”段标注编号；不得在“不确定性”段标注编号来“支撑”信息不足的表述。
4. 材料中若标注“仅发布稿摘要”，不得当作判决书全文引用，也不得据此推断裁判文书细节。
5. 材料里如果有与处理路径相关的程序条款（例如申请时限），应在“追问”或“不确定性”中提示核对，
   但只提示适用条件，不要机械抄出全部期限。

【回答范围规则】
1. 围绕用户实际提出的诉求作答。不要在用户没问的情况下，主动扩展到解除、二倍工资、竞业限制等其他争点；
   若认为其他争点可能相关，只能在“追问”里简短询问，不得直接展开分析。
2. **条件不齐时只能做条件性分析**：统一写成“如果……则可能……；若……则……”，
   不得前面先作确定判断、后面又说信息不足。同一回答内不得自相矛盾。
3. 严格区分“必要条件”与“充分条件”。例如“有责任认定书”只是认定工伤的必要条件之一，
   不等于“有责任认定书就符合工伤认定条件”。
4. 不得给出确定金额、胜诉率、确定期限；期限只在确实影响本问题处理路径时，按适用事实附条件提示。
5. 若结论所需事实或依据不足，必须在“不确定性”写明缺什么；不得凭常识补全法律规则。

【输出结构】必须使用以下五个二级标题，顺序固定，每段都要有内容：

## 追问
## 处理方向
## 证据清单
## 引用
## 不确定性

输出语言为中文，不要输出与本结构无关的开场白或免责声明堆砌。"""

SYSTEM_PROMPT_NO_MATERIALS = """你是面向普通人的法律事务辅助助手。本轮**没有提供任何检索材料**，请按以下规则回答：

【无材料规则（最重要）】
1. 本次可引用的编号集合为**空集**。回答中**不得出现任何形如方括号加字母 M 加数字的编号标记**，
   无论是紧贴引号、带空格、还是放在句末，一律不允许。
2. “引用”段只写一句中文说明，例如：本次未提供检索材料，无可引用原文。
3. 不得凭记忆复述法条原文再附编号；可以说明“一般需要核对相关规定”，但不能伪装成已核验引用。
4. 不得使用“示例引用”“此处为示例”“仅作示例”等变通写法来保留编号。
5. 仍然要给出有用的内容：追问关键事实、说明可考虑的处理方向、列出应准备的证据、写明不确定性与缺什么。

【回答范围规则】
1. 围绕用户实际提出的诉求作答。不要在用户没问的情况下，主动扩展到解除、二倍工资、竞业限制等其他争点；
   若认为其他争点可能相关，只能在“追问”里简短询问，不得直接展开分析。
2. **条件不齐时只能做条件性分析**：统一写成“如果……则可能……；若……则……”，
   不得前面先作确定判断、后面又说信息不足。同一回答内不得自相矛盾。
3. 严格区分“必要条件”与“充分条件”。例如“有责任认定书”只是认定工伤的必要条件之一，
   不等于“有责任认定书就符合工伤认定条件”。
4. 不得给出确定金额、胜诉率、确定期限；期限只在确实影响本问题处理路径时，按适用事实附条件提示。
5. 若结论所需事实或依据不足，必须在“不确定性”写明缺什么；不得凭常识补全法律规则。

【输出结构】必须使用以下五个二级标题，顺序固定，每段都要有内容：

## 追问
## 处理方向
## 证据清单
## 引用
## 不确定性

输出语言为中文，不要输出与本结构无关的开场白或免责声明堆砌。"""

# 兼容旧接口
SYSTEM_PROMPT = SYSTEM_PROMPT_WITH_MATERIALS


def render_prompt(task: dict, with_materials: bool) -> str:
    parts = ["【用户问题】", task["question"].strip(), ""]

    parts.append("【已知事实】")
    facts = task.get("facts") or []
    if facts:
        for f in facts:
            note = f"（{f['note']}）" if f.get("note") else ""
            parts.append(f"- {f['name']}：{f['value']}｜来源：{f['source']}{note}")
    else:
        parts.append("- 无（用户仅提供了问题本身）")
    parts.append("")

    parts.append("【检索材料】")
    materials = task.get("materials") or []
    if with_materials and materials:
        for m in materials:
            scope_note = ""
            if m.get("content_scope") == "RELEASE_SUMMARY_ONLY":
                scope_note = "\n注意：本条为官方发布稿摘要，不是判决书全文，不得当作全文引用。"
            parts.append(
                f"[{m['material_id']}] {m['kind']}｜{m['title']}｜{m['locator']}｜版本 {m['version_id']}\n"
                f"来源：{m['source_url']}\n"
                f"核验范围：{m['verified_scope']}{scope_note}\n"
                f"原文：{m['text']}"
            )
            parts.append("")
    else:
        parts.append("（本次无检索材料）")
        parts.append("")

    parts.append("【覆盖缺口】")
    gaps = task.get("gaps") or []
    if with_materials:
        if gaps:
            parts.extend(f"- {g}" for g in gaps)
        else:
            parts.append("- 无已知缺口")
    else:
        parts.append("- 本次未提供任何检索材料")
    parts.append("")

    parts.append("请按系统要求的结构作答，只引用上方材料的编号。")
    return "\n".join(parts)


def build_messages(task: dict, with_materials: bool) -> list[dict]:
    """按是否有材料选择系统提示词分支。"""
    has_materials = bool(with_materials and task.get("materials"))
    system = SYSTEM_PROMPT_WITH_MATERIALS if has_materials else SYSTEM_PROMPT_NO_MATERIALS
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": render_prompt(task, with_materials)},
    ]


# ---------------------------------------------------------------- 模型调用


class MockModel:
    """干跑用：按固定模板生成答案，用于验证评测流程与检查逻辑。"""

    name = "mock"

    def generate(self, messages: list[dict]) -> tuple[str, dict]:
        prompt = messages[-1]["content"]
        has_materials = "（本次无检索材料）" not in prompt
        if not has_materials:
            answer = (
                "## 追问\n"
                "请说明事情发生时间、涉及主体与现有材料。\n\n"
                "## 处理方向\n"
                "本次未提供任何检索材料，无法给出有依据的方向。\n\n"
                "## 证据清单\n"
                "建议先保留合同、工资记录、沟通记录等。\n\n"
                "## 引用\n"
                "无（未提供材料）。\n\n"
                "## 不确定性\n"
                "现有材料不足以支持确定结论。\n"
            )
            return answer, {"prompt_tokens": len(prompt), "completion_tokens": len(answer)}

        ids = re.findall(r"\[(M\d+)\]", prompt)
        quotes = re.findall(r"原文：([^\n]{10,80})", prompt)
        cite = ids[0] if ids else "M1"
        quote = quotes[0] if quotes else ""
        answer = (
            "## 追问\n"
            "请补充具体时间与是否已取得书面材料。\n\n"
            "## 处理方向\n"
            f"根据材料，可先核对事实与条件，再决定路径“{quote}”[{cite}]。\n\n"
            "## 证据清单\n"
            "合同、工资记录、通知文件、沟通记录。\n\n"
            "## 引用\n"
            f"[{cite}]\n\n"
            "## 不确定性\n"
            "期限与适用版本仍需核对；现有材料不足以支持确定结论。\n"
        )
        return answer, {"prompt_tokens": len(prompt), "completion_tokens": len(answer)}


class HFModel:
    """Qwen3-8B 等 HF 模型加载与生成。"""

    def __init__(self, model_name: str, load_4bit: bool = True, max_new_tokens: int = 900):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        kwargs = {"device_map": "auto", "trust_remote_code": True}
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

    def generate(self, messages: list[dict]) -> tuple[str, dict]:
        torch = self.torch
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        started = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        generated = output[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True)
        return answer, {
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": int(generated.shape[0]),
            "latency_s": round(elapsed, 2),
        }


# ---------------------------------------------------------------- 评测主流程


def run(tasks: list[dict], snapshots: dict[str, str], model, *, mode: str) -> list[dict]:
    rows = []
    for task in tasks:
        with_materials = mode == "rag"
        messages = build_messages(task, with_materials)
        started = time.perf_counter()
        answer, meta = model.generate(messages)
        wall = time.perf_counter() - started
        materials = (task.get("materials") or []) if with_materials else []
        check = check_answer(answer, materials, snapshots)
        post = postprocess_answer(answer, materials)
        rows.append(
            {
                "id": task["id"],
                "category": task.get("category"),
                "mode": mode,
                "answer": answer,
                "raw_answer": answer,
                "check": check,
                "postprocess": post,
                # 只有引用检查与后处理都放行，才算“可作为核验通过结果展示”
                "displayable_as_verified": (
                    check["verdict"] == "PASS" and post["displayable_as_verified"]
                ),
                "meta": {**meta, "wall_s": round(wall, 2)},
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    """汇总：编号级状态 + 疑似问题 + 耗时。

    注意：program_appended_cards 是程序附列的原文卡片数，**不计入**模型引用成绩。
    """
    total = len(rows)
    if not total:
        return {"total": 0}

    def cstate(row):
        return row["check"].get("citation_state", "UNKNOWN")

    states: dict[str, int] = {}
    for row in rows:
        states[cstate(row)] = states.get(cstate(row), 0) + 1

    verdicts: dict[str, int] = {}
    for row in rows:
        v = row["check"].get("verdict", "UNKNOWN")
        verdicts[v] = verdicts.get(v, 0) + 1

    blocking = sum(1 for r in rows if r.get("postprocess", {}).get("blocking"))
    suspected = sum(1 for r in rows if r.get("postprocess", {}).get("suspected"))
    displayable = sum(1 for r in rows if r.get("displayable_as_verified"))

    suspected_types: dict[str, int] = {}
    for row in rows:
        for issue in row.get("postprocess", {}).get("suspected_issues", []):
            suspected_types[issue["type"]] = suspected_types.get(issue["type"], 0) + 1

    cards = sum(len(r.get("postprocess", {}).get("citation_cards", [])) for r in rows)
    legacy_quotes = sum(r["check"].get("legacy_verbatim", {}).get("quotes_total", 0) for r in rows)

    latencies = [r["meta"].get("latency_s", r["meta"].get("wall_s", 0)) for r in rows]
    lat_sorted = sorted(latencies)

    return {
        "total": total,
        "citation_state_counts": states,
        "verdict_counts": verdicts,
        "postprocess_blocked": blocking,
        "postprocess_suspected": suspected,
        "postprocess_suspected_types": suspected_types,
        "displayable_as_verified": displayable,
        "program_appended_cards": cards,
        "legacy_self_written_quotes": legacy_quotes,
        "structure_ok_rate": round(sum(1 for r in rows if r["check"]["structure_ok"]) / total, 3),
        "latency_p50_s": lat_sorted[len(lat_sorted) // 2],
        "latency_max_s": lat_sorted[-1],
        "note": (
            "program_appended_cards 为程序附列原文，不计入模型引用成绩；"
            "legacy_self_written_quotes 为模型自行抄写的引文数（仅诊断）"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RAG vs 基座对照评测")
    parser.add_argument("--tasks", required=True, help="eval_tasks.json")
    parser.add_argument("--snapshots", required=True, help="snapshots.json")
    parser.add_argument("--model", default=None, help="HF 模型名，如 Qwen/Qwen3-8B")
    parser.add_argument("--no-4bit", action="store_true", help="不用 4bit 量化")
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--modes", default="baseline,rag")
    parser.add_argument("--mock", action="store_true", help="干跑，不加载模型")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部）")
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args(argv)

    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    tasks = tasks["questions"] if isinstance(tasks, dict) else tasks
    if args.limit:
        tasks = tasks[: args.limit]
    snapshots = json.loads(Path(args.snapshots).read_text(encoding="utf-8"))

    if args.mock or not args.model:
        model = MockModel()
        print("[warn] 使用 MockModel 干跑：结果只能验证流程，不代表模型能力")
    else:
        model = HFModel(args.model, load_4bit=not args.no_4bit, max_new_tokens=args.max_new_tokens)
        print(f"[info] 已加载 {args.model}")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    payload = {"model": getattr(model, "name", "unknown"), "tasks": len(tasks), "results": {}, "summary": {}}
    for mode in modes:
        rows = run(tasks, snapshots, model, mode=mode)
        payload["results"][mode] = rows
        payload["summary"][mode] = summarize(rows)
        print(f"== {mode} ==")
        for key, value in payload["summary"][mode].items():
            print(f"  {key}: {value}")

    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] 结果写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
