"""期限提示检查：主体 + 起算 + 依据编号（围绕期限句本身判断）。

修正 2026-10-19：此前在全文里找关键词，只要全文任意处出现“社会保险行政部门”“之日起”
就算通过，导致“事故发生之日起1年内提出”这种缺少申请主体的表述被放过。
现在改为在**期限出现位置的上下文窗口**内判断，并要求主体区分（单位/职工）与起算事实同时可读。
"""

from __future__ import annotations

import re

DEADLINE_RE = re.compile(r"(?:\d+\s*(?:日|天|个月|年)内|自[^，。；]{0,24}之日起\s*\d+\s*(?:日|天|个月|年))")

# 编号识别复用 citation_check（带不带方括号都算），避免两套正则漂移
from prototype.rag.citation_check import find_citations as _find_citations  # noqa: E402

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
    """截取期限前后的有限上下文，避免全文任意主体被当作该期限的主体。"""
    return text[max(0, start - CONTEXT_WINDOW) : min(len(text), end + CONTEXT_WINDOW)]


def extract_deadline_mentions(answer: str) -> list[str]:
    """提取规则能识别的期限表达；没有匹配不代表文本没有期限问题。"""
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
        has_citation = bool(_find_citations(ctx))
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
