"""引用检查：编号级校验（主）+ 逐字引文核验（仅用于旧答案诊断）。

主阅读入口为 check_answer()；parse_used_ids()/find_citations() 共用声明边界，兼容句尾连写并保留标点。

## 为什么改

2026-10-19 第三轮实测确认：让模型在自由文本里同时完成「分析 + 抄条文 + 绑编号」不可靠——
长引文容易漏绑编号，“引用”段退化为概述。因此改为：

- **模型只声明用到的编号**（`USED_IDS: M1, M2`）；
- **原文由程序从材料生成引用卡片**，标记 `系统附列原文`；
- 程序附列的原文**不算模型逐字引用成功**。

## 编号级状态（主判定）

| 状态 | 含义 | verdict |
| --- | --- | --- |
| `NO_MATERIAL_CLEAN` | 无材料且无编号 | PASS |
| `NO_MATERIAL_HAS_CITATION` | 无材料却出现编号 | FAIL |
| `IDS_OK` | 有材料，声明编号均有效 | PASS |
| `IDS_NONE_USED` | 有材料，声明未使用任何材料 | PASS_WITH_NOTE |
| `IDS_MISSING` | 有材料，但缺少 USED_IDS 声明 | FAIL |
| `IDS_UNKNOWN` | 声明或正文出现未提供编号 | FAIL |
| `FAKE_EXAMPLE` | 用“示例引用”等方式保留编号 | FAIL |

## 逐字核验（legacy_verbatim）

仅当答案里**确有引号引文**时才运行，用于诊断旧版本答案或发现模型仍在自行抄写。
结果单独放在 `legacy_verbatim`，**不参与编号级 verdict**，也不计入“模型引用通过率”。

边界：通过检查 ≠ 法律适用正确。
"""

from __future__ import annotations

import re

# 编号形式：
#   [M1] / [ M1 ] / [M 1]   —— 显式方括号
#   M1 / M 1                —— 正文简写（如“依据 M1 分析”），不带方括号
BRACKET_CITATION_RE = re.compile(r"\[\s*M\s*(\d+)\s*\]")
BARE_CITATION_RE = re.compile(r"(?<![A-Za-z0-9])M\s*(\d+)(?![0-9])")
CITATION_RE = BRACKET_CITATION_RE  # 兼容旧引用点
QUOTE_PATTERNS = (
    re.compile(r"“([^”]{2,400})”"),
    re.compile(r"「([^」]{2,400})」"),
)
EXPLICIT_QUOTE_RE = re.compile(r"原文\s*[:：]\s*$")
SECTION_RE = re.compile(r"^#{1,4}\s*(.+?)\s*$")
# 声明可独立或连在句尾，大小写/冒号/水平空格容错；左右标识符边界不能误吞普通变量。
# 第一个捕获组仍是编号列表，供解析、正文排除与服务展示清理共用。
# 声明及其值限于同行；空声明后不能跨行把正文当编号列表解析或从展示中删掉。
USED_IDS_RE = re.compile(r"(?<![A-Za-z0-9_])USED[ _]*IDS(?=[ \t:：\d]|M[ \t]*\d|无|none\b|$)"
                         r"[ \t]*[:：]?[ \t]*(.*)$", re.MULTILINE | re.IGNORECASE)

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
    """去除空白以便比对文本；不用于证明改写后的引文准确。"""
    return re.sub(r"\s+", "", text or "")


def extract_citations(answer: str) -> list[int]:
    """提取方括号材料编号中的数字，去重并排序；正文简写由find_citations处理。"""
    nums = {int(m.group(1)) for m in BRACKET_CITATION_RE.finditer(answer or "")}
    return sorted(nums)


def find_citations(answer: str) -> list[str]:
    """输入原始回答，返回正文中带或不带方括号的去重编号；无外部副作用。

    独立行或句尾 `USED_IDS:` 声明不算正文引用；去声明时保留前面的正文与句末标点。
    """
    text = answer or ""
    body = USED_IDS_RE.sub("", text)
    seen: set[str] = set()
    out: list[str] = []
    for m in BRACKET_CITATION_RE.finditer(body):
        key = f"M{m.group(1)}"
        if key not in seen:
            seen.add(key)
            out.append(key)
    for m in BARE_CITATION_RE.finditer(body):
        key = f"M{m.group(1)}"
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def parse_used_ids(answer: str):
    """解析原回答中独立行或句尾声明并返回去重编号；未出现返回 None，空/无返回 []。

    仅识别声明及编号，不判断依据的法律适用性；无文件、网络或会话副作用。
    """
    matches = list(USED_IDS_RE.finditer(answer or ""))
    if not matches:
        return None
    seen, out = set(), []
    for match in matches:
        raw = match.group(1).strip()
        # 仅声明区域允许省略M；正文普通数字仍绝不解释为材料编号。
        for match_id in re.finditer(r"M[ \t]*(\d+)|(?<![A-Za-z0-9])(\d+)(?![A-Za-z0-9])", raw, re.IGNORECASE):
            n = match_id[1] or match_id[2]
            key = f"M{n}"
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def body_material_ids(answer, materials):
    """返回正文明确且唯一绑定的合法编号，供声明补齐；不改原文、不推断法律适用。

    显式编号只取本轮存在者；法名/简称+条号须唯一到一张卡，裸条号或同法多条不猜编号。
    """
    from prototype.rag.grounding import ARTICLE, numeric_value, resolve_law_names
    provided = {m['material_id'] for m in materials}
    output = list(dict.fromkeys(mid for mid in find_citations(answer) if mid in provided))
    body = USED_IDS_RE.sub('', answer or '')
    def key(text):
        match = re.search(ARTICLE, text)
        if not match:
            return None
        number, _, suffix = match[0][1:].partition('条')
        return numeric_value(number), suffix
    for clause in re.split(r'[。；;\n]', body):
        for match in re.finditer(ARTICLE, clause):
            names = resolve_law_names(clause[:match.start()], materials)
            candidates = [m for m in materials if names and m.get('title') in names
                          and key(m.get('locator', '')) == key(match[0])]
            if len(candidates) == 1 and candidates[0]['material_id'] not in output:
                output.append(candidates[0]['material_id'])
        if not re.search(ARTICLE, clause):
            for match in re.finditer(r'《[^《》]+》', clause):
                names = resolve_law_names(match[0], materials)
                candidates = [m for m in materials if names and m.get('title') in names]
                if len(candidates) == 1 and candidates[0]['material_id'] not in output:
                    output.append(candidates[0]['material_id'])
    return sorted(output, key=lambda mid: int(mid[1:]))


def detect_fake_example(answer: str) -> bool:
    """识别用示例引用措辞保留编号的情况，供引用检查使用。"""
    text = answer or ""
    return any(m in text for m in FAKE_MARKERS) and bool(CITATION_RE.search(text))


def extract_quotes(answer: str) -> list[str]:
    """兼容接口：返回答案中的全部非强调引文。"""
    return [b["quote"] for b in extract_quote_bindings(answer) if b.get("kind") != "emphasis"]


def parse_sections(answer: str) -> dict[str, str]:
    """按Markdown标题拆分回答正文，供结构与分区检查使用。"""
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
    """核对约定段落标题是否齐全；结构齐全不能证明内容正确。"""
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
        if cited or unknown_declared:
            state = "NO_MATERIAL_HAS_CITATION"
            reasons.append(f"本轮无材料，但回答出现或声明编号：{', '.join(unknown)}")
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
    """寻找当前段落结束位置，避免将后文其他段落的编号绑定到本句。"""
    rest = text[start:]
    blank = re.search(r"\n\s*\n", rest)
    heading = re.search(r"\n#{1,4}\s", rest)
    ends = [m.start() for m in (blank, heading) if m]
    return start + (min(ends) if ends else len(rest))


def _quote_spans(answer: str):
    """定位引号中的文本及其位置，标记是否带显式原文前缀。"""
    text = answer or ""
    spans = []
    for pattern in QUOTE_PATTERNS:
        for m in pattern.finditer(text):
            prefix = text[max(0, m.start() - 12) : m.start()]
            spans.append((m.start(), m.end(), m.group(1), bool(EXPLICIT_QUOTE_RE.search(prefix))))
    spans.sort(key=lambda x: x[0])
    return spans


def _unique(items):
    """按首次出现顺序去重，避免重复编号改变绑定判定。"""
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_quote_bindings(answer: str) -> list[dict]:
    """提取旧式引文与候选编号的绑定关系，无法唯一绑定时保留歧义状态。"""
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
    """按所标材料及快照判断旧式引文的归属，不用其他文档替错误编号兜底。"""
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
