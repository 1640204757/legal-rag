"""导入与检索最小原型（首批真实验证用）。

边界：
- 只读取本地已落实的原文，不联网、不采集。
- 不做向量检索、不做重排；只做条款定位、关键词/元数据过滤与引用渲染。
- 所有输出都保留原文定位，供人工复核。

运行：python -m prototype.cli --self-check
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

CN_NUM = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "百": 100,
}


def cn_to_int(text: str) -> int | None:
    """把“二十一”“一百”等中文数字转为整数；失败返回 None。"""
    if not text:
        return None
    total = 0
    if "百" in text:
        left, _, right = text.partition("百")
        total += (CN_NUM.get(left, 0) if left else 1) * 100
        text = right
    if "十" in text:
        left, _, right = text.partition("十")
        total += (CN_NUM.get(left, 1) if left else 1) * 10
        total += CN_NUM.get(right, 0) if right else 0
    elif text:
        total += CN_NUM.get(text, 0)
    return total or None


ARTICLE_RE = re.compile(r"^第([一二三四五六七八九十百]+)条\s*")
CHAPTER_RE = re.compile(r"^第([一二三四五六七八九十百]+)(章|节)\s*")


@dataclass
class Article:
    article_id: str
    article_no: str
    article_num: int | None
    level: str
    parent_id: str | None
    text: str
    locator: str


@dataclass
class LegalDocument:
    doc_id: str
    version_id: str
    title: str
    doc_type: str
    issuing_body: str
    jurisdiction: str
    validity_status: str
    authority_roles: list[str]
    official_url: str
    retrieved_at: str
    articles: list[Article] = field(default_factory=list)
    content_hash: str = ""
    repeal_date: str | None = None
    imported_at: str | None = None
    capture_manifest_ref: str | None = None
    citation_eligibility: dict | None = None
    extra: dict = field(default_factory=dict)

    def to_schema_dict(self) -> dict:
        """把规范对象转换为Schema字段字典；序列化本身不等于通过Schema验证。"""
        return {
            "schema_version": "0.3.0",
            "doc_id": self.doc_id,
            "version_id": self.version_id,
            "title": self.title,
            "doc_type": self.doc_type,
            "issuing_body": self.issuing_body,
            "document_number": self.extra.get("document_number"),
            "field_tags": self.extra.get("field_tags", []),
            "jurisdiction": self.jurisdiction,
            "publish_date": self.extra.get("publish_date"),
            "effective_date": self.extra.get("effective_date"),
            "revision_date": self.extra.get("revision_date"),
            "repeal_date": self.repeal_date,
            "validity_status": self.validity_status,
            "validity_basis": self.extra.get("validity_basis"),
            "authority_level": self.extra.get("authority_level", "UNKNOWN"),
            "authority_roles": self.authority_roles,
            "citation_eligibility": self.citation_eligibility,
            "legal_basis": self.extra.get("legal_basis"),
            "transition_rules": self.extra.get("transition_rules"),
            "version_relations": self.extra.get("version_relations", []),
            "official_url": self.official_url,
            "retrieved_at": self.retrieved_at,
            "imported_at": self.imported_at,
            "capture_manifest_ref": self.capture_manifest_ref,
            "verified_at": self.extra.get("verified_at"),
            "content_hash": self.content_hash,
            "reuse_basis": self.extra.get("reuse_basis"),
            "articles": [
                {
                    "article_id": a.article_id,
                    "version_id": self.version_id,
                    "parent_id": a.parent_id,
                    "level": a.level,
                    "article_no": a.article_no,
                    "text": a.text,
                    "locator": a.locator,
                    "cross_refs": [],
                }
                for a in self.articles
            ],
        }


def normalize_line(line: str) -> str:
    """只规范行内空白，便于按结构识别条款；原始快照另外保留。"""
    line = line.replace("\u3000", " ").replace("\xa0", " ")
    return re.sub(r"[ \t]+", " ", line).strip()


def slice_between(lines: list[str], start_pat: str, end_pat: str | None) -> list[tuple[int, str]]:
    """返回 [start, end) 区间内的原始行号与内容。"""
    start = None
    for i, line in enumerate(lines):
        if re.search(start_pat, line):
            start = i
            break
    if start is None:
        return []
    out = []
    for i in range(start, len(lines)):
        line = lines[i]
        if end_pat and i > start and re.search(end_pat, line):
            break
        out.append((i, line))
    return out


def parse_articles(raw_lines: Iterable[str], *, id_prefix: str, start: int = 1) -> list[Article]:
    """把原文行解析为条款列表。

    只做结构性切分：连续行归属当前条款；无法识别的行为归入当前条款正文或 UNCLASSIFIED。
    不做任何内容改写、补全或摘要。
    """
    articles: list[Article] = []
    current: Article | None = None
    current_parent: str | None = None
    seen: dict[str, int] = {}

    for idx, raw in enumerate(raw_lines, start=start):
        line = normalize_line(raw)
        if not line:
            continue
        chapter = CHAPTER_RE.match(line)
        if chapter:
            num = cn_to_int(chapter.group(1))
            current_parent = f"{id_prefix}-CH-{num}"
            current = None
            continue
        match = ARTICLE_RE.match(line)
        if match:
            num = cn_to_int(match.group(1))
            key = f"{num}"
            seen[key] = seen.get(key, 0) + 1
            article_id = f"{id_prefix}-ART-{num:03d}" if seen[key] == 1 else f"{id_prefix}-ART-{num:03d}-{seen[key]}"
            current = Article(
                article_id=article_id,
                article_no=f"第{match.group(1)}条",
                article_num=num,
                level="ARTICLE",
                parent_id=current_parent,
                text=line,
                locator=f"L{idx}",
            )
            articles.append(current)
            continue
        if current is not None:
            # 同一条文内的续行（款、项），保留原文并按行号记录
            current.text = current.text + "\n" + line
            current.locator = f"{current.locator}-L{idx}"
    return articles


def load_capture_manifest() -> dict[str, dict]:
    """读取真实采集记录；缺失则报错，不允许用导入时间顶替采集时间。"""
    path = RAW_DIR / "capture_manifest.json"
    if not path.exists():
        raise RuntimeError("缺少采集记录 data/raw/capture_manifest.json：不得用导入时间冒充采集时间")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {c["doc_id"]: c for c in payload["captures"]}


def capture_for(doc_id: str, raw_file: str) -> dict:
    """核对指定资料的采集记录、原文件名与SHA256，缺失或不一致时直接拒绝。"""
    manifest = load_capture_manifest()
    if doc_id not in manifest:
        raise RuntimeError(f"采集记录中没有 {doc_id}")
    capture = manifest[doc_id]
    if capture["raw_file"] != raw_file:
        raise RuntimeError(f"采集记录文件名不匹配：{capture['raw_file']} != {raw_file}")
    actual = hashlib.sha256((RAW_DIR / raw_file).read_bytes()).hexdigest()
    if actual.lower() != capture["content_hash"].lower():
        raise RuntimeError(
            f"原文快照与采集记录哈希不一致：{doc_id} "
            f"(actual={actual[:16]} expected={capture['content_hash'][:16]})"
        )
    return capture


# 正文块的起止边界：导入与逐字核验必须共用，避免核验区间包含页脚
BODY_BOUNDARIES: dict[str, tuple[str, str | None]] = {
    "LAW-LABOR-INTERP-2": (r"^法释〔2025〕12号$", r"^相关链接"),
    "LAW-WORK-INJURY-REGULATION": (r"^第一章\s*总则", r"^返回$"),
}


def extract_body_text(doc_id: str, raw_text: str) -> str:
    """按统一边界截取正文，供解析与核验共用。"""
    if doc_id not in BODY_BOUNDARIES:
        raise RuntimeError(f"未定义正文边界：{doc_id}")
    start_pat, end_pat = BODY_BOUNDARIES[doc_id]
    lines = raw_text.splitlines()
    block = slice_between(lines, start_pat, end_pat)
    if not block:
        raise RuntimeError(f"未能按边界截取正文：{doc_id}")
    return "\n".join(line for _, line in block)


def parse_labor_interpretation_2() -> LegalDocument:
    """解析既有劳动争议解释（二）快照并附来源/版本字段；不自动采集其他法律。"""
    path = RAW_DIR / "law-labor-interp-2.txt"
    capture = capture_for("LAW-LABOR-INTERP-2", "law-labor-interp-2.txt")
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    start_pat, end_pat = BODY_BOUNDARIES["LAW-LABOR-INTERP-2"]
    block = slice_between(lines, start_pat, end_pat)
    if not block:
        raise RuntimeError("未在原文中定位到法释〔2025〕12号正文块")
    articles = parse_articles([line for _, line in block], id_prefix="LABOR-INTERP-2", start=block[0][0] + 1)
    body = "\n".join(line for _, line in block)
    return LegalDocument(
        doc_id="LAW-LABOR-INTERP-2",
        version_id="2025-09-01",
        title="最高人民法院关于审理劳动争议案件适用法律问题的解释（二）",
        doc_type="JUDICIAL_INTERPRETATION",
        issuing_body="最高人民法院",
        jurisdiction="CN",
        validity_status="IN_FORCE",
        authority_roles=["ADJUDICATIVE_BASIS"],
        official_url="https://www.court.gov.cn/fabu/xiangqing/472691.html",
        retrieved_at=capture["retrieved_at"],
        imported_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        capture_manifest_ref="data/raw/capture_manifest.json",
        articles=articles,
        content_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        extra={
            "document_number": "法释〔2025〕12号",
            "field_tags": ["LABOR"],
            "publish_date": "2025-07-31",
            "effective_date": "2025-09-01",
            "authority_level": "JUDICIAL_INTERPRETATION",
            "validity_basis": "新闻发布稿载明“自2025年9月1日起施行”，正文第二十一条同",
            "transition_rules": "第二十一条：解释（一）第三十二条第一款同时废止；此前司法解释与本解释不一致的，以本解释为准",
            "version_relations": [{"relation": "SUPERSEDES", "version_id": "2021-01-01", "note": "整合/替代此前劳动争议司法解释中不一致内容（需逐条核对）"}],
            "verified_at": None,
            "reuse_basis": None,
        },
    )


def parse_work_injury_regulation() -> LegalDocument:
    """解析既有工伤保险条例快照并附来源/版本字段；实际版本适用仍需另行核对。"""
    path = RAW_DIR / "law-work-injury-regulation.txt"
    capture = capture_for("LAW-WORK-INJURY-REGULATION", "law-work-injury-regulation.txt")
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    start_pat, end_pat = BODY_BOUNDARIES["LAW-WORK-INJURY-REGULATION"]
    block = slice_between(lines, start_pat, end_pat)
    if not block:
        raise RuntimeError("未在原文中定位到《工伤保险条例》正文块")
    articles = parse_articles([line for _, line in block], id_prefix="WORK-INJURY", start=block[0][0] + 1)
    body = "\n".join(line for _, line in block)
    return LegalDocument(
        doc_id="LAW-WORK-INJURY-REGULATION",
        version_id="2010-12-20",
        title="工伤保险条例",
        doc_type="ADMINISTRATIVE_REGULATION",
        issuing_body="国务院",
        jurisdiction="CN",
        validity_status="IN_FORCE",
        authority_roles=["ADJUDICATIVE_BASIS"],
        official_url="https://xzfg.moj.gov.cn/front/law/detail?LawID=610",
        retrieved_at=capture["retrieved_at"],
        imported_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        capture_manifest_ref="data/raw/capture_manifest.json",
        articles=articles,
        content_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        extra={
            "document_number": "国务院令第375号（2010年修订）",
            "field_tags": ["LABOR"],
            "publish_date": "2010-12-20",
            "effective_date": "2011-01-01",
            "authority_level": "ADMINISTRATIVE_REGULATION",
            "validity_basis": "页面标注 2003-04-27 公布、2010-12-20 修订；施行日期见第六十七条（需核验是否为修订后文本）",
            "verified_at": None,
            "reuse_basis": None,
        },
    )


# ---------------------------------------------------------------- 导入


def load_documents() -> list[LegalDocument]:
    """加载早期原型中固定的两部规范；新增通用资料由local_rag.ingest链路处理。"""
    return [parse_labor_interpretation_2(), parse_work_injury_regulation()]


def build_index(docs: list[LegalDocument]) -> list[dict]:
    """把规范中的条款展开为词面检索记录；该原型函数本身不生成向量。"""
    index = []
    for doc in docs:
        for art in doc.articles:
            index.append(
                {
                    "doc_id": doc.doc_id,
                    "version_id": doc.version_id,
                    "title": doc.title,
                    "article_id": art.article_id,
                    "article_no": art.article_no,
                    "text": art.text,
                    "locator": art.locator,
                    "authority_roles": doc.authority_roles,
                    "validity_status": doc.validity_status,
                    "official_url": doc.official_url,
                }
            )
    return index


# ---------------------------------------------------------------- 检索与引用


def _bigrams(text: str) -> set[str]:
    """字符二元组集合，用于词面重叠打分（不做语义、不做同义词）。"""
    cleaned = re.sub(r"[\s，。；、,.;？！?！（）()「」“”]+", "", text)
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


class LexicalIndex:
    """带 IDF 权重的词面索引。

    IDF 只统计“某个二元组出现在多少个条款里”，纯词频统计：
    像“工伤”“责任”这类到处出现的组合权重低，
    “转包”“招用”这类少见的组合权重高。不涉及语义理解。
    """

    def __init__(self, rows: list[dict]):
        """初始化对象依赖与内部状态；后续读写通过本类的方法完成。"""
        self.rows = rows
        self.grams: list[set[str]] = []
        df: dict[str, int] = {}
        for row in rows:
            grams = _bigrams(row["text"])
            self.grams.append(grams)
            for g in grams:
                df[g] = df.get(g, 0) + 1
        n = max(1, len(rows))
        # 平滑 IDF：越稀有权重越高
        self.idf = {g: (n / c) for g, c in df.items()}
        self._default_idf = n

    def score(self, query: str) -> list[tuple[float, dict, str]]:
        """计算原型词面候选得分并排序；只反映词语相关，不核验结论支持性。"""
        terms = [t for t in re.split(r"[\s，。；、,.;？！?！]+", query) if len(t) >= 2]
        qgrams = _bigrams(query)
        n = max(1, len(self.rows))

        # 整词也要按稀有度加权：像“用人单位”“劳动者”这类处处出现的词权重应低，
        # 否则会压过真正有区分度的“未参加”“转包”等表述。
        term_df = {}
        for t in terms:
            term_df[t] = max(1, sum(1 for row in self.rows if t in row["text"]))
        term_idf = {t: (n / c) for t, c in term_df.items()}

        results = []
        for row, grams in zip(self.rows, self.grams):
            exact_score = 0.0
            exact_terms = 0
            for t in terms:
                if t in row["text"]:
                    exact_terms += 1
                    exact_score += term_idf[t]
            overlap = qgrams & grams
            weighted = sum(self.idf.get(g, self._default_idf) for g in overlap)
            if exact_terms > 0:
                # 覆盖率优先（命中几个词），其次看稀有度
                coverage = exact_terms / max(1, len(terms))
                results.append((coverage * 10_000 + exact_score * 10 + weighted, row, "exact"))
            elif overlap:
                coverage = len(overlap) / max(1, len(qgrams))
                if coverage >= 0.25:
                    results.append((weighted, row, "bigram"))
        results.sort(key=lambda x: (-x[0], x[1].get("doc_id", ""), x[1].get("article_id", x[1].get("case_id", ""))))
        return results


def search(index: list[dict], query: str, *, top_k: int = 5, lexical: "LexicalIndex | None" = None) -> list[dict]:
    """词面检索：整词命中优先，IDF 加权二元组兜底。不做语义推断。"""
    scorer = lexical or LexicalIndex(index)
    results = scorer.score(query)
    return [{"hits": round(s, 2), "match_mode": mode, **row} for s, row, mode in results[:top_k]]


# ---------------------------------------------------------------- 引用标签与核验

LABEL_STATUTORY = "本案处置法定依据"
LABEL_ANALOGOUS = "类案参考"
LABEL_BACKGROUND = "背景资料"

# 本原型尚未实现逐字比对、版本适用与事实要件匹配，因此全局关闭“法定依据”标签。
STATUTORY_LABEL_ENABLED = False


class StatutoryLabelBlocked(RuntimeError):
    """未完成核验就请求“法定依据”标签时抛出。"""


def build_label_set() -> set[str]:
    """返回当前允许展示的引用标签；法定依据总开关关闭时不能出现该标签。"""
    labels = {LABEL_ANALOGOUS, LABEL_BACKGROUND}
    if STATUTORY_LABEL_ENABLED:
        labels.add(LABEL_STATUTORY)
    return labels


def render_citation(row: dict, *, eligibility: dict | None = None, raw_text: str | None = None) -> dict:
    """渲染引用卡片。

    标签由核验结果决定，不接受调用方直接指定：
    - 未提供 eligibility 或未全部满足 → 背景资料/待核验；
    - 只有 eligibility.verdict == ELIGIBLE 且四项均为 true 才允许“法定依据”；
    - 本原型 STATUTORY_LABEL_ENABLED=False，任何情况下都不会输出该标签。

    若提供 raw_text，会对卡片展示的引用片段做逐字核验，结果写入 quote_verification；
    核验失败时不隐藏、不改写文本，而是明确标出失败。
    """
    label = LABEL_BACKGROUND
    reasons: list[str] = []

    if not STATUTORY_LABEL_ENABLED:
        reasons.append("本原型未启用“法定依据”标签：版本适用与事实匹配未实现")
    elif eligibility and eligibility.get("verdict") == "ELIGIBLE":
        checks = ("verified_verbatim", "version_confirmed", "authority_eligible", "facts_matched")
        if all(eligibility.get(k) is True for k in checks):
            label = LABEL_STATUTORY
        else:
            reasons.append("引用资格核验未全部通过")
    else:
        reasons.append("未提供引用资格核验结果")

    quote = row["text"][:160]
    quote_verification: dict = {"checked": False}
    if raw_text is not None:
        from prototype.verifier import verify_quote

        ok, why = verify_quote(raw_text, quote)
        quote_verification = {"checked": True, "verified": ok, "reason": why}
        if not ok:
            reasons.append(f"引用片段逐字核验失败：{why}")

    return {
        "label": label,
        "label_reasons": reasons,
        "title": row["title"],
        "article_no": row["article_no"],
        "version_id": row["version_id"],
        "locator": row["locator"],
        "source_url": row["official_url"],
        "text": quote,
        "quote_verification": quote_verification,
    }


def coverage_statement(has_case_docs: bool) -> str:
    """按是否有案例资料生成覆盖声明，区别案例摘要与文书全文。"""
    if has_case_docs:
        return "本次检索覆盖：法规原文（已核验可取得正文）；案例仅覆盖已发布的官方要旨，未覆盖案例全文。"
    return "本次检索覆盖：法规原文；案例库检索需登录，本次未覆盖任何案例全文或类案。"
