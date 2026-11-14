"""本地问答流程：本地检索为主，缺失专项可受控定位官方目录，核对输出并整理展示。
重点阅读ChatService._run()；页面展示的前8条与"更多候选"由_display_candidates()另取，不送模型。
内存会话之外，每轮另写本地 JSONL 冷归档（local_rag/session_memory.py）；材料相关不代表法律适用正确。

阅读指南：docs/guides/PYCHARM_PROJECT_READING.md（相对项目根目录）。
"""
import asyncio
from dataclasses import dataclass, field
from datetime import date
import math
import re
import secrets
import time

from prototype.rag.input_builder import Material, Fact, RagInput, build_messages
from prototype.rag.postprocess import postprocess
from prototype.rag.citation_check import USED_IDS_RE, find_citations, parse_used_ids
from .evaluation import judge_degrade
from .answer_permission import MATERIAL_READING_SYSTEM, material_reading_issues
from .answer_assembly import MATERIAL_SELECTION_SYSTEM, build_final_answer, split_source_excerpts
from .official_capture import public_url, normalize_catalog_title
from .retrieval import load_official_whitelist_catalog, match_official_whitelist_catalog
from . import session_memory


OFFICIAL_SUPPLEMENT_SCOPE = "官方白名单实时检索补充；已核验官方来源与条号，未独立复核个案适用"
_ARTICLE_NUMBER = re.compile(r"第[零〇一二三四五六七八九十百千万两\d]+条(?:之[零〇一二三四五六七八九十百千万两\d]+)?")
# 页面"检索候选"展示前8条；送给模型的仍是默认检索的前4条及原预算筛选结果。
DISPLAY_TOP_K = 8
DISPLAY_FIELDS = ("id", "kind", "title", "locator", "version_id", "text", "source_url", "verified_scope", "content_scope")

# ---------------------------------------------------------------- 提示词预算
# 装配目标 6800 字节，最终消息硬顶 7000 字节；单条消息按 UTF-8 字节 + 32 字节记账。
# 顺序是“材料第一顺位”：先按原规则整条装材料，再用剩余空间倒序装历史；
# 历史为空时这条路径与改造前逐字节一致（见 test_single_turn_query_and_messages_are_byte_identical）。
PROMPT_BUDGET = 6800
MESSAGE_LIMIT = 7000
# 有历史时先为最近一轮留出的字节数，防止材料把历史挤成 0 条（反向饿死）。
HISTORY_RESERVE = 700
# 送模问题副本的字节上限；检索问句与金额/期限豁免仍用完整问题，不受此截断影响。
QUESTION_PROMPT_LIMIT = 600
# 历史视图压缩：用户发言保留开头（案情通常在首次陈述里），助手旧回答只留一行正文。
HISTORY_USER_LIMIT = 400
HISTORY_ANSWER_LIMIT = 120
BUDGET_GAP_TEXT = "部分候选材料因上下文预算未送入模型；右侧仅列出实际提供的材料。"
LONG_QUESTION_GAP_TEXT = "本轮问题过长，只把前面一段送入模型；完整内容已保留，可分次追问。"


def _truncate_bytes(text, limit):
    """按 UTF-8 字节截断文本，且不会把汉字切半；无副作用。"""
    raw = (text or "").encode("utf-8")
    if len(raw) <= limit:
        return text or ""
    return raw[:limit].decode("utf-8", "ignore")


def _first_body_line(text, limit):
    """取回答中第一条非标题正文并截断，作为一行可读的阶段结论。"""
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return _truncate_bytes(stripped, limit)
    return ""


def _compact_history(history):
    """把会话历史压成装得进预算的只读视图；不修改原始历史，也不改写用户陈述。

    用户发言保留开头，助手旧回答只留一行正文，避免旧长篇占满预算。
    仅影响本轮送模内容：完整历史仍留在 Session.history，供归档与回查使用。
    """
    view = []
    for message in history or []:
        content = message.get("content") or ""
        if message.get("role") == "user":
            compact = _truncate_bytes(" ".join(content.split()), HISTORY_USER_LIMIT)
        else:
            compact = _first_body_line(content, HISTORY_ANSWER_LIMIT)
        if compact:
            view.append({"role": message.get("role"), "content": compact})
    return view


def _official_slice(entry, row):
    """核对注入抓取器切片的官方URL、法名与正文起始条号，返回可送模副本或None；无外部副作用。

    抓取器是受信任适配器，须先核验真实响应及每次重定向；本检查不下载正文、不证明个案适用。
    来源地址本身不能证明引文真实；目录元数据及不完整/错位返回不能伪装成材料。
    """
    if not isinstance(row, dict) or normalize_catalog_title(row.get('title')) != normalize_catalog_title(entry['title']):
        return None
    text, locator = row.get('text'), row.get('locator')
    if not isinstance(text, str) or not isinstance(locator, str):
        return None
    article, body_article = _ARTICLE_NUMBER.match(locator.strip()), _ARTICLE_NUMBER.match(text.strip())
    if not article or not body_article or article.group() != body_article.group():
        return None
    # 标题/条号本身不是条文切片，空白或只有标点的载荷不得套上已核验标签。
    if not re.search(r'[\u4e00-\u9fffA-Za-z0-9]', text.strip()[body_article.end():]):
        return None
    source = row.get('source_url') or row.get('url')
    try:
        public_url(source, allow_government=True)
        for key in ('url', 'official_url', 'final_url'):
            if row.get(key):
                public_url(row[key], allow_government=True)
        for hop in row.get('redirect_chain') or []:
            public_url(hop, allow_government=True)
    except (ValueError, TypeError):
        return None
    return dict(kind='LAW_ARTICLE', title=entry['title'], locator=article.group(), text=text.strip(),
                version_id=str(row.get('version_id') or entry['publication_date'] or '官方实时补充（版本待核）'),
                source_url=source, verified_scope=OFFICIAL_SUPPLEMENT_SCOPE, content_scope='ARTICLES_ONLY')


def _official_trigger(hits, mode):
    """为已匹配的缺失专项记录触发原因；仅使用有限检索分数，缺分数不臆测低置信度。"""
    if not hits:
        return 'LOCAL_EMPTY'
    scores = [row.get('retrieval_score') for row in hits]
    scores = [value for value in scores if isinstance(value, (int, float)) and math.isfinite(value)]
    if scores and ((mode.startswith('lexical') and max(scores) < 35.0)
                   or (mode.startswith('hybrid') and max(scores) < 0.045)):
        return 'LOW_LOCAL_CONFIDENCE'
    return 'MISSING_SPECIAL_TOPIC'


def _display_row(row, sent, **extra):
    """整理一条展示候选：只留页面要用的字段，标明是否送入模型及送入时的编号；无外部副作用。"""
    key = (row.get("title"), row.get("locator"), row.get("version_id"))
    return dict({k: row.get(k) for k in DISPLAY_FIELDS}, sent_to_model=key in sent, material_id=sent.get(key), **extra)


def _material_rows(materials):
    """转为校验/审计用的普通字典；剥离仅供模型选号的 excerpts，保留其余原始字段。"""
    return [{key: value for key, value in material.__dict__.items() if key != "excerpts"}
            for material in materials]


def clean_display_answer(raw, materials):
    """用本轮材料法名补全条号并隐藏内部编号，返回展示文本；无外部副作用。

    输入原始回答及 Material 对象；仅用于原始引用和期限校验通过后的展示阶段。
    未知编号不补造法名，不修复或替代校验，原始回答及引用卡片由调用方独立保留。
    """
    # 与原始编号校验共用声明边界，兼容句尾连写且不移除声明前的句末标点。
    text = USED_IDS_RE.sub("", raw or "").strip()
    titles = {m.material_id: m.title for m in materials
              if getattr(m, "material_id", "") and getattr(m, "title", "")}

    # 阅读格式先补来源法名，再隐藏内部编号；仅在原始校验全部通过后用于展示。
    text = re.sub(r"材料\[(M[1-9]\d*)\](?=原文|中的)",
                  lambda m: f"《{titles[m[1]]}》" if m[1] in titles else m[0], text)

    def expand_article(match):
        """只展开已知材料的条号引用；已有紧邻法名时避免重复插入书名号。"""
        mid = "M" + (match.group(1) or match.group(2))
        article = match.group(3)
        if mid not in titles:
            return match.group()
        if text[:match.start()].rstrip(" \t").endswith("》"):
            return article
        return f"《{titles[mid]}》{article}"

    text = re.sub(
        r"(?:\[\s*M\s*(\d+)\s*\]|(?<![A-Za-z0-9])M\s*(\d+))\s*(第[零〇一二三四五六七八九十百千万两\d]+条)",
        expand_article, text,
    )
    # 编号前只吃水平空白，不能吞掉标题后换行，把下一行正文并入标题。
    text = re.sub(r"[ \t]*(?:\[\s*M\s*\d+\s*\]|[（(]\s*M\s*\d+\s*[)）])+", "", text)
    # 裸编号同样是内部引用；合法者补出法名，未知者原样保留供上游硬校验，不掩盖错误。
    text = re.sub(r'(?<![A-Za-z0-9])M\s*(\d+)(?!\d)',
                  lambda match: f"《{titles['M'+match[1]]}》" if 'M'+match[1] in titles else match[0], text)
    # 只整理标点前的水平空白，保留标题和实操步骤之间的换行。
    return re.sub(r"[ \t]+([，。；：！？、,.!?;:])", r"\1", text).strip()


def answer_scenario_issues(question, answer):
    """按原问句核对明确错误的刑民路径和无关主体模板，返回拦截原因；无外部副作用。

    只检测列出的显式误用，不认证完整管辖或法律推理；否定民事反诉、另行民事赔偿不误拦。
    """
    criminal = bool(re.search(r'强奸|猥亵|性侵|强暴|诬告|陷害|反咬|刑事|报案|报警|诈骗|盗窃|抢劫', question))
    contextual = bool(re.search(r'劳动|工资|欠薪|工伤|辞退|学校|校园|老师|学生|在单位|在公司|办公室|工作场所', question))
    issues = []
    body = USED_IDS_RE.sub('', answer or '')
    if criminal:
        civil_case = bool(re.search(r'民事(?:诉讼|起诉|案件)|(?:借款|欠款|货款|合同).{0,12}起诉', question))
        for clause in re.split(r'[。！？；;\n，,]', body):
            for match in re.finditer(r'民事反诉|反诉|民事(?:诉讼|起诉).{0,12}(?:刑事责任|强奸罪|猥亵罪|诬告陷害罪)', clause):
                prefix = re.split(r'但是|但|而是|不过', clause[:match.start()])[-1]
                # 明示并存的独立民事案件可讨论债权反诉，不能把该例外用于处理犯罪指控本身。
                if (civil_case and re.search(r'借款|欠款|货款|合同|民事案件', clause)
                        and not re.search(r'强奸|猥亵|诬告|诈骗|刑事|犯罪', clause)):
                    continue
                # 否定词必须直接约束反诉路径；“不要犹豫”“不能只报警还应反诉”仍是正向错误建议。
                if not re.search(r'(?:不能|不得|不应当?|不要|不宜|严禁|禁止|避免|不属于|不建议|无需)'
                                 r'(?:再|直接)?(?:以|用|通过)?(?:提起|提出|进行|采取|申请|启动)?[ \t“"「]*$', prefix):
                    issues.append(dict(type='CRIMINAL_CIVIL_ROUTE_CONFUSION', detail='犯罪指控不能以民事反诉替代报案、调查取证或刑事辩护'))
    if criminal and not contextual and re.search(r'单位法务|教务处|学生申诉|员工手册|单位管理责任', body):
        issues.append(dict(type='UNRELATED_INSTITUTION_TEMPLATE', detail='无用工或校园背景的刑事/公共事件出现无关内部机构模板'))
    return issues


def _cited_texts(answer, materials, start, end):
    """返回这句话应当对照的材料正文。

    段落里写了材料编号，就只核这些编号。只有一条可用材料时，用这一条。
    没有可对照材料时返回空列表，调用方应按对不上处理。不改回答，也不访问网络。
    """
    by_id = {m['material_id']: m.get('text') or '' for m in materials}
    used_ids = [mid for mid in (parse_used_ids(answer) or []) if mid in by_id]
    all_body_refs = [mid for mid in find_citations(answer) if mid in by_id]
    fallback_ids = used_ids or all_body_refs
    paragraph_start = answer.rfind('\n', 0, start) + 1
    paragraph_end = answer.find('\n', end)
    paragraph = answer[paragraph_start:paragraph_end if paragraph_end >= 0 else len(answer)]
    refs = find_citations(paragraph)
    target_ids = sorted(set(refs) | set(fallback_ids)) if len(fallback_ids) == 1 else (refs if refs else fallback_ids)
    return [by_id.get(mid, '') for mid in target_ids]


def deadline_grounding_issues(answer, materials, *, question=""):
    """拦截所引材料中找不到的期限数值；只做词面对照，不核验法律适用。

    校验规则：
    1. 优先核对当前段落内直接标注的 [M1]/M1 等材料；
    2. 若当前段落未标注编号，回退核对回答末尾 USED_IDS 声明的合法材料（以及全文引用的合法材料）；
    3. 若期限数值直接来自用户提问原话（如用户自述“合同签了一年”、“约定六个月”），视为复述用户事实，不作虚构法定期限拦截。
    """
    from prototype.rag.grounding import numeric_value
    amount = r'(?:\d+(?:\.\d+)?|[一二三四五六七八九十百千万两]+|半)'
    unit = r'(?:个\s*)?(?:工作日|星期|季度|小时|个月|年|月|周|日|天)'
    token = rf'{amount}\s*{unit}'
    duration = re.compile(rf'({amount})\s*({unit})')
    def values(text):
        """提取并统一期限数值及单位，用于词面对照；相同数值不代表法律适用条件相同。"""
        result = set()
        for n, u in duration.findall(text or ''):
            normalized = re.sub(r'^个', '', re.sub(r'\s+', '', u))
            normalized = {'天': '日', '星期': '周'}.get(normalized, normalized)
            value = 0.5 if n == '半' else float(n) if re.fullmatch(r'\d+\.\d+', n) else numeric_value(n)
            # 半个月不换算为十五日；季度不换算为九十日，避免暗中引入历法假设。
            result.add((value, normalized))
        return result
    question_vals = values(question)
    issues=[]
    # 前缀与后缀均识别；工作日与自然日保持不同单位，不能相互作为依据。
    pattern = rf'(?:不得超过|不超过|至少(?:等待)?(?:满)?|(?:期限|通知期|周期|时限)(?:通常|一般)?[为是]|等待|提前)\s*{token}|(?:在\s*)?{token}(?:之内|以内|内|之前|以前|的通知期|的期限)'
    for match in re.finditer(pattern, answer or ''):
        needed = values(match.group()) - question_vals
        if not needed:
            continue
        if not any(needed <= values(text) for text in _cited_texts(answer, materials, match.start(), match.end())):
            issues.append({'type':'DEADLINE_VALUE_NOT_IN_CITED_TEXT',
                           'detail':'生成的期限在该句关联材料中无法核对，已拦截；不代表其他期限已经法律审查'})
    return issues


# 只认赔偿、价款、费用旁边的倍数，以及“退一赔三/十”。单独出现的“三倍”不当作法律标准。
_AMOUNT_MULTIPLE = re.compile(
    r'退一赔([三九十])'
    r'|([一二三四五六七八九十两\d]+)\s*倍(?:的)?(?:赔偿|价款)'
    r'|(?:惩罚性赔偿|增加赔偿|价款|费用).{0,12}?([一二三四五六七八九十两\d]+)\s*倍'
)
_AMOUNT_MONEY = re.compile(
    r'(?:赔偿|支付|退还|返还|补贴|金额(?:为|是)?|门槛(?:为|是)?|主张).{0,16}?'
    r'((?:\d+(?:\.\d+)?|[一二三四五六七八九十百千万两]+)\s*(?:万元|元))'
)
_MULTIPLE_DIGITS = {'2': '二', '3': '三', '5': '五', '10': '十'}


def _multiple_needles(digit):
    """倍数的可对照写法。3倍与三倍视为同一词面；不把一万换算成10000。"""
    needles = {f'{digit}倍'}
    mapped = _MULTIPLE_DIGITS.get(digit)
    if mapped:
        needles.add(f'{mapped}倍')
    if digit == '两':
        needles.add('二倍')
    if digit == '二':
        needles.add('两倍')
    return needles


def amount_grounding_issues(answer, materials, *, question=""):
    """拦截所引材料中对不上的赔偿倍数和确定金额；只做词面对照，不判断该不该赔。

    倍数是法律标准，即使用户问到“能不能要三倍”，也必须出现在所引材料里。
    用户原话里的具体金额（如“我付了3000元”）视为复述事实，不拦截。
    没有可对照材料时，这些确定说法同样拦截。
    """
    issues = []

    def reject(needles, start, end):
        texts = _cited_texts(answer, materials, start, end)
        if any(any(needle in text for needle in needles) for text in texts):
            return
        issues.append({'type': 'AMOUNT_VALUE_NOT_IN_CITED_TEXT',
                       'detail': '生成的赔偿倍数或确定金额在该句关联材料中无法核对，已拦截；不代表其他金额已经法律审查'})

    for match in _AMOUNT_MULTIPLE.finditer(answer or ''):
        digit = next(group for group in match.groups() if group)
        reject(_multiple_needles(digit), match.start(), match.end())
    for match in _AMOUNT_MONEY.finditer(answer or ''):
        token = re.sub(r'\s+', '', match.group(1))
        if token in (question or ''):
            continue
        reject({token}, match.start(), match.end())
    return issues


class SessionError(Exception):
    """可直接转换为接口提示的会话异常，包含HTTP状态码。"""
    def __init__(self, message="会话不存在或已过期，请重新开始", code=410):
        """初始化对象依赖与内部状态；后续读写通过本类的方法完成。"""
        super().__init__(message)
        self.code = code


def uncovered_local_rule_region(question, rows):
    """返回缺少标题覆盖的国内地方细则地域，未触发则返回 None；无读写或网络副作用。

    输入为本轮问题及当前全部语料。仅识别带省/市/区/县后缀的有限场景，不是行政区划库；
    标题出现地域只表示未触发本缺口检查，不认证该年规则完整、有效或适用。
    """
    # 全国反补贴规范中的“补贴”不能单独触发地方福利规则判断。
    # 排除明确拒绝采用的地域规则，保留逗号或“而是”之后真正请求的地区。
    local_request = re.sub(
        r"(?:不要|不用|不应|不得|无需|不能|不)(?:按(?:照)?|依据|依照|适用|参考|参照)"
        r"[^，。；;！？?\n]*?(?=[，。；;！？?\n]|而是|而要|但要|$)",
        "", question,
    ).replace("反补贴", "")
    if not re.search(r"补贴|实施细则", local_request):
        return None
    if not re.search(r"申请|条件|门槛|金额|期限|标准|多少|领取|资格|细则|规则", local_request):
        return None
    # 先分开请求动词与地名，避免把“我想申请上海市”整体误认为城市名称。
    clauses = re.split(
        r"[，。；;！？?\n]|按照|依据|根据|请按|按|请问|请以|以|位于|居住在|住在|孩子在|我在|在|申请|了解|关于|说明|告诉我",
        local_request,
    )
    for clause in clauses:
        for match in re.finditer(r"(?:[\u4e00-\u9fff]{2,7}?(?:省|市|区|县))+", clause):
            region = match.group()
            names = re.findall(r"[\u4e00-\u9fff]{2,7}?[省市区县]", region)
            specific = names[-1]
            if specific.endswith(("小区", "社区", "园区", "校区")) or specific in {
                "所在市", "所在区", "所在县", "所在地区", "所在城市", "本地区", "该地区",
            }:
                continue
            # 区县同名时还须核对用户明确提供的上级市，避免福州鼓楼区替代南京鼓楼区。
            # 省名可在标题中省略；市级或区县级名称必须在同一标题中出现。
            required = [specific] + [name for name in names[:-1] if name.endswith("市")]
            if not any(all(name in row.get("title", "") for name in required) for row in rows):
                return region
    return None


def uncovered_foreign_region(question, rows):
    """检查境外国别及国内地方细则缺口，保持原函数签名供服务与既有测试调用。

    返回尚未覆盖的地域名称或None；不认证带地域标签材料的完整性和法律适用性。
    明确要求按中国法回答时仍走国内流程，不因提到居住国外就一律拒绝。
    """
    # 明确排除中国法的子句不是选择中国法，先剔除有限否定表达再判断正向请求。
    positive_request = re.sub(r'(?:不要|不得|不应|不能|不用|不按|不适用|不依据|而非|不是|排除|无需)[^，。；;\n]{0,30}(?:中国法律|中国民法典|中国法)', '', question)
    if re.search(r'按(?:照)?中国|依据中国|中国法律|中国民法典', positive_request):
        return uncovered_local_rule_region(question, rows)
    names = {'美国': ('美国', '加利福尼亚', '加州'), '德国': ('德国', '柏林'),
             '英国': ('英国',), '法国': ('法国',), '日本': ('日本',), '加拿大': ('加拿大',),
             '澳大利亚': ('澳大利亚',), '新加坡': ('新加坡',), '境外': ('境外法律',)}
    for country, aliases in names.items():
        if any(alias in question for alias in aliases):
            if not any(row.get('jurisdiction') == country for row in rows):
                return country
    return uncovered_local_rule_region(question, rows)


@dataclass
class Session:
    """一段内存会话的历史、任务与有效状态；清空后旧任务不能写回。"""
    history: list = field(default_factory=list)
    jobs: dict = field(default_factory=dict)
    touched: float = field(default_factory=time.monotonic)
    valid: bool = True
    # 用户已发起的轮次（含被拦截/取消的轮次），用于归档“第N轮”和定向回查。
    turns: int = 0


class ChatService:
    """问答业务入口：隔离会话，控制并发/超时，组织材料与结果检查。"""
    def __init__(self, retriever, model, ttl=1800, timeout=180, *, official_catalog=None, official_fetcher=None,
                 session_dir=None):
        """注入本地检索/模型与可选官方目录、异步抓取器；保留原有位置参数。

        默认只读一次缓存的本地司法目录；显式空列表可关闭匹配。初始化不联网、不写盘。
        抓取器收到(entry, 合并用户问题)，须自行控制官方响应/重定向及访问限制；默认不配置。
        ``session_dir`` 是冷归档目录：默认 data/sessions（可用 LEGAL_RAG_SESSION_DIR 覆盖），
        传 False 可关闭归档；归档失败绝不影响问答主流程。
        """
        self.retriever = retriever
        self.model = model
        self.ttl = ttl
        self.timeout = timeout
        self.sessions = {}
        self.model_lock = asyncio.Lock()
        catalog = load_official_whitelist_catalog() if official_catalog is None else official_catalog
        if isinstance(catalog, dict):
            catalog = catalog.get('entries', [])
        self.official_catalog = tuple(dict(entry) for entry in catalog if isinstance(entry, dict))
        self.official_fetcher = official_fetcher
        self.session_dir = session_dir

    async def _official_supplements(self, question, query, hits, mode):
        """仅定位明确未入库专项，返回补充切片、来源记录及缺口说明；不发布或写入本地库。

        高分且无缺失专项时只作内存目录判断，不调用抓取器。最多匹配两项，每项抓取限时10秒；
        抓取失败/拒绝返回只保留目录定位，异常载荷不对外暴露；取消仍由外层正常传播。
        """
        titles = {row.get('title', '') for row in getattr(self.retriever, 'rows', [])}
        entries = match_official_whitelist_catalog(question, self.official_catalog, titles, top_k=2)
        records, supplements, gaps = [], [], []
        for entry in entries:
            record = dict(entry, trigger=_official_trigger(hits, mode), sent_to_model=False, material_ids=[], article_sources=[])
            if any(entry[key] and entry[key] > date.today().isoformat() for key in ('publication_date', 'effective_date')):
                # 离线盘点可提示待施行新版；在线不能因目录sxx标记而把未来版本当作现行依据。
                record['date_not_current'] = True
                gaps.append(f"本地库暂未收录《{entry['title']}》全文；已定位官方白名单目录，但公布/施行日期尚未到，仅供维护核对，不作为现行条文补充。")
                records.append(record)
                continue
            supplemented = None
            if self.official_fetcher is not None:
                try:
                    # 入口先校验，返回后再检查实际来源/条号；适配器不能把任意网址带入材料。
                    public_url(entry['official_url'], allow_government=True)
                    async with asyncio.timeout(10):
                        row = await self.official_fetcher(dict(entry), query)
                    supplemented = _official_slice(entry, row)
                except Exception:
                    # 抓取失败不破坏原本地问答，不输出含问题内容或网络载荷的异常文本。
                    supplemented = None
            if supplemented is not None:
                supplements.append(supplemented)
                record['article_sources'] = [supplemented['source_url']]
                record['locator'] = supplemented['locator']
                gaps.append(f"《{entry['title']}》仅本轮补充官方白名单条文切片，全文未收录，未独立复核个案适用。")
            else:
                gaps.append(f"本地库暂未收录《{entry['title']}》全文；已从官方白名单目录定位到对应条目供核对，未从非官方站点抓取。")
            records.append(record)
        return supplements, records, gaps

    async def _display_candidates(self, query):
        """取页面展示用的前8条和"更多候选"，只供展示、不送模型；失败时返回空列表，不影响回答。

        前8条单独调用 search(top_k=8)：复合问题的锚点重排随 top_k 变化，不能从前4推出前8。
        保存并恢复主检索的 state/error，展示查询的成功、内部降级、外抛异常或取消都不覆盖它们。
        尾部失败时仍保留前8条；检索器没有 tail_candidates() 时尾部为空。
        """
        # _run 在 model_lock 内依次做主检索和展示检索；这里只保护已有状态，兼容无状态的旧检索器。
        main_state = {name: getattr(self.retriever, name) for name in ("state", "error")
                      if hasattr(self.retriever, name)}
        try:
            top, _ = await self.retriever.search(query, top_k=DISPLAY_TOP_K)
        except Exception:
            return [], []
        finally:
            # search 会把网络故障转成正常返回的关键词结果，所以不能只在 except 中恢复。
            for name, value in main_state.items():
                setattr(self.retriever, name, value)
        tail_method = getattr(self.retriever, "tail_candidates", None)
        try:
            tail = await tail_method(query, {row.get("id") for row in top}) if tail_method else []
        except Exception:
            tail = []
        return list(top), list(tail)

    def create(self):
        """创建内存会话并限制会话总数，避免长期打开页面无限占用资源。"""
        if len(self.sessions) >= 20:
            self.cleanup()
        if len(self.sessions) >= 20:
            raise SessionError("会话数量已达上限，请先清空旧会话", 429)
        sid = secrets.token_urlsafe(32)
        self.sessions[sid] = Session()
        return sid

    def get(self, sid, exempt=False):
        """取得有效会话；不存在或闲置过期时拒绝访问，不把访问当作自动恢复。"""
        s = self.sessions.get(sid)
        if not s or not s.valid:
            raise SessionError()
        active = any(j["status"] == "running" for j in s.jobs.values())
        if not exempt and not active and time.monotonic() - s.touched >= self.ttl:
            s.valid = False
            self.sessions.pop(sid, None)
            raise SessionError()
        return s

    def start(self, sid, question):
        """校验并启动本轮异步问答，返回任务编号；实际计算由后台流程完成。

        用户提问同时排入冷归档并立即开始检索/生成；归档失败只跳过归档，不阻断回答。
        """
        s = self.get(sid)
        if any(j["status"] == "running" for j in s.jobs.values()):
            raise SessionError("上一轮仍在处理，请等待或停止", 409)
        rid = secrets.token_urlsafe(18)
        s.touched = time.monotonic()
        # 轮次含被拦截/取消的提问，保证用户口中的“第N轮”与页面一致。
        s.turns += 1
        round_no = s.turns
        job = {"status": "running", "phase": "正在检索本地资料", "created": s.touched}
        s.jobs[rid] = job
        # 限制每个会话保留的任务结果数量；聊天正文另走本地冷归档。
        for old in list(s.jobs)[:-4]:
            if s.jobs[old]["status"] != "running":
                del s.jobs[old]
        job["task"] = asyncio.create_task(self._archive_and_run(sid, s, rid, question, round_no))
        return rid

    def _archive_enabled(self):
        """是否启用冷归档；False 只用于需要完全离线/纯内存的调用方。"""
        return self.session_dir is not False

    async def _archive_user(self, sid, round_no, question):
        """把用户原话追加到冷归档；任何磁盘错误都只跳过，不影响本轮问答。"""
        if not self._archive_enabled():
            return
        try:
            await session_memory.append_event(
                sid, session_memory.archive_user_event(round_no, question), self.session_dir)
        except Exception:
            pass

    async def _archive_answer(self, sid, round_no, result=None, status="done", blocked=False,
                              blocking_issues=None, error=None):
        """把回答或错误追加到冷归档；拦截原因、原始输出与实际送模材料一并保留。"""
        if not self._archive_enabled():
            return
        result = result or {}
        materials_sent = [{"material_id": m.get("material_id"), "title": m.get("title"),
                           "locator": m.get("locator")} for m in result.get("retrieved") or []]
        event = session_memory.archive_answer_event(
            round_no, raw_answer=result.get("raw_answer", ""), answer=result.get("answer", ""),
            blocked=blocked, blocking_issues=blocking_issues or (result.get("warnings") or []),
            materials_sent=materials_sent, status=status)
        if error:
            event["error"] = error
        try:
            await session_memory.append_event(sid, event, self.session_dir)
        except Exception:
            pass

    async def _archive_and_run(self, sid, s, rid, question, round_no):
        """用户提问先排入归档线程，同时立刻开始本轮；写回答前再等它落完，保证文件顺序。"""
        user_archive = asyncio.create_task(self._archive_user(sid, round_no, question))
        await self._run(sid, s, rid, question, round_no, user_archive=user_archive)

    async def _recall_history(self, sid, s, question, round_no):
        """取出与本轮相关的远期用户原话；只查用户发言，助手回答永不进入回查索引。

        优先读冷归档（含被拦截轮次），读取失败或为空时退回内存历史。
        最近一轮已由历史视图承担，这里只补更早的原话；没有明显指代或相关度不足时返回空。
        """
        if round_no <= 1:
            return []
        turns = []
        if self._archive_enabled():
            try:
                turns = await asyncio.to_thread(session_memory.read_user_turns, sid, self.session_dir)
            except Exception:
                turns = []
        if not turns:
            turns = session_memory.turns_from_history(s.history)
        return session_memory.recall(question, turns, current_round=round_no)

    def delete_archive(self, sid):
        """删除该会话的本地冷归档；只动 data/sessions 下对应文件，不影响内存会话。"""
        if not self._archive_enabled():
            return False
        try:
            return session_memory.delete_session_archive(sid, self.session_dir)
        except Exception:
            return False

    async def _run(self, sid, s, rid, question, round_no=None, *, user_archive=None):
        """执行一轮问答：检索→预算内组装材料→本机推理→检查→返回；仅有效且未取消的会话可写回。"""
        job = s.jobs[rid]
        started = time.monotonic()

        async def settle_user_archive():
            """等用户提问先落盘，保证同一文件里用户条排在回答条之前；失败已由写入方吞掉。"""
            if user_archive is not None:
                try:
                    await user_archive
                except Exception:
                    pass
        try:
            async with asyncio.timeout(self.timeout):
                async with self.model_lock:
                    if not s.valid:
                        return
                    # 第一步：用用户历史陈述补足检索问题，不把模型上次的推测当成事实。
                    prior = " ".join(m["content"] for m in s.history if m["role"] == "user")[-800:]
                    query = (prior + " " + question).strip()
                    unsupported = any(w in question for w in ("刑事", "判刑", "杀人", "盗窃", "诈骗", "离婚", "遗产", "行政复议", "行政诉讼"))
                    labor = any(w in question for w in ("工资", "工伤", "劳动", "老板", "公司", "社保", "辞退", "上班"))
                    covered = {r.get('field') for r in getattr(self.retriever, 'rows', [])}
                    # 领域判断只用于覆盖提示，不代表法律性质鉴定。
                    out_of_scope = unsupported and not labor and not covered.intersection({'CRIMINAL','CIVIL','ADMINISTRATIVE'})
                    hits, mode = ([], "uncovered") if out_of_scope else await self.retriever.search(query)
                    # 仍执行真实检索；缺少地域规则时在材料组装前清空候选，防止无关条例充当依据。
                    scope_region = uncovered_foreign_region(question, getattr(self.retriever, 'rows', []))
                    scope_degraded = scope_region is not None
                    if scope_degraded:
                        out_of_scope = True
                        hits = []
                    # 展示候选只给页面看，不进入下面的材料组装；送模型的仍是上面这次默认检索的结果。
                    # 领域或地域未覆盖时不展示，也不多做检索。
                    display_top, display_tail = ([], []) if out_of_scope else await self._display_candidates(query)
                    official_fallback, official_gaps = [], []
                    if not scope_degraded:
                        # 全国目录不能绕过境外/地方细则硬降级；明确缺失专项可补足弱命中或同主题高分缺口。
                        supplements, official_fallback, official_gaps = await self._official_supplements(question, query, hits, mode)
                        hits = list(hits) + supplements
                        if supplements:
                            out_of_scope = False
                    # 第二步：限制材料预算，整条选择，不把截断条文当作完整依据。
                    picked = []
                    count = 0
                    for row in hits:
                        if count + len(row["text"]) <= 3600:
                            picked.append(row)
                            count += len(row["text"])
                    candidates = [Material(material_id=f"M{i+1}", kind=r["kind"], title=r["title"],
                                           locator=r["locator"], version_id=r["version_id"], text=r["text"],
                                           source_url=r["source_url"], verified_scope=r["verified_scope"],
                                           content_scope=r["content_scope"],
                                           excerpts=split_source_excerpts(r["text"]))
                                  for i, r in enumerate(picked)]
                    gaps = ["仅覆盖已下载入库的规范和劳动案例摘要，未覆盖全部法规、地方规定与类案。",
                            "相似检索不代表适用；版本适用、依据资格和事实构成尚未核实。"]
                    gaps.extend(official_gaps)
                    if out_of_scope:
                        gaps.append("该问题所属领域尚无知识资料，只能整理事实与待核实事项，不提供确定法律结论或期限。")
                    if not candidates:
                        gaps.append("本次没有找到可核验的直接材料，不能编造引用或凭记忆列法条。")
                    # 第三步：用UTF-8字节数保守控制输入预算，为生成和模板预留空间。
                    # 650字是输出篇幅建议；新增系统提示仍计入输入预算。
                    # 本模式只允许可核对的原句、原文要素问句与通用保存/核实步骤，不要求模型自由判案。
                    # 许可在原始引用、期限及金额检查之后整份核对；历史补充也不解锁个案结论。
                    # 有候选材料时切换为“程序切句、模型选号、程序回填原文”的选择提示；
                    # 无材料仍用原阅读提示，保持 no_materials 单轮快照逐字节不变。
                    system = MATERIAL_SELECTION_SYSTEM if candidates else MATERIAL_READING_SYSTEM
                    if out_of_scope:
                        system += "当前资料未覆盖所问地域或领域，必须明确提示未覆盖、无法确认具体申请条件、收入门槛、金额或期限；不得输出任何具体法条条号、境外法律依据或法定期限。只整理待核实事项，USED_IDS: 无。"

                    def size(msgs):
                        """用UTF-8字节数保守估计输入预算，防止超长历史或材料被引擎静默截断。"""
                        return sum(len(m["content"].encode("utf-8")) + 32 for m in msgs)

                    def assemble(ms, history_view, gap_list, asked):
                        """把系统规则、压缩后的历史视图、远期原话和本轮材料装成模型输入；历史不作为法律依据。"""
                        packet = RagInput(question=asked, materials=ms, gaps=gap_list,
                                          coverage_note="已收录资料；核验仅限本地快照",
                                          recalled=recalled_lines)
                        return ([{"role": "system", "content": system}] + list(history_view)
                                + [{"role": "user", "content": packet.render_user_prompt()}])

                    def fit_materials(gap_list, budget, asked):
                        """材料第一顺位：按原规则整条装入；历史为空时与改造前逐字节一致。"""
                        kept = []
                        for material in candidates:
                            if size(assemble(kept + [material], [], gap_list, asked)) <= budget:
                                kept.append(material)
                        return kept

                    def fit_history(gap_list, kept_materials, asked):
                        """余量倒序装历史：从最新一条往前装，装不下就停；不留孤立的助手消息。"""
                        kept = []
                        for message in reversed(history_notes):
                            trial = [message] + kept
                            if size(assemble(kept_materials, trial, gap_list, asked)) > PROMPT_BUDGET:
                                break
                            kept = trial
                        if kept and kept[0]["role"] != "user":
                            kept = kept[1:]
                        return kept

                    def history_budget():
                        """有历史时先给最近一轮留出的余量；历史为空时就是完整 6800。"""
                        return PROMPT_BUDGET - (HISTORY_RESERVE if history_notes else 0)

                    # 远期原话回查：只检索用户发言，助手回答永不进入索引。
                    # 当前轮已写入归档但要排除；最近一轮已由历史视图承担，这里只补更早的原话。
                    # 命中最多两条，单片≤200字节、合计≤500字节，放在本轮用户消息里，不进 system。
                    recalled = await self._recall_history(sid, s, question, round_no or 0)
                    recalled_lines = [item["line"] for item in recalled]
                    recalled_user_text = " ".join(item["content"] for item in recalled)
                    # 历史为空时 budget 就是 6800，材料选择结果与改造前完全相同。
                    history_notes = _compact_history(s.history)
                    asked = question
                    fitting = fit_materials(gaps, history_budget(), asked)
                    if len(fitting) != len(candidates):
                        # 缺口文案本身也占字节：先定稿再重排一次，避免“先选后加”把总字节顶破。
                        gaps = gaps + [BUDGET_GAP_TEXT]
                        if history_notes:
                            # 单轮保持改造前“先定材料、后加缺口”的顺序，逐字节不变；
                            # 只有多轮历史在场时才重排，因为那时才需要严格守住 6800。
                            fitting = fit_materials(gaps, history_budget(), asked)
                    history_view = fit_history(gaps, fitting, asked) if history_notes else []
                    # 双侧保底：确实放不下最近一轮时，从名次最低的材料回吐，但至少保留一条完整材料。
                    while not history_view and history_notes and len(fitting) > 1:
                        fitting = fitting[:-1]
                        if BUDGET_GAP_TEXT not in gaps:
                            gaps = gaps + [BUDGET_GAP_TEXT]
                            fitting = fit_materials(gaps, history_budget(), asked)
                        history_view = fit_history(gaps, fitting, asked)
                    # 第四步：硬顶兜底。任何极端输入都走降级，绝不向外抛超长异常。
                    messages = assemble(fitting, history_view, gaps, asked)
                    if size(messages) > MESSAGE_LIMIT:
                        history_view = []
                        messages = assemble(fitting, history_view, gaps, asked)
                    if size(messages) > MESSAGE_LIMIT:
                        # 只有问题本身超长时才会走到这里：先截断送模副本，再按新长度重选材料，
                        # 让材料保底优先于“完整问题”；检索与金额/期限豁免仍用完整问题。
                        asked = (_truncate_bytes(question, QUESTION_PROMPT_LIMIT)
                                 + "……（问题过长，本轮只按前面这段分析）")
                        gaps = gaps + [LONG_QUESTION_GAP_TEXT]
                        fitting = fit_materials(gaps, history_budget(), asked)
                        if len(fitting) != len(candidates) and BUDGET_GAP_TEXT not in gaps:
                            gaps = gaps + [BUDGET_GAP_TEXT]
                        history_view = fit_history(gaps, fitting, asked) if history_notes else []
                        messages = assemble(fitting, history_view, gaps, asked)
                    if size(messages) > MESSAGE_LIMIT:
                        fitting = fitting[:1]
                        history_view = []
                        messages = assemble(fitting, history_view, gaps, asked)
                    if size(messages) > MESSAGE_LIMIT:
                        # 理论上到不了这里（问题已截到600字节）；真到了也绝不崩溃，退为无材料追问。
                        fitting = []
                        messages = assemble(fitting, [], gaps, asked)
                    materials = fitting
                    # 页面展示时标明哪些条文真正送入了模型：按法名、条号、版本对应到本轮材料编号。
                    sent = {(m.title, m.locator, m.version_id): m.material_id for m in materials}
                    display = dict(top=[_display_row(row, sent) for row in display_top],
                                   tail=[_display_row(row, sent, routes=row.get("tail_routes")) for row in display_tail])
                    # 抓到切片不等于送模成功：字符/字节预算可能整条排除，只记录真正提供的编号。
                    for source in official_fallback:
                        source['material_ids'] = [m.material_id for m in materials
                                                  if m.source_url in source['article_sources']
                                                  and normalize_catalog_title(m.title) == normalize_catalog_title(source['title'])
                                                  and m.locator == source.get('locator')]
                        source['sent_to_model'] = bool(source['material_ids'])
                    job["phase"] = "本机模型正在分析，首次加载可能较慢"
                    raw = await self.model.chat(messages)
                    # 模型只负责“选号/选短语”（或仍输出旧阅读格式）；原文、标题和固定步骤
                    # 由程序从 materials 内存原文回填。装配结果再走下面同一套原校验器。
                    final_answer, assembly = build_final_answer(raw, materials, question=question)
                    material_rows = _material_rows(materials)
                    checked = postprocess(final_answer, material_rows)
                    scenario_issues = answer_scenario_issues(question, final_answer)
                    if scenario_issues:
                        checked['blocking'] = True
                        checked['blocking_issues'].extend(scenario_issues)
                    # 仍保留并先校验最终回答；仅将唯一绑定成功后的声明补齐文本用于期限取证，绝不改期限正文。
                    checked_answer = checked['normalized_answer']
                    # 校验器豁免上下文 = 本轮问题 + 当轮实际注入的远期原话。
                    # 这样用户第 1 轮说“借了5万元”、第 5 轮如实复述时，不会被当成捏造数据。
                    context_question = (question + (" " + recalled_user_text if recalled_user_text else "")).strip()
                    duration_issues = deadline_grounding_issues(checked_answer, material_rows, question=context_question)
                    amount_issues = amount_grounding_issues(checked_answer, material_rows, question=context_question)
                    # 场景与主体判断仍只看本轮问题，不能被旧原话带偏。
                    permission_issues = material_reading_issues(final_answer, material_rows)
                    if duration_issues or amount_issues or permission_issues:
                        checked['blocking'] = True
                        checked['blocking_issues'].extend(duration_issues + amount_issues + permission_issues)
                    cards = checked["citation_cards"]
                    scopes = {m.material_id:m.verified_scope for m in materials}
                    for card in cards:
                        card['verified_scope'] = scopes.get(card.get('material_id'), '核验范围待确认')
                    if scope_degraded:
                        # 最终答复由覆盖事实生成，不呈现模型猜测；真实推理和安全告警仍完整保留。
                        # blocked描述最终答复是否被阻断，raw_output_blocked另记原始模型的拦截事实。
                        answer = (f"当前资料未覆盖{scope_region}对应的地域规则、专门实施细则或补贴规则，已严格降级。"
                                  "无法确认具体申请条件、收入门槛、金额或期限，也无法确认该地域的具体法律结论或条款。"
                                  "请补充所问地域、事项及已取得的官方规则材料，再核实适用条件。")
                        cards = []
                        # 与评测共用最终降级判定；原始模型是否被拦截单独留存，不冒充最终答复失败。
                        if not judge_degrade(answer, False, [scope_region]):
                            raise RuntimeError("地域范围提示未通过降级校验，请补充官方规则材料后重试")
                    elif checked["blocking"]:
                        # 第四步：引用、期限或金额检查失败时显示拦截提示，原始输出单独保留待核对。
                        answer = "这次生成超出材料阅读范围，或包含无法核对的引用、期限、金额或处理路径，已拦截。请查看检索候选原文，补充事实后重试。"
                        cards = []
                    else:
                        # 最终回答（可能是程序回填的原文）已完成引用与期限校验；只清理展示文本。
                        answer = clean_display_answer(final_answer, materials)
                    result = {"answer": answer, "blocked": checked["blocking"] and not scope_degraded, "raw_answer": raw,
                              # 校验器实际看到的装配文本；只用于审计/评测，不改展示稿。
                              "final_answer": final_answer,
                              "raw_output_blocked": checked['blocking'], "scope_degraded": scope_degraded,
                              "declaration_repair": checked['declaration_repair'],
                              "warnings": checked["blocking_issues"] + checked["suspected_issues"],
                              "citations": cards, "retrieved": material_rows, "display": display,
                              "gaps": gaps, "retrieval_mode": mode, "elapsed": round(time.monotonic()-started, 2),
                              "official_fallback": official_fallback, "answer_assembly": assembly,
                              "answer_mode": "material_reading", "review_note": "材料阅读；原文节选及核实线索不认定个案适用，待核实事项并不穷尽。"}
                    if s.valid and self.sessions.get(sid) is s and job["status"] == "running":
                        job.update(status="done", result=result)
                        if not checked["blocking"]:
                            s.history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
                        # 归档忠实保留原始输出、拦截原因和实际送模材料；被拦截也照记，方便事后核对。
                        await settle_user_archive()
                        await self._archive_answer(sid, round_no or s.turns, result, status="done",
                                                   blocked=result["blocked"],
                                                   blocking_issues=result["warnings"])
        except asyncio.CancelledError:
            job["status"] = "cancelled"
            raise
        except TimeoutError:
            job.update(status="error", error="本机处理超时，已停止；可缩短问题或更换更小模型，没有转用云端")
            await settle_user_archive()
            await self._archive_answer(sid, round_no or s.turns, status="error", blocked=True, error=job["error"])
        except Exception as exc:
            # Avoid exposing network payloads, full prompts or private data in error messages/logs.
            message = str(exc) if isinstance(exc, RuntimeError) else "本机服务暂不可用，请检查模型和资料状态；没有转用云端"
            job.update(status="error", error=message)
            await settle_user_archive()
            await self._archive_answer(sid, round_no or s.turns, status="error", blocked=True, error=message)
        finally:
            if s.valid:
                s.touched = time.monotonic()

    def result(self, sid, rid):
        """读取指定会话中的任务状态与结果，不能跨会话读取其他任务。"""
        s = self.get(sid)
        if rid not in s.jobs:
            raise SessionError("请求不存在", 404)
        return {k: v for k, v in s.jobs[rid].items() if k != "task"}

    async def cancel(self, sid, rid):
        """请求取消指定任务并等待清理；不把未完成回答写入聊天历史。"""
        s = self.sessions.get(sid)
        if s and rid in s.jobs:
            job = s.jobs[rid]
            if job["status"] == "running":
                job["status"] = "cancelled"
                job["task"].cancel()
                await asyncio.gather(job["task"], return_exceptions=True)

    async def reset(self, sid):
        """清空旧会话并创建新会话；先作废旧对象，防止在途任务完成后晚写回。"""
        s = self.sessions.pop(sid, None)
        if s:
            s.valid = False
            for rid in list(s.jobs):
                job = s.jobs[rid]
                if job["status"] == "running":
                    job["status"] = "cancelled"
                    job["task"].cancel()
            await asyncio.gather(*(j["task"] for j in s.jobs.values()), return_exceptions=True)
            s.history.clear()
            s.jobs.clear()
        return self.create()

    def cleanup(self):
        """移除已闲置过期且没有运行任务的会话，不打断仍在处理的请求。"""
        for sid, s in list(self.sessions.items()):
            if (time.monotonic() - s.touched >= self.ttl
                    and not any(j["status"] == "running" for j in s.jobs.values())):
                s.valid = False
                self.sessions.pop(sid, None)
